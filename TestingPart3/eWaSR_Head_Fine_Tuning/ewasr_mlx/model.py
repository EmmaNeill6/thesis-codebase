"""Trainable MLX port of the non-IMU eWaSR ResNet-18 model.

The architecture and parameter names intentionally match the validated MLX
inference port in ``TestingPart1/eWaSR``.  For the first adaptation experiment,
all parameters are frozen except ``decoder.seg_head``.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def _apply_layers(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def _resize(x, target_hw, mode):
    """Resize an NHWC tensor to an integer multiple of its current size."""

    height, width = x.shape[1:3]
    target_height, target_width = target_hw
    if (height, width) == (target_height, target_width):
        return x
    if target_height % height or target_width % width:
        raise ValueError(
            f"MLX resize requires integer scaling, got {(height, width)} -> "
            f"{(target_height, target_width)}"
        )
    scale = (target_height // height, target_width // width)
    return nn.Upsample(scale_factor=scale, mode=mode, align_corners=False)(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels)
        self.downsample = (
            [
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm(out_channels),
            ]
            if stride != 1 or in_channels != out_channels
            else None
        )

    def __call__(self, x):
        identity = x
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = _apply_layers(self.downsample, identity)
        return nn.relu(out + identity)


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = [BasicBlock(64, 64), BasicBlock(64, 64)]
        self.layer2 = [BasicBlock(64, 128, 2), BasicBlock(128, 128)]
        self.layer3 = [BasicBlock(128, 256, 2), BasicBlock(256, 256)]
        self.layer4 = [BasicBlock(256, 512, 2), BasicBlock(512, 512)]

    def __call__(self, x):
        x = self.maxpool(nn.relu(self.bn1(self.conv1(x))))
        skip1 = _apply_layers(self.layer1, x)
        skip2 = _apply_layers(self.layer2, skip1)
        aux = _apply_layers(self.layer3, skip2)
        out = _apply_layers(self.layer4, aux)
        return out, aux, skip2, skip1


class AttentionRefinementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.bn1 = nn.BatchNorm(channels)

    def __call__(self, x):
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        weights = mx.sigmoid(self.bn1(self.conv1(pooled)))
        return weights * x


class ARMMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.arm = AttentionRefinementModule(dim)

    def __call__(self, x):
        return self.arm(x)


class SpatialAttentionMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def __call__(self, x):
        average = mx.mean(x, axis=-1, keepdims=True)
        maximum = mx.max(x, axis=-1, keepdims=True)
        attention = mx.sigmoid(self.conv(mx.concatenate([average, maximum], axis=-1)))
        return attention * x


class Mlp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_channels = channels * 4
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1)
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class MetaFormerBlock(nn.Module):
    def __init__(self, channels, mixer):
        super().__init__()
        self.layer_scale_1 = mx.ones((channels,)) * 1e-5
        self.layer_scale_2 = mx.ones((channels,)) * 1e-5
        self.norm1 = nn.BatchNorm(channels)
        self.token_mixer = (
            ARMMixer(channels) if mixer == "arm" else SpatialAttentionMixer()
        )
        self.norm2 = nn.BatchNorm(channels)
        self.mlp = Mlp(channels)

    def __call__(self, x):
        scale1 = self.layer_scale_1.reshape((1, 1, 1, -1))
        scale2 = self.layer_scale_2.reshape((1, 1, 1, -1))
        x = x + scale1 * self.token_mixer(self.norm1(x))
        return x + scale2 * self.mlp(self.norm2(x))


class SIM(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv_lc = nn.Conv2d(in_channels, out_channels, 1)
        self.bn_lc = nn.BatchNorm(out_channels)
        self.conv_gc = nn.Conv2d(in_channels, out_channels, 1)
        self.bn_gc = nn.BatchNorm(out_channels)
        self.conv_gc1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn_gc1 = nn.BatchNorm(out_channels)

    def __call__(self, local, global_):
        global_ = _resize(global_, local.shape[1:3], "nearest")
        local = self.bn_lc(self.conv_lc(local))
        multiplier = mx.sigmoid(self.bn_gc(self.conv_gc(global_)))
        global_ = self.bn_gc1(self.conv_gc1(global_))
        return local * multiplier + global_


class SegHead(nn.Module):
    def __init__(self, channels=256, num_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm(channels)
        self.conv1 = nn.Conv2d(channels, num_classes, 1)

    def __call__(self, x):
        x = mx.clip(self.bn(self.conv(x)), 0, 6)
        return self.conv1(x)


class EWaSRDecoder(nn.Module):
    channels = (512, 256, 128, 64)

    def __init__(self):
        super().__init__()
        total_channels = sum(self.channels)
        self.metaformers = [
            MetaFormerBlock(total_channels, "arm") for _ in range(4)
        ] + [MetaFormerBlock(total_channels, "spatial") for _ in range(2)]
        self.metaformers_skip2 = [
            MetaFormerBlock(self.channels[2], "spatial") for _ in range(2)
        ]
        self.sim1 = SIM(self.channels[0])
        self.sim2 = SIM(self.channels[1])
        self.sim3 = SIM(self.channels[2])
        self.sim4 = SIM(self.channels[3])
        self.seg_head = SegHead()

    @staticmethod
    def _pool_features(inputs):
        target_height = (inputs[0].shape[1] - 1) // 2 + 1
        target_width = (inputs[0].shape[2] - 1) // 2 + 1
        pooled = []
        for value in inputs:
            height, width = value.shape[1:3]
            if height % target_height or width % target_width:
                raise ValueError(
                    "Input dimensions must be divisible by 64 for pyramid pooling"
                )
            kernel = (height // target_height, width // target_width)
            pooled.append(nn.AvgPool2d(kernel, stride=kernel)(value))
        return mx.concatenate(pooled, axis=-1)

    def __call__(self, out, aux, skip2, skip1):
        tokens = self._pool_features([out, aux, skip2, skip1])
        tokens = _apply_layers(self.metaformers, tokens)
        skip2 = _apply_layers(self.metaformers_skip2, skip2)

        boundaries = (512, 768, 896)
        f1 = self.sim1(out, tokens[..., : boundaries[0]])
        f2 = self.sim2(aux, tokens[..., boundaries[0] : boundaries[1]])
        f3 = self.sim3(skip2, tokens[..., boundaries[1] : boundaries[2]])
        f4 = self.sim4(skip1, tokens[..., boundaries[2] :])

        target_hw = f4.shape[1:3]
        fused = (
            _resize(f1, target_hw, "linear")
            + _resize(f2, target_hw, "linear")
            + _resize(f3, target_hw, "linear")
            + f4
        )
        return self.seg_head(fused)


class EWaSR(nn.Module):
    """ResNet-18 eWaSR with NHWC tensors and no IMU input."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.decoder = EWaSRDecoder()

    def __call__(self, image):
        return self.decoder(*self.backbone(image))

    def configure_head_only_training(self):
        """Freeze the encoder/refinement path and train only the segmentation head.

        Frozen BatchNorm layers stay in evaluation mode.  The head's BatchNorm
        layer uses training statistics while its running statistics remain
        non-gradient state, as required by MLX BatchNorm.
        """

        self.freeze()
        self.decoder.seg_head.unfreeze()
        # Recursive unfreeze clears a child's local no-gradient set in MLX, so
        # explicitly restore BatchNorm running statistics as non-gradient state.
        self.decoder.seg_head.bn.freeze(
            keys=["running_mean", "running_var"], recurse=False, strict=True
        )
        self.eval()
        self.decoder.seg_head.train()
        return self

    def set_head_training_mode(self, training: bool):
        """Switch only the head between train/evaluation modes."""

        self.eval()
        if training:
            self.decoder.seg_head.train()
        return self


def create_head_finetuning_model(weights_path: Path) -> EWaSR:
    """Load the pretrained checkpoint and expose only head parameters to autograd."""

    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained eWaSR weights not found: {weights_path}")
    model = EWaSR()
    model.load_weights(str(weights_path))
    model.configure_head_only_training()
    mx.eval(model.parameters())
    return model
