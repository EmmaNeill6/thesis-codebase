import os
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import train_head_sim4 as experiment


class HeadSim4Tests(unittest.TestCase):
    def test_only_sim4_and_head_affine_parameters_are_trainable(self):
        names = [
            "backbone.conv1.weight",
            "decoder.sim1.conv_lc.weight",
            "decoder.sim2.conv_lc.weight",
            "decoder.sim3.conv_lc.weight",
            "decoder.sim4.conv_lc.weight",
            "decoder.sim4.bn_lc.weight",
            "decoder.sim4.bn_lc.running_mean",
            "decoder.sim4.bn_lc.running_var",
            "decoder.seg_head.conv.weight",
            "decoder.seg_head.bn.weight",
            "decoder.seg_head.bn.running_mean",
        ]
        allowed = ["decoder.sim4.conv_lc.weight", "decoder.sim4.bn_lc.weight",
                   "decoder.seg_head.conv.weight", "decoder.seg_head.bn.weight"]
        experiment.check_trainable_names(allowed, names)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            experiment.check_trainable_names(allowed + ["decoder.sim3.conv_lc.weight"], names)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            experiment.check_trainable_names(allowed + ["decoder.sim4.bn_lc.running_mean"], names)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            experiment.check_trainable_names(allowed[:-1], names)

    def test_frozen_checkpoint_includes_other_sims_and_sim4_running_stats(self):
        original = {
            "backbone.conv1.weight": np.array([1], dtype=np.float32),
            "decoder.sim1.conv_lc.weight": np.array([2], dtype=np.float32),
            "decoder.sim4.bn_lc.running_mean": np.array([3], dtype=np.float32),
            "decoder.sim4.conv_lc.weight": np.array([4], dtype=np.float32),
            "decoder.seg_head.conv.weight": np.array([5], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before, after = root / "before.npz", root / "after.npz"
            np.savez(before, **original)
            updated = {**original,
                       "decoder.sim4.conv_lc.weight": np.array([8], dtype=np.float32),
                       "decoder.seg_head.conv.weight": np.array([9], dtype=np.float32)}
            np.savez(after, **updated)
            audit = experiment.audit_frozen_checkpoint(before, after)
            self.assertTrue(audit["frozen_tensors_unchanged"])
            for frozen_name in ("decoder.sim1.conv_lc.weight",
                                "decoder.sim4.bn_lc.running_mean"):
                changed = {**updated, frozen_name: np.array([99], dtype=np.float32)}
                np.savez(after, **changed)
                with self.assertRaisesRegex(RuntimeError, "Frozen pretrained tensors changed"):
                    experiment.audit_frozen_checkpoint(before, after)

    def test_reuses_recorded_v3_roles_even_when_definition_boundaries_differ(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "pretrained.npz"
            checkpoint.write_bytes(b"unused in data check")
            samples = {}
            for domain in ("canal", "pier"):
                images, masks = root / domain / "images", root / domain / "masks"
                images.mkdir(parents=True)
                masks.mkdir()
                for index in (1, 2):
                    image = images / f"{index:03d}.png"
                    mask = masks / f"{index:03d}.png"
                    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
                    value = 255 if (domain, index) == ("pier", 2) else 1
                    Image.fromarray(np.full((8, 8), value, dtype=np.uint8)).save(mask)
                    samples[(domain, index)] = {"image": str(image), "mask": str(mask)}
            # The read-only definition names the same source universe. Its
            # selection differs intentionally from the recorded V3 roles.
            roles = {"train": ("canal", 2), "validation": ("pier", 1),
                     "test": ("canal", 1), "unseen": ("pier", 2)}
            definition = {
                "schema_version": 1,
                "labels": {"obstacle": 0, "water": 1, "sky": 2},
                "model": {"checkpoint": str(checkpoint)},
                "data": {role: {"sources": [{
                    "source_id": f"{domain}_{role}",
                    "images_dir": str(root / domain / "images"),
                    "masks_dir": str(root / domain / "masks"),
                    "selection": {"kind": "inclusive_range", "start": 2, "end": 2},
                }]} for role, (domain, _) in roles.items()},
            }
            definition_path = root / "definition.yaml"
            definition_path.write_text(yaml.safe_dump(definition))
            manifest = {role: [samples[key]] for role, key in roles.items()}
            manifest["train"] = [{key: os.path.relpath(value, experiment.HERE.parents[1])
                                  for key, value in manifest["train"][0].items()}]
            manifest_path = root / "v3.json"
            manifest_path.write_text(json.dumps(manifest))
            config = {"split_source": {"expected_counts": {role: 1 for role in roles}}}
            paths = {"definition": definition_path, "manifest": manifest_path,
                     "checkpoint": checkpoint.resolve()}
            splits, domains, loaded = experiment.load_v3_splits(config, paths)
            self.assertEqual(loaded, manifest)
            self.assertEqual(splits["train"][0].image_path,
                             Path(samples[("canal", 2)]["image"]).resolve())
            self.assertEqual(domains[splits["test"][0].image_path], "canal")
            # The reserved unseen mask contains 255 and would fail pixel
            # validation; loading the split must leave it unread.
            self.assertEqual(splits["unseen"][0].mask_path.read_bytes()[:8],
                             b"\x89PNG\r\n\x1a\n")

    def test_locked_source_and_nonempty_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.yaml"
            source.write_text("first")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            experiment.require_locked_file(source, digest, "source")
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                experiment.require_locked_file(source, digest, "source")
            output = root / "output"
            output.mkdir()
            experiment.ensure_empty_output_dir(output)
            (output / "old-result.txt").write_text("existing")
            with self.assertRaises(FileExistsError):
                experiment.ensure_empty_output_dir(output)

    def test_directional_confusion_metrics(self):
        metrics = experiment.add_directional_metrics({
            "confusion_matrix": [[8, 2, 0], [3, 7, 0], [0, 0, 10]],
        })
        self.assertEqual(metrics["water_predicted_as_obstacle_count"], 3)
        self.assertAlmostEqual(metrics["water_predicted_as_obstacle_rate"], .3)
        self.assertEqual(metrics["obstacle_missed_as_water_count"], 2)
        self.assertAlmostEqual(metrics["obstacle_missed_as_water_rate"], .2)


if __name__ == "__main__":
    unittest.main()
