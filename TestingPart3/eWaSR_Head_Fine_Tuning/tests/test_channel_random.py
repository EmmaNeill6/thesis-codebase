"""Ensure the separate channel attempt cannot mix open/dock source data."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_experiment import load_config
from run_tampere_channel_random import DEFAULT_CONFIG, prepare_splits, resolved_config
from train_head import Sample


class ChannelRandomTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG)

    def test_open_images_are_rejected_before_training(self):
        for spec in self.config["data"].values():
            spec["images_dir"] = str(Path(spec["images_dir"]).parent / "open")
        with self.assertRaisesRegex(ValueError, "only Tampere channel"):
            prepare_splits(self.config)

    def test_dock_masks_are_rejected_before_training(self):
        self.config["data"]["test"]["masks_dir"] = "/tmp/Tampere-Dock_5ppt/masks"
        with self.assertRaisesRegex(ValueError, "only Tampere channel"):
            prepare_splits(self.config)

    def test_resolved_yaml_keeps_actual_channel_membership(self):
        before = copy.deepcopy(self.config)
        samples = {"test": [Sample(Path("/tmp/images/channel_image_007.jpg"),
                                   Path("/tmp/masks/channel_image_007.png"))]}
        resolved = resolved_config(self.config, samples)
        self.assertEqual(self.config, before)
        selection = resolved["data"]["test"]["sources"][0]["selection"]
        self.assertEqual(selection["index_pattern"], r"^channel_image_(\d+)$")
        self.assertEqual((selection["start"], selection["end"]), (7, 7))


if __name__ == "__main__":
    unittest.main()
