import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import train_lars_head_sim4 as experiment
from train_head import Sample


class LaSRHeadSim4Tests(unittest.TestCase):
    def make_splits(self, root):
        return {role: [Sample(root / f"{role}.jpg", root / f"{role}.png")]
                for role in experiment.ROLES}

    def make_manifest(self, splits):
        return {role: [{"image": str(samples[0].image_path),
                        "mask": str(samples[0].mask_path)}]
                for role, samples in splits.items()}

    def test_v2_match_requires_same_pair_in_each_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            splits = self.make_splits(Path(temporary))
            manifest = self.make_manifest(splits)
            counts = {role: 1 for role in experiment.ROLES}
            experiment.assert_matching_v2_manifest(splits, manifest, counts)
            manifest["test"], manifest["validation"] = manifest["validation"], manifest["test"]
            with self.assertRaisesRegex(ValueError, "membership does not match"):
                experiment.assert_matching_v2_manifest(splits, manifest, counts)

    def test_v2_match_rejects_duplicate_or_missing_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            splits = self.make_splits(Path(temporary))
            manifest = self.make_manifest(splits)
            manifest["train"].append(dict(manifest["train"][0]))
            counts = {role: 1 for role in experiment.ROLES}
            with self.assertRaisesRegex(ValueError, "train count"):
                experiment.assert_matching_v2_manifest(splits, manifest, counts)
            del manifest["unseen"]
            with self.assertRaisesRegex(ValueError, "must both contain"):
                experiment.assert_matching_v2_manifest(splits, manifest, counts)

    def test_config_is_locked_to_original_checkpoint_and_new_output(self):
        config, paths = experiment.load_experiment_config(experiment.DEFAULT_CONFIG)
        self.assertEqual(paths["definition"], experiment.SOURCE_YAML)
        self.assertEqual(paths["manifest"], experiment.V2_MANIFEST)
        self.assertEqual(paths["output"], experiment.OUTPUT_DIR)
        self.assertEqual(config["training"]["epochs"], 20)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            experiment.ensure_empty_output_dir(output)
            (output / "prior_result.json").write_text(json.dumps({"done": True}))
            with self.assertRaises(FileExistsError):
                experiment.ensure_empty_output_dir(output)


if __name__ == "__main__":
    unittest.main()
