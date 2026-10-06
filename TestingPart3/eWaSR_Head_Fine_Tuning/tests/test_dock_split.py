"""Protect recording boundaries and exhaustive membership for the dock attempt."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_tampere_dock import audit_split_independence, random_image_splits, read_sequence_assignments
from train_head import Sample


class DockSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "recordings.csv"
        self.names = {f"dock_image_{i:03d}.jpg" for i in range(1, 201)}
        self.rows = [
            {"image_name": f"dock_image_{i:03d}.jpg",
             "sequence_id": f"recording_{(i - 1) // 40}",
             "split": "train" if i <= 120 else "validation" if i <= 160 else "test"}
            for i in range(1, 201)
        ]

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        with self.path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=("image_name", "sequence_id", "split"))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_whole_recordings_and_all_200_frames(self):
        self.write()
        roles = read_sequence_assignments(self.path, self.names)
        self.assertEqual(set(roles), self.names)
        self.assertEqual(sum(role == "train" for role in roles.values()), 120)

    def test_one_recording_cannot_cross_splits(self):
        self.rows[120]["sequence_id"] = self.rows[0]["sequence_id"]
        self.write()
        with self.assertRaisesRegex(ValueError, "crosses split boundaries"):
            read_sequence_assignments(self.path, self.names)

    def test_missing_frame_is_rejected(self):
        self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "omits 1"):
            read_sequence_assignments(self.path, self.names)

    def test_duplicate_frame_is_rejected(self):
        self.rows.append(self.rows[0])
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_sequence_assignments(self.path, self.names)

    def test_wrong_proportions_are_rejected_without_cutting_a_recording(self):
        for row in self.rows[-40:]:
            row["split"] = "validation"
        self.write()
        with self.assertRaisesRegex(ValueError, "expected 40"):
            read_sequence_assignments(self.path, self.names)

    def test_absent_recording_metadata_does_not_become_a_random_split(self):
        with self.assertRaisesRegex(FileNotFoundError, "no trip IDs"):
            read_sequence_assignments(self.path, self.names)

    def test_random_membership_is_disjoint_complete_and_reproducible(self):
        samples = [Sample(Path(name), Path(name).with_suffix(".png"))
                   for name in sorted(self.names)]
        first = random_image_splits(samples, 4022)
        second = random_image_splits(list(reversed(samples)), 4022)
        self.assertEqual(first, second)
        self.assertEqual([len(first[role]) for role in ("train", "validation", "test")],
                         [120, 40, 40])
        members = [{sample.image_path for sample in values} for values in first.values()]
        self.assertFalse(members[0] & members[1] or members[0] & members[2] or members[1] & members[2])
        self.assertEqual(set.union(*members), {sample.image_path for sample in samples})
        self.assertNotEqual(first, random_image_splits(samples, 123))

    def test_identical_images_cannot_cross_roles(self):
        root = Path(self.temp.name)
        image_a, image_b = root / "a.jpg", root / "b.jpg"
        image_a.write_bytes(b"same image bytes")
        image_b.write_bytes(image_a.read_bytes())
        splits = {"train": [Sample(image_a, root / "a.png")],
                  "test": [Sample(image_b, root / "b.png")]}
        with self.assertRaisesRegex(ValueError, "Identical image bytes"):
            audit_split_independence(splits)

    def test_shared_mask_cannot_cross_roles(self):
        root = Path(self.temp.name)
        image_a, image_b = root / "a.jpg", root / "b.jpg"
        image_a.write_bytes(b"a")
        image_b.write_bytes(b"b")
        splits = {"train": [Sample(image_a, root / "same.png")],
                  "test": [Sample(image_b, root / "same.png")]}
        with self.assertRaisesRegex(ValueError, "Mask reused"):
            audit_split_independence(splits)


if __name__ == "__main__":
    unittest.main()
