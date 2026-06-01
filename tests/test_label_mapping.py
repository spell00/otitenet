import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from otitenet.data.labels import CANONICAL_LABELS, labels_for_task, normalize_label
from otitenet.data.dataset_paths import infer_output_subdir_from_split_datasets, resolve_processed_dataset_path
from otitenet.data.make_dataset2 import dataset_output_subdir, make_dataset_folder


class LabelMappingTests(unittest.TestCase):
    def test_requested_raw_labels_map_to_canonical_training_labels(self):
        expected = {
            "Normal": "Normal",
            "normal": "Normal",
            "Effusion": "NotNormal",
            "Aom": "NotNormal",
            "Chronic": "NotNormal",
            "Chronic otitis media": "NotNormal",
            "OtitExterna": "NotNormal",
            "Pseudomebrane": "NotNormal",
            "tympanoskleros": "NotNormal",
            "myringosclerosis": "NotNormal",
            "foreign": "NotNormal",
            "earwax": "Wax",
            "earwax plug": "Wax",
            "Tube": "Tube",
            "Earventulation": "Tube",
        }

        for raw, canonical in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_label(raw, scheme="four_class"), canonical)

    def test_binary_scheme_keeps_old_normal_notnormal_labels(self):
        for raw in ["Effusion", "earwax", "earwax plug", "Tube", "Earventulation"]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_label(raw, scheme="binary"), "NotNormal")

        self.assertEqual(labels_for_task("notNormal"), ("Normal", "NotNormal"))
        self.assertEqual(labels_for_task("otite_four_class"), ("Normal", "NotNormal", "Wax", "Tube"))

    def test_known_dataset_spellings_are_supported(self):
        aliases = {
            "Chornic": "NotNormal",
            "PseduoMembran": "NotNormal",
            "Myringosclerosis": "NotNormal",
            "Foreign": "NotNormal",
            "Earwax": "Wax",
            "Earwax plug": "Wax",
            "Anormal": "NotNormal",
            "anormal": "NotNormal",
        }

        for raw, canonical in aliases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_label(raw, scheme="four_class"), canonical)

    def test_unknown_label_fails_in_strict_mode(self):
        with self.assertRaises(ValueError):
            normalize_label("DefinitelyUnknown")

    def test_preprocessing_output_subdir_is_derived_from_included_datasets(self):
        include = [
            "Banque_Calaman_USA_2020_trie_CM",
            "Banque_Viscaino_Chili_2020",
            "Banque_Comert_Turquie_2020_jpg",
            "GMFUNL_jan2023",
        ]

        self.assertEqual(
            dataset_output_subdir(include),
            "USA_Turquie_Chili_GMFUNL",
        )

    def test_output_subdir_can_be_inferred_from_split_datasets(self):
        self.assertEqual(
            infer_output_subdir_from_split_datasets(
                train_datasets="Banque_Comert_Turquie_2020_jpg,Banque_Calaman_USA_2020_trie_CM,GMFUNL_jan2023",
                valid_dataset="Banque_Viscaino_Chili_2020",
                test_dataset="inference",
            ),
            "USA_Turquie_Chili_GMFUNL_inference",
        )

    def test_processed_dataset_path_resolves_order_insensitive_subdir_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "otite_ds_64" / "USA_Turquie_Chili_GMFUNL_inference"
            expected.mkdir(parents=True)
            (expected / "infos.csv").write_text("dataset,name,label,group\n", encoding="utf-8")

            requested = root / "otite_ds_64" / "Turquie_GMFUNL_inference_Chili_USA"

            self.assertEqual(resolve_processed_dataset_path(requested), str(expected))

    def test_processed_dataset_path_prefers_smallest_superset_for_missing_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "otite_ds_64"
            exact_superset = base / "USA_Turquie_Chili_GMFUNL"
            larger_superset = base / "USA_Turquie_Chili_GMFUNL_inference"
            exact_superset.mkdir(parents=True)
            larger_superset.mkdir(parents=True)
            (exact_superset / "infos.csv").write_text("dataset,name,label,group\n", encoding="utf-8")
            (larger_superset / "infos.csv").write_text("dataset,name,label,group\n", encoding="utf-8")

            requested = base / "USA_Chili_GMFUNL"

            self.assertEqual(resolve_processed_dataset_path(requested), str(exact_superset))

    def test_preprocessing_writes_canonical_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "datasets" / "Banque_Calaman_USA_2020_trie_CM"
            for label in ["Normal", "Effusion", "Tube"]:
                label_dir = source / label
                label_dir.mkdir(parents=True)
                Image.new("RGB", (8, 8), color=(120, 80, 40)).save(label_dir / f"{label}.jpg")

            output_base = root / "otite_ds"
            make_dataset_folder(
                path=str(root / "datasets"),
                size=16,
                new_path=str(output_base),
                include_datasets=["Banque_Calaman_USA_2020_trie_CM"],
                split_mode="by_dataset",
                label_scheme="four_class",
            )

            infos = pd.read_csv(root / "otite_ds_16" / "USA" / "infos.csv")
            self.assertEqual(set(infos["label"]), {"Normal", "NotNormal", "Tube"})
            self.assertEqual(set(infos["raw_label"]), {"Normal", "Effusion", "Tube"})
            self.assertTrue(set(infos["label"]).issubset(set(CANONICAL_LABELS)))

    def test_preprocessing_can_write_old_binary_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "datasets" / "Banque_Calaman_USA_2020_trie_CM"
            for label in ["Normal", "Effusion", "Tube"]:
                label_dir = source / label
                label_dir.mkdir(parents=True)
                Image.new("RGB", (8, 8), color=(120, 80, 40)).save(label_dir / f"{label}.jpg")

            output_base = root / "otite_ds"
            make_dataset_folder(
                path=str(root / "datasets"),
                size=16,
                new_path=str(output_base),
                include_datasets=["Banque_Calaman_USA_2020_trie_CM"],
                split_mode="by_dataset",
                label_scheme="binary",
            )

            infos = pd.read_csv(root / "otite_ds_16" / "USA" / "infos.csv")
            self.assertEqual(set(infos["label"]), {"Normal", "NotNormal"})
            self.assertEqual(set(infos["raw_label"]), {"Normal", "Effusion", "Tube"})


if __name__ == "__main__":
    unittest.main()
