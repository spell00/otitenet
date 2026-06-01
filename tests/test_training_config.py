import argparse
import unittest

from otitenet.utils.training_config import (
    disable_stn_when_unsupported,
    stn_supported_for_image_size,
    validate_n_calibration,
)


class TrainingConfigTests(unittest.TestCase):
    def test_stn_requires_224_geometry(self):
        self.assertFalse(stn_supported_for_image_size(64))
        self.assertFalse(stn_supported_for_image_size(128))
        self.assertTrue(stn_supported_for_image_size(224))

    def test_disable_stn_for_64px_training(self):
        args = argparse.Namespace(is_stn=1, new_size=64)
        out = disable_stn_when_unsupported(args)
        self.assertEqual(out.is_stn, 0)

    def test_keep_stn_for_224px_training(self):
        args = argparse.Namespace(is_stn=1, new_size=224)
        out = disable_stn_when_unsupported(args)
        self.assertEqual(out.is_stn, 1)

    def test_validate_n_calibration_preserves_positive_value(self):
        args = argparse.Namespace(n_calibration=4)
        out = validate_n_calibration(args)
        self.assertEqual(out.n_calibration, 4)

    def test_validate_n_calibration_rejects_negative_value(self):
        args = argparse.Namespace(n_calibration=-1)
        with self.assertRaises(ValueError):
            validate_n_calibration(args)


if __name__ == "__main__":
    unittest.main()
