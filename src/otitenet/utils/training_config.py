"""Small training configuration helpers without heavy ML imports."""

from __future__ import annotations


def stn_supported_for_image_size(image_size) -> bool:
    """The current CNN STN localization head expects the 224px geometry."""
    try:
        return int(image_size) >= 224
    except Exception:
        return False


def disable_stn_when_unsupported(args):
    if int(getattr(args, "is_stn", 0) or 0) and not stn_supported_for_image_size(getattr(args, "new_size", None)):
        print(
            "[Config] Disabling STN because the current localization head requires "
            f"new_size>=224; received new_size={getattr(args, 'new_size', None)}."
        )
        args.is_stn = 0
    return args


def validate_n_calibration(args):
    try:
        n_calibration = int(getattr(args, "n_calibration", 0) or 0)
    except Exception as exc:
        raise ValueError(f"n_calibration must be an integer; received {getattr(args, 'n_calibration', None)!r}.") from exc

    if n_calibration < 0:
        raise ValueError(f"n_calibration must be >= 0; received {n_calibration}.")

    args.n_calibration = n_calibration
    return args
