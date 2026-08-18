from pathlib import Path

from otitenet.offline.ui import find_gradcam_image_paths, result_guidance


def test_result_guidance_mentions_next_step():
    guidance = result_guidance("Not Normal", 0.72)
    assert "Next step" in guidance
    assert "doctor" in guidance.lower()


def test_find_gradcam_image_paths_matches_stem(tmp_path: Path):
    gradcam_dir = tmp_path / "gradcam"
    gradcam_dir.mkdir()
    (gradcam_dir / "sample_1.png").write_bytes(b"png")
    (gradcam_dir / "sample_2.png").write_bytes(b"png")
    (gradcam_dir / "other.png").write_bytes(b"png")

    matches = find_gradcam_image_paths(gradcam_dir, "sample.jpg", max_count=4)

    assert len(matches) == 2
    assert all(Path(path).stem.startswith("sample") for path in matches)
