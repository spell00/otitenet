from pathlib import Path


def test_launcher_validates_cnn_mlp_nested_best_model_artifacts():
    source = Path("launch.sh").read_text(encoding="utf-8")

    assert '"logs/${task}/cnn_mlp_compare"/*/"${uuid}"' in source
    assert 'model_path="${run_dir}/${variant}/best_model.pth"' in source
    assert '${run_dir}/cnn_transfer/best_model.pth' in source
    assert '${run_dir}/cnn_scratch/best_model.pth' in source
    assert '${run_dir}/mlp/best_model.pth' in source
