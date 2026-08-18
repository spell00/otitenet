from argparse import Namespace

from otitenet.app.database import ensure_results_log_path_capacity
from otitenet.app.utils import dataset_path_segment, get_model_params_path


class FakeConnection:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeCursor:
    def __init__(self, column_row):
        self.column_row = column_row
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((" ".join(query.split()), params))

    def fetchone(self):
        return self.column_row


def test_absolute_dataset_path_uses_portable_artifact_segment():
    path = "/home/simon/otitenet/data/otite_ds_224/USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v1_train0p5_seed42"
    expected = "otite_ds_224/USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v1_train0p5_seed42"

    assert dataset_path_segment(path) == expected
    assert dataset_path_segment(f"data/{expected}") == expected

    args = Namespace(
        path=path,
        new_size=224,
        fgsm=0,
        n_calibration=0,
        n_positives=1,
        n_negatives=1,
        n_neighbors=1,
        prototypes_to_use="no",
        dist_fct="cosine",
        normalize="yes",
        prototype_strategy="mean",
        prototype_components=1,
        classif_loss="ce",
        dloss="no",
        split_config_in_path=True,
        train_datasets="from_infos_csv",
        valid_dataset="from_infos_csv",
        test_dataset="from_infos_csv",
    )
    query_path = f"logs/best_models/four_classes_220726/resnet18/{get_model_params_path(args)}/queries"

    old_host_specific_path = query_path.replace(expected, f"home/simon/otitenet/data/{expected}")
    assert "/home/simon/otitenet" not in query_path
    assert len(old_host_specific_path) > 255
    assert len(query_path) < len(old_host_specific_path)


def test_results_log_path_migration_widens_varchar_255_to_text():
    conn = FakeConnection()
    cursor = FakeCursor(("varchar", 255))

    ensure_results_log_path_capacity(conn, cursor)

    assert any("ALTER TABLE results MODIFY COLUMN log_path TEXT NULL" in query for query, _ in cursor.executed)
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_results_log_path_migration_is_idempotent_for_text():
    conn = FakeConnection()
    cursor = FakeCursor(("text", 65535))

    ensure_results_log_path_capacity(conn, cursor)

    assert not any(query.startswith("ALTER TABLE") for query, _ in cursor.executed)
    assert conn.commits == 0
    assert conn.rollbacks == 0
