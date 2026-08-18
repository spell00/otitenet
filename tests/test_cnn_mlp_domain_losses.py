import pytest
import torch

from otitenet.train.train_cnn_mlp_compare import (
    _build_inverse_triplet_loss,
    _domain_loss_implementation,
    _domain_loss_kind,
)


def test_domain_loss_labels_are_distinct():
    assert _domain_loss_kind("no") == "no"
    assert _domain_loss_kind("DANN") == "dann"
    assert _domain_loss_kind("inverseTriplet") == "inverse_triplet"
    assert _domain_loss_implementation("DANN") == "gradient_reversal_domain_cross_entropy"
    assert _domain_loss_implementation("inverseTriplet") == "reversed_domain_triplet_margin"
    with pytest.raises(ValueError):
        _domain_loss_kind("inverseTriplet_as_DANN")


@pytest.mark.parametrize("distance", ["euclidean", "cosine"])
def test_inverse_triplet_pulls_different_domains_together(distance):
    criterion = _build_inverse_triplet_loss(distance, margin=0.5)
    anchor = torch.tensor([[1.0, 0.0]])
    different_domain = torch.tensor([[1.0, 0.0]])
    same_domain = torch.tensor([[0.0, 1.0]])

    correct_order = criterion(anchor, different_domain, same_domain)
    wrong_order = criterion(anchor, same_domain, different_domain)

    assert correct_order.item() == pytest.approx(0.0)
    assert wrong_order.item() > 0.5
