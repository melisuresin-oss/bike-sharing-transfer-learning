import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gru import GRUForecaster, build_model_from_config


def make_model(n_covariates=10, hidden_size=8, num_layers=1, dropout=0.1, bidirectional=False):
    return GRUForecaster(
        n_covariates=n_covariates,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )


def test_output_shape_matches_batch_size():
    model = make_model()
    batch = 5
    sequence = torch.randn(batch, 24, 1)
    covariates = torch.randn(batch, 10)
    out = model(sequence, covariates)
    assert out.shape == (batch,)


def test_output_has_no_activation_and_can_be_negative():
    # Zero out the output layer's weights and force a negative bias -- if
    # any activation (softplus, ReLU, ...) followed the linear layer, a
    # negative bias could never survive to the output. It must here,
    # regardless of the (irrelevant, since weights are zeroed) input.
    model = make_model(hidden_size=4, n_covariates=3)
    with torch.no_grad():
        model.output_layer.weight.zero_()
        model.output_layer.bias.fill_(-7.5)

    sequence = torch.randn(6, 24, 1)
    covariates = torch.randn(6, 3)
    out = model(sequence, covariates)
    assert torch.allclose(out, torch.full((6,), -7.5))


def test_output_is_not_floored_at_zero_across_random_inputs():
    # A softer, statistical check on top of the deterministic one above:
    # with an untrained (random-weight) model and enough random inputs, at
    # least some raw outputs should be negative. If an activation floored
    # the output at 0, none would be.
    torch.manual_seed(0)
    model = make_model(hidden_size=16, n_covariates=10)
    sequence = torch.randn(200, 24, 1)
    covariates = torch.randn(200, 10)
    out = model(sequence, covariates)
    assert (out < 0).any()


def test_bidirectional_hidden_state_is_concatenated_correctly():
    model = make_model(hidden_size=6, n_covariates=2, bidirectional=True)
    sequence = torch.randn(4, 24, 1)
    covariates = torch.randn(4, 2)
    out = model(sequence, covariates)
    assert out.shape == (4,)
    # output_layer's input width must account for both directions.
    assert model.output_layer.in_features == 6 * 2 + 2


def test_build_model_from_config_reads_model_hyperparameters():
    config = {
        "model": {"hidden_size": 32, "num_layers": 2, "dropout": 0.2, "bidirectional": False},
    }
    model = build_model_from_config(config, n_covariates=10)
    assert model.gru.hidden_size == 32
    assert model.gru.num_layers == 2
    assert model.output_layer.in_features == 32 + 10
