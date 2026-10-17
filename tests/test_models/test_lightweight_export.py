"""Tiny CPU-only checks for lightweight model export with continuous features."""

import pytest
import torch

from demand_forecast.models.lightweight import LightweightDemandModel


@pytest.fixture(params=[0, 1], ids=["without_cont", "with_cont"])
def export_model(request):
    return LightweightDemandModel(
        sku_vocab_size=2,
        cat_features_dim={"category": 2, "color": 2},
        sku_emb_dim=4,
        cat_emb_dims=4,
        tcn_channels=[4],
        dropout=0,
        n_out=16,
        cont_features_dim=request.param,
    ).eval()


@pytest.mark.parametrize("optimize", [False, True])
def test_to_torchscript_with_cont_features(export_model, optimize):
    traced = export_model.to_torchscript(optimize=optimize)
    # Use a different batch size and continuous values than the tracing inputs.
    inputs = (
        torch.randn(2, 52, 1),
        torch.randn(2, 52, 4),
        torch.randn(2, 16, 4),
        torch.tensor([0, 1]),
    )
    cats = {name: torch.tensor([[0, 1], [1, 0]]) for name in export_model.cat_embeddings}
    cont = torch.tensor([[0.0], [3.0]]) if export_model.cont_proj is not None else None
    static_inputs = (*cats.values(),) if cont is None else (*cats.values(), cont)
    with torch.no_grad():
        expected = export_model(*inputs, cats, cont)
        actual = traced(*inputs, *static_inputs)
        assert actual.shape == (2, 16)
        torch.testing.assert_close(actual, expected)
        if cont is not None:
            changed = traced(*inputs, *cats.values(), cont + 5)
            torch.testing.assert_close(changed, export_model(*inputs, cats, cont + 5))
            assert not torch.allclose(actual, changed)


def test_export_onnx_with_cont_features(export_model, tmp_path):
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "lightweight.onnx"
    export_model.export_onnx(str(path), past_len=4)
    exported = onnx.load(path)
    onnx.checker.check_model(exported)
    inputs = {value.name: value for value in exported.graph.input}
    if export_model.cont_proj is not None:
        dimensions = inputs["cont"].type.tensor_type.shape.dim
        assert dimensions[0].dim_param == "batch_size"
        assert dimensions[1].dim_value == 1
    else:
        assert "cont" not in inputs
