import numpy as np

from minitorch import Linear, Module, Parameter, Tensor
from minitorch.intelligence.model_xray import build_model_xray, mri_hook_placeholder


class Layer(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter([1.0, 2.0])


class Model(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Layer()
        self.head = Linear(2, 1)

    def forward(self, x):
        hidden = x * self.layer1.weight.reshape(1, -1)
        return self.head(hidden)


def test_xray_discovers_architecture_and_parameters():
    model = Model()
    report = model.xray()

    assert len(report.architecture) >= 2
    assert len(report.parameter_infos) >= 3
    assert report.summary()["parameter_count"] == len(report.parameter_infos)


def test_xray_uses_parameter_registry_names():
    model = Model()
    report = model.xray()
    names = {p.name for p in report.parameter_infos}

    assert "layer1.weight" in names
    assert "head.weight" in names
    assert "head.bias" in names


def test_xray_graph_and_dsa_when_output_provided():
    model = Model()
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    loss = model(x).sum()
    loss.backward()

    report = model.xray(output=loss)

    assert report.graph_summary is not None
    assert report.dsa_profile is not None
    assert report.graph_summary["vertices"] > 0
    assert report.dsa_profile["dfs_visits"] == report.dsa_profile["vertices"]
    assert report.output_tensor_id == loss.id


def test_xray_gradient_section():
    model = Model()
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    model(x).sum().backward()

    report = model.xray()
    grad_report = report.gradients_section()

    assert grad_report is not None
    assert grad_report.parameters_inspected > 0


def test_xray_str_representation():
    model = Model()
    text = str(model.xray())
    assert "MODEL X-RAY" in text
    assert "Architecture" in text


def test_mri_hook_documents_composition():
    text = mri_hook_placeholder().lower()
    assert "ai-mri" in text
    assert "model.mri" in text


def test_xray_parameter_memory_bytes():
    model = Layer()
    info = model.xray().parameter_infos[0]
    assert info.memory_bytes == info.numel * 8
