import numpy as np

from minitorch import Linear, Module, Parameter, Tensor
from minitorch.intelligence.gradient_detective import detect_gradients, format_gradient_report


def test_detect_normal_gradients():
    layer = Linear(2, 3)
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    loss = layer(x).sum()
    loss.backward()

    report = layer.gradient_report()

    assert report.parameters_inspected == 2
    assert report.missing_count == 0
    assert report.largest is not None
    assert len(report.largest_gradients) == 2


def test_detect_missing_gradients():
    model = Module()
    model.weight = Parameter([1.0, 2.0])

    report = detect_gradients(model)

    assert report.missing_count == 1
    assert any(f.kind == "missing" for f in report.findings)


def test_detect_near_zero_gradients():
    model = Module()
    model.weight = Parameter([1.0])
    model.weight.grad = np.array([1e-10])

    report = detect_gradients(model, near_zero_threshold=1e-6)

    assert report.near_zero_count >= 1
    assert any(f.kind == "near_zero" for f in report.findings)


def test_detect_large_gradients():
    model = Module()
    model.weight = Parameter([1.0])
    model.weight.grad = np.array([1e4])

    report = detect_gradients(model, large_threshold=10.0)

    assert any(f.kind == "large" for f in report.findings)


def test_topk_largest_uses_heap_not_full_sort():
    model = Module()
    for index in range(5):
        p = Parameter([float(index + 1)])
        setattr(model, f"p{index}", p)

    for name, param in model.named_parameters():
        param.grad = np.array([float(name.split("p")[1]) + 1.0])

    report = detect_gradients(model, top_k=3)
    magnitudes = [item.magnitude for item in report.largest_gradients]

    assert magnitudes == [5.0, 4.0, 3.0]


def test_topk_smallest_nonzero():
    model = Module()
    model.a = Parameter([0.001])
    model.b = Parameter([0.5])
    model.c = Parameter([0.01])

    model.a.grad = np.array([0.001])
    model.b.grad = np.array([0.5])
    model.c.grad = np.array([0.01])

    report = detect_gradients(model, top_k=2)
    smallest = [item.magnitude for item in report.smallest_gradients]

    assert smallest == [0.001, 0.01]


def test_non_finite_gradients_reported():
    model = Module()
    model.weight = Parameter([1.0])
    model.weight.grad = np.array([np.inf])

    report = detect_gradients(model)

    assert report.non_finite_count == 1
    assert any(f.kind == "non_finite" for f in report.findings)


def test_format_gradient_report_text():
    layer = Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    layer(x).sum().backward()

    text = format_gradient_report(layer.gradient_report())
    assert "Gradient Health" in text
    assert "Parameters inspected" in text


def test_topk_k_zero_returns_empty_rankings():
    layer = Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    layer(x).sum().backward()

    report = detect_gradients(layer, top_k=0)
    assert report.largest_gradients == ()
