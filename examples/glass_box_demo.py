"""Glass Box Intelligence demonstration."""

from minitorch import Linear, Module, Tensor
from minitorch.intelligence.gradient_detective import format_gradient_report


class TinyModel(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(2, 4)
        self.layer2 = Linear(4, 1)

    def forward(self, x):
        return self.layer2(self.layer1(x))


def main():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    model = TinyModel()
    loss = model(x).sum()

    loss.backward()

    print(loss.explain(x))
    print()
    print(loss.explore().describe())
    print()
    print(format_gradient_report(model.gradient_report()))
    print()
    print(model.xray(output=loss))


if __name__ == "__main__":
    main()
