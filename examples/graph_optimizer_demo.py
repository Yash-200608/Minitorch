"""Graph optimizer demonstration."""

from minitorch import Tensor
from minitorch.intelligence.graph_optimizer import optimize


def main():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    a = x * 1
    b = a + 0
    loss = b.sum()

    print("Before optimization:")
    print(loss.explore().describe())
    print()

    loss.backward()
    print(f"Original x.grad: {x.grad}")
    print()

    result = optimize(loss)
    print(result.report())
    print()

    if result.accepted:
        result.optimized_output.backward()
        print(f"Optimized x.grad: {x.grad}")
        print()
        print("After optimization:")
        print(result.optimized_output.explore().describe())


if __name__ == "__main__":
    main()
