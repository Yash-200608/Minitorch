from __future__ import annotations

from typing import Any, Iterator

from .registry import ParameterRegistry, _PARAMETER_MARKER
from .tensor import Tensor


class Parameter(Tensor):
    """Trainable tensor wrapper registered by name in a Module registry."""

    def __init__(self, data: Any) -> None:
        super().__init__(data, requires_grad=True)
        self._registry_name: str | None = None
        setattr(self, _PARAMETER_MARKER, True)

    @property
    def name(self) -> str | None:
        return self._registry_name


class Module:
    """
    Base module that owns a HashMap parameter registry and nested submodules.

    Parameters assigned as attributes are registered by dotted path, for example
    `model.layer1.weight` after nesting modules.
    """

    def __init__(self) -> None:
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}
        self._registry = ParameterRegistry()
        self._training = True

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_parameters", "_modules", "_registry", "_training"} or name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if isinstance(value, Parameter):
            object.__setattr__(self, name, value)
            self._parameters[name] = value
            self._registry.register(name, value)
        elif isinstance(value, Module):
            object.__setattr__(self, name, value)
            self._modules[name] = value
            self._register_module_params(name, value)
        else:
            object.__setattr__(self, name, value)

    def _register_module_params(self, prefix: str, module: Module) -> None:
        for local_name, parameter in module._parameters.items():
            self._registry.register(f"{prefix}.{local_name}", parameter)
        for child_name, child in module._modules.items():
            self._register_module_params(f"{prefix}.{child_name}", child)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented.")

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.forward(*inputs, **kwargs)

    def parameter(self, name: str) -> Parameter:
        return self._registry.parameter(name)

    def parameters(self) -> list[Parameter]:
        return [
            self._registry.as_dict()[name]
            for name in sorted(self._registry.names())
        ]

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name in sorted(self._registry.names()):
            parameter = self._registry.get(name)
            if prefix:
                if not name.startswith(prefix):
                    continue
                yield name, parameter
            else:
                yield name, parameter

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self, mode: bool = True) -> Module:
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    @property
    def training(self) -> bool:
        return self._training

    def registry_summary(self) -> dict[str, Any]:
        return self._registry.summary()

    def gradient_report(self, **kwargs: Any):
        """Analyze parameter gradients using Gradient Detective."""
        from .intelligence.gradient_detective import detect_gradients

        return detect_gradients(self, **kwargs)

    def xray(self, output: Tensor | None = None):
        """Return a structured Model X-Ray inspection report."""
        from .intelligence.model_xray import build_model_xray

        return build_model_xray(self, output=output)

    def mri(
        self,
        output: Tensor | None = None,
        *,
        performance: Any | None = None,
        compression: Any | None = None,
        qat_experiment: Any | None = None,
        training_state: dict[str, Any] | None = None,
        experiment: Any | None = None,
    ):
        """Unified AI-MRI diagnostic report over existing MiniTorch evidence."""
        from .intelligence.ai_mri import inspect_model_mri

        return inspect_model_mri(
            self,
            output=output,
            performance=performance,
            compression=compression,
            qat_experiment=qat_experiment,
            training_state=training_state,
            experiment=experiment,
        )

    def save(self, path: str) -> None:
        """Persist structure and parameter values to a MiniTorch artifact."""
        from .persistence import save_model

        save_model(self, path)


class Linear(Module):
    """Fully connected layer: y = x @ weight.T + bias."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        import numpy as np

        self.weight = Parameter(np.zeros((out_features, in_features), dtype=np.float64))
        self.bias = Parameter(np.zeros(out_features, dtype=np.float64))
        self.qat_config: Any | None = None

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        if self._qat_active():
            weight = self.weight.fake_quantize(
                bits=self.qat_config.bits,
                ste_mode=self.qat_config.ste_mode,
            )
        return x @ weight.T + self.bias

    def _qat_active(self) -> bool:
        return (
            self.qat_config is not None
            and self.qat_config.enabled
            and self.training
            and self.qat_config.target == "weights"
        )


class Sequential(Module):
    """Run child modules in registration order."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self._layer_order: list[str] = []
        for index, module in enumerate(modules):
            name = str(index)
            setattr(self, name, module)
            self._layer_order.append(name)

    def forward(self, x: Tensor) -> Tensor:
        for name in self._layer_order:
            x = self._modules[name](x)
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
