from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..nn import Parameter


class Optimizer:
    """Base optimizer over a deterministic list of parameters."""

    def __init__(self, parameters: Iterable[Parameter]) -> None:
        self._parameters = list(parameters)

    @property
    def parameters(self) -> list[Parameter]:
        return list(self._parameters)

    def zero_grad(self) -> None:
        for parameter in self._parameters:
            parameter.zero_grad()

    def config_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def state_dict(self, name_by_id: dict[int, str]) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(
        self,
        state: dict[str, Any],
        parameters_by_name: dict[str, Parameter],
        *,
        strict: bool = True,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def validate_config_match(
        saved: dict[str, Any],
        optimizer: Optimizer,
        *,
        strict: bool = True,
    ) -> None:
        current = optimizer.config_dict()
        if saved.get("type") != current.get("type"):
            raise ValueError(
                f"Checkpoint optimizer type {saved.get('type')!r} "
                f"does not match {current.get('type')!r}."
            )
        if strict:
            for key in ("lr", "momentum", "beta1", "beta2", "eps"):
                if key in saved and key in current and saved[key] != current[key]:
                    raise ValueError(
                        f"Checkpoint optimizer {key}={saved[key]} "
                        f"does not match runtime {key}={current[key]}."
                    )


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._velocities: dict[int, np.ndarray] = {
            id(parameter): np.zeros_like(parameter.data, dtype=np.float64)
            for parameter in self._parameters
        }

    def step(self) -> None:
        for parameter in self._parameters:
            if parameter.grad is None:
                continue

            if self.momentum:
                velocity = self._velocities[id(parameter)]
                velocity = self.momentum * velocity + parameter.grad
                self._velocities[id(parameter)] = velocity
                parameter.data = parameter.data - self.lr * velocity
            else:
                parameter.data = parameter.data - self.lr * parameter.grad

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": "SGD",
            "lr": self.lr,
            "momentum": self.momentum,
        }

    def state_dict(self, name_by_id: dict[int, str]) -> dict[str, Any]:
        velocities: dict[str, np.ndarray] = {}
        for parameter in self._parameters:
            name = name_by_id.get(id(parameter))
            if name is None:
                continue
            if self.momentum:
                velocities[name] = self._velocities[id(parameter)].copy()
        return {
            "type": "SGD",
            "lr": self.lr,
            "momentum": self.momentum,
            "velocities": velocities,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        parameters_by_name: dict[str, Parameter],
        *,
        strict: bool = True,
    ) -> None:
        Optimizer.validate_config_match(state, self, strict=strict)
        if state.get("type") != "SGD":
            raise ValueError("Cannot load non-SGD state into SGD optimizer.")
        if not self.momentum:
            return
        saved_velocities = state.get("velocities", {})
        for name, parameter in parameters_by_name.items():
            if name not in saved_velocities:
                if strict:
                    raise ValueError(f"Checkpoint missing SGD velocity for parameter {name!r}.")
                continue
            velocity = np.asarray(saved_velocities[name], dtype=np.float64)
            if velocity.shape != parameter.data.shape:
                raise ValueError(f"SGD velocity shape mismatch for parameter {name!r}.")
            self._velocities[id(parameter)] = velocity.copy()


class Adam(Optimizer):
    """Adam optimizer with per-parameter state isolated from the autograd graph."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._step = 0
        self._moment1: dict[int, np.ndarray] = {
            id(parameter): np.zeros_like(parameter.data, dtype=np.float64)
            for parameter in self._parameters
        }
        self._moment2: dict[int, np.ndarray] = {
            id(parameter): np.zeros_like(parameter.data, dtype=np.float64)
            for parameter in self._parameters
        }

    def step(self) -> None:
        self._step += 1
        bias_correction1 = 1.0 - self.beta1 ** self._step
        bias_correction2 = 1.0 - self.beta2 ** self._step

        for parameter in self._parameters:
            if parameter.grad is None:
                continue

            gradient = parameter.grad
            moment1 = self._moment1[id(parameter)]
            moment2 = self._moment2[id(parameter)]

            moment1 = self.beta1 * moment1 + (1.0 - self.beta1) * gradient
            moment2 = self.beta2 * moment2 + (1.0 - self.beta2) * (gradient * gradient)
            self._moment1[id(parameter)] = moment1
            self._moment2[id(parameter)] = moment2

            corrected_moment1 = moment1 / bias_correction1
            corrected_moment2 = moment2 / bias_correction2
            update = self.lr * corrected_moment1 / (np.sqrt(corrected_moment2) + self.eps)
            parameter.data = parameter.data - update

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": "Adam",
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }

    def state_dict(self, name_by_id: dict[int, str]) -> dict[str, Any]:
        moment1: dict[str, np.ndarray] = {}
        moment2: dict[str, np.ndarray] = {}
        for parameter in self._parameters:
            name = name_by_id.get(id(parameter))
            if name is None:
                continue
            moment1[name] = self._moment1[id(parameter)].copy()
            moment2[name] = self._moment2[id(parameter)].copy()
        return {
            "type": "Adam",
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "step": self._step,
            "moment1": moment1,
            "moment2": moment2,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        parameters_by_name: dict[str, Parameter],
        *,
        strict: bool = True,
    ) -> None:
        Optimizer.validate_config_match(state, self, strict=strict)
        if state.get("type") != "Adam":
            raise ValueError("Cannot load non-Adam state into Adam optimizer.")
        self._step = int(state.get("step", 0))
        saved_m1 = state.get("moment1", {})
        saved_m2 = state.get("moment2", {})
        for name, parameter in parameters_by_name.items():
            if name not in saved_m1 or name not in saved_m2:
                if strict:
                    raise ValueError(f"Checkpoint missing Adam state for parameter {name!r}.")
                continue
            m1 = np.asarray(saved_m1[name], dtype=np.float64)
            m2 = np.asarray(saved_m2[name], dtype=np.float64)
            if m1.shape != parameter.data.shape or m2.shape != parameter.data.shape:
                raise ValueError(f"Adam state shape mismatch for parameter {name!r}.")
            self._moment1[id(parameter)] = m1.copy()
            self._moment2[id(parameter)] = m2.copy()
