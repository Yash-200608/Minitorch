from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..cache import clear_computation_cache, disable_computation_cache
from ..intelligence.gradient_detective import GradientHealthReport
from ..nn import Module
from ..profiling.profile import PerformanceProfile, ProfilerMode, profile_training_step
from ..qat import QATConfig, get_model_qat_config, prepare_qat
from ..tensor import Tensor
from .optimizers import Adam, Optimizer, SGD


@dataclass
class TrainingHistory:
    """Structured metrics collected during training."""

    losses: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    diagnostics: list[GradientHealthReport] | None = None
    profile: PerformanceProfile | None = None


class Trainer:
    """
    Minimal training loop with an explicit step sequence:

    zero_grad -> forward -> loss -> backward -> metrics -> optimizer.step

    Training position semantics:
    - ``completed_epochs``: number of finished epoch iterations
    - ``completed_steps``: number of finished optimizer updates
  """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        qat_config: QATConfig | None = None,
    ) -> None:
        if qat_config is not None:
            model = prepare_qat(model, qat_config)
            optimizer = _rebuild_optimizer(optimizer, model)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.completed_epochs = 0
        self.completed_steps = 0
        self._loss_history: list[float] = []
        self._epoch_history: list[int] = []

    def train_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        self.optimizer.zero_grad()
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        self.optimizer.step()
        self.completed_steps += 1
        return loss

    def fit(
        self,
        inputs: Tensor,
        targets: Tensor,
        epochs: int = 100,
        diagnostics: bool = False,
        profile: bool = False,
        resume: bool = False,
    ) -> TrainingHistory:
        """
        Train for ``epochs`` iterations.

        When ``resume=True``, epoch indices continue from ``completed_epochs``.
        """
        disable_computation_cache()
        clear_computation_cache()

        history = TrainingHistory(
            losses=list(self._loss_history),
            epochs=list(self._epoch_history),
            diagnostics=[] if diagnostics else None,
        )

        start_epoch = self.completed_epochs if resume else 0
        if not resume:
            self.completed_epochs = 0
            self.completed_steps = 0
            self._loss_history = []
            self._epoch_history = []
            history.losses = []
            history.epochs = []

        for offset in range(epochs):
            epoch_index = start_epoch + offset if resume else offset
            loss = self.train_step(inputs, targets)
            loss_value = float(loss.data)
            history.losses.append(loss_value)
            history.epochs.append(epoch_index)
            self._loss_history.append(loss_value)
            self._epoch_history.append(epoch_index)
            self.completed_epochs = epoch_index + 1
            if diagnostics:
                if history.diagnostics is None:
                    history.diagnostics = []
                history.diagnostics.append(self.model.gradient_report())

        if profile:
            history.profile = self.profile_step(inputs, targets)

        return history

    def resume_state(self) -> Any:
        from ..persistence.checkpoint import TrainingResumeState

        return TrainingResumeState(
            completed_epochs=self.completed_epochs,
            completed_steps=self.completed_steps,
            losses=list(self._loss_history),
            history_epochs=list(self._epoch_history),
            qat_config=(
                get_model_qat_config(self.model).to_dict()
                if get_model_qat_config(self.model) is not None
                else None
            ),
        )

    def apply_resume_state(self, resume_state: Any) -> None:
        self.completed_epochs = int(resume_state.completed_epochs)
        self.completed_steps = int(resume_state.completed_steps)
        self._loss_history = list(resume_state.losses)
        self._epoch_history = list(resume_state.history_epochs)

    def save_checkpoint(self, path: str, *, include_rng: bool = True) -> None:
        from ..persistence.checkpoint import save_training_checkpoint

        save_training_checkpoint(
            self.model,
            self.optimizer,
            self.resume_state(),
            path,
            include_rng=include_rng,
        )

    @staticmethod
    def from_checkpoint(
        path: str,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer | None = None,
        *,
        strict: bool = True,
    ) -> Trainer:
        from ..persistence.checkpoint import load_training_checkpoint

        checkpoint = load_training_checkpoint(path)
        model, restored_optimizer, resume_state = checkpoint.restore(
            optimizer=optimizer,
            strict=strict,
        )
        trainer = Trainer(model, restored_optimizer, loss_fn)
        trainer.apply_resume_state(resume_state)
        return trainer

    def profile_step(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        mode: ProfilerMode = ProfilerMode.LIGHTWEIGHT,
    ) -> PerformanceProfile:
        """Run one training step and return measured timings plus graph/DSA counters."""
        return profile_training_step(
            self.model,
            self.optimizer,
            self.loss_fn,
            inputs,
            targets,
            mode=mode,
        )


def _rebuild_optimizer(optimizer: Optimizer, model: Module) -> Optimizer:
    parameters = model.parameters()
    if isinstance(optimizer, SGD):
        return SGD(parameters, lr=optimizer.lr, momentum=optimizer.momentum)
    if isinstance(optimizer, Adam):
        return Adam(
            parameters,
            lr=optimizer.lr,
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            eps=optimizer.eps,
        )
    raise TypeError(f"Cannot rebuild optimizer type {type(optimizer).__name__}.")
