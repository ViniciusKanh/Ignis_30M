"""Loop mínimo de treinamento, usado apenas em smoke tests e overfit controlado.

Não é o pipeline de pré-treinamento completo (agendamento de LR, grad
accumulation, checkpoint periódico, retomada) — isso é escopo da Fase 3.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    grad_clip: float | None = 1.0,
    device: str = "cpu",
) -> list[float]:
    """Executa um passo de treino por batch em `batches`; retorna as perdas."""
    model.train()
    perdas: list[float] = []
    for input_ids, target_ids in batches:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        perdas.append(loss.item())
    return perdas
