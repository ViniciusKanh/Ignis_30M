"""Utilitário de seed determinística (Python, NumPy e PyTorch)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Força implementações determinísticas do PyTorch quando disponíveis.
    #
    # Descoberto na Fase 3, rodando de verdade no Windows nativo (não algo
    # assumido de antemão): o backward de nn.Embedding em CPU pode usar
    # acumulação paralela (index_add_) para somar o gradiente de linhas de
    # embedding repetidas no mesmo lote (tokens repetidos), e essa soma em
    # ponto flutuante não é associativa -- a ordem em que as threads
    # terminam pode variar entre execuções, produzindo pesos finais
    # ligeiramente diferentes mesmo com a MESMA seed e os MESMOS dados
    # (visto em testes de equivalência treino-contínuo-vs-retomado e de
    # determinismo, ambos em CPU, sem nenhuma GPU envolvida). Isso não é
    # uma falha do loop de treinamento em si -- é uma característica do
    # backend CPU multi-thread do PyTorch.
    #
    # `warn_only=True` evita que o treinamento quebre caso algum operador
    # usado no futuro não tenha implementação determinística em CPU; nesse
    # caso o PyTorch apenas emite um aviso e aquele operador específico
    # volta a ser não-determinístico, em vez de levantar uma exceção.
    #
    # Trade-off explícito: isso pode tornar o treinamento mais lento (a
    # soma determinística costuma ser sequencial, não paralela) -- o custo
    # real deve ser medido com scripts/benchmark_training.py, nunca
    # assumido.
    torch.use_deterministic_algorithms(True, warn_only=True)
