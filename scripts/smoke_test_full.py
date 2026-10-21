#!/usr/bin/env python3
"""Smoke test do Ignis-30M completo (configuração real, 30.416.384 parâmetros).

Executa apenas:
  - instanciação do modelo completo;
  - algumas repetições de forward pass (batch e sequência pequenos, controláveis
    por argumento) + cálculo de loss de cross-entropy sobre alvos sintéticos;
  - backward pass, verificando que a loss e os gradientes são finitos.

NÃO realiza nenhum passo de otimização (nenhum optimizer.step()) nem
pré-treinamento — é apenas uma verificação de sanidade de que o modelo
completo roda sem erros e sem estourar a memória disponível, com números
efetivamente medidos (tempo e, se psutil estiver instalado, RAM do processo).

Uso:
    python scripts/smoke_test_full.py --batch-size 1 --seq-len 32 --repeticoes 3
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from ignis.config import IgnisConfig
from ignis.model import IgnisModel
from ignis.seeding import set_seed

try:
    import psutil

    _TEM_PSUTIL = True
except ImportError:
    _TEM_PSUTIL = False


def _ram_atual_mb() -> float | None:
    if not _TEM_PSUTIL:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--repeticoes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    config = IgnisConfig.ignis_30m().modelo
    if args.seq_len > config.max_seq_len:
        raise SystemExit(
            f"--seq-len ({args.seq_len}) não pode exceder max_seq_len da "
            f"config ({config.max_seq_len})"
        )

    print(f"Instanciando Ignis-30M (config={config.nome})...")
    t0 = time.perf_counter()
    modelo = IgnisModel(config)
    t_instanciacao = time.perf_counter() - t0
    n_params = modelo.contar_parametros()
    print(f"  parâmetros treináveis: {n_params:,}".replace(",", "."))
    print(f"  tempo de instanciação: {t_instanciacao:.2f}s")

    ram_antes = _ram_atual_mb()

    tempos_forward: list[float] = []
    tempos_backward: list[float] = []
    perdas: list[float] = []

    for i in range(args.repeticoes):
        entrada = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len))
        alvo = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len))

        modelo.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        logits = modelo(entrada)
        t_forward = time.perf_counter() - t0

        assert logits.shape == (args.batch_size, args.seq_len, config.vocab_size)
        assert torch.isfinite(logits).all(), "logits não finitos no forward"

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), alvo.reshape(-1))
        assert torch.isfinite(loss).item(), "loss não finita"

        t0 = time.perf_counter()
        loss.backward()
        t_backward = time.perf_counter() - t0

        grads_finitos = all(
            torch.isfinite(p.grad).all()
            for p in modelo.parameters()
            if p.grad is not None
        )
        assert grads_finitos, "gradientes não finitos detectados"

        tempos_forward.append(t_forward)
        tempos_backward.append(t_backward)
        perdas.append(loss.item())

        print(
            f"  repetição {i + 1}/{args.repeticoes}: "
            f"loss={loss.item():.4f}  forward={t_forward:.3f}s  backward={t_backward:.3f}s"
        )

    ram_depois = _ram_atual_mb()

    print()
    print("=== Resumo do smoke test (Ignis-30M completo) ===")
    print(f"batch_size={args.batch_size}  seq_len={args.seq_len}  repeticoes={args.repeticoes}")
    print("loss finita em todas as repetições: True")
    print("gradientes finitos em todas as repetições: True")
    print(f"tempo médio forward:  {sum(tempos_forward) / len(tempos_forward):.3f}s")
    print(f"tempo médio backward: {sum(tempos_backward) / len(tempos_backward):.3f}s")
    if ram_antes is not None and ram_depois is not None:
        print(f"RAM do processo antes:  {ram_antes:.1f} MB")
        print(f"RAM do processo depois: {ram_depois:.1f} MB")
        print(f"delta RAM:              {ram_depois - ram_antes:.1f} MB")
    else:
        print(
            "psutil não disponível — medição de RAM do processo não realizada "
            "(instale com 'pip install psutil' para obter esse dado)."
        )
    print()
    print(
        "IMPORTANTE: isto NÃO é treinamento. Nenhum passo de otimização foi "
        "executado; o modelo permanece com pesos aleatórios."
    )


if __name__ == "__main__":
    main()
