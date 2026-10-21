#!/usr/bin/env python3
"""Instancia o Ignis-30M e compara a contagem real de parâmetros com a
derivação analítica. Uso: python scripts/count_params.py
"""

from __future__ import annotations

from ignis.config import IgnisConfig
from ignis.model import IgnisModel, contagem_parametros_esperada


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> None:
    config = IgnisConfig.ignis_30m().modelo
    esperado = contagem_parametros_esperada(config)

    modelo = IgnisModel(config)
    real = modelo.contar_parametros()

    print(f"Configuração: {config.nome}")
    print(f"Parâmetros esperados (derivação analítica): {_fmt(esperado)}")
    print(f"Parâmetros reais (sum(p.numel()) do modelo): {_fmt(real)}")
    print(f"Coincidem: {esperado == real}")
    if esperado != real:
        print(
            "ATENÇÃO: divergência entre contagem analítica e real — "
            "investigar antes de prosseguir para a Fase 2."
        )


if __name__ == "__main__":
    main()
