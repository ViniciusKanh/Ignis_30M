"""Registro de métricas de treinamento em JSONL.

Cada linha é um objeto JSON com pelo menos `step`; os demais campos variam
conforme o evento (passo de treino, avaliação de validação, checkpoint
gerado, falha numérica) — por isso JSONL (um objeto heterogêneo por linha) é
mais simples do que CSV aqui, que exigiria um schema de colunas fixo.

A integração com MLflow é OPCIONAL e nunca obrigatória: só é usada quando
`habilitar_mlflow()` é chamado explicitamente, e uma falha ao importar
`mlflow` não quebra o treinamento — apenas desativa esse espelhamento e
registra um aviso.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    import psutil

    _PSUTIL_DISPONIVEL = True
except ImportError:
    _PSUTIL_DISPONIVEL = False


def uso_memoria_mb() -> float | None:
    """RAM residente (RSS) do processo atual, em MB, ou None se `psutil` não
    estiver instalado (dependência opcional de desenvolvimento)."""
    if not _PSUTIL_DISPONIVEL:
        return None
    processo = psutil.Process()
    return processo.memory_info().rss / (1024 * 1024)


class RegistradorMetricas:
    def __init__(self, caminho: str | Path, usar_mlflow: bool = False):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._mlflow = None
        self._pico_ram_mb = 0.0
        if usar_mlflow:
            self.habilitar_mlflow()

    def habilitar_mlflow(self, experiment_name: str | None = None) -> bool:
        """Tenta ativar o espelhamento das métricas para o MLflow. Retorna
        True se ativado com sucesso, False se o pacote `mlflow` não está
        instalado (o registro em JSONL continua funcionando normalmente)."""
        try:
            import mlflow

            if experiment_name:
                mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            return True
        except ImportError:
            self._mlflow = None
            return False

    def registrar(self, campos: dict[str, Any]) -> None:
        linha = dict(campos)
        linha.setdefault("timestamp_unix", time.time())

        ram_atual = uso_memoria_mb()
        if ram_atual is not None:
            linha.setdefault("ram_mb", ram_atual)
            self._pico_ram_mb = max(self._pico_ram_mb, ram_atual)
            linha.setdefault("pico_ram_mb", self._pico_ram_mb)

        with self.caminho.open("a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False, default=str))
            f.write("\n")

        if self._mlflow is not None:
            passo = campos.get("step")
            for chave, valor in campos.items():
                if chave == "step" or not isinstance(valor, (int, float)):
                    continue
                try:
                    self._mlflow.log_metric(chave, float(valor), step=passo)
                except Exception:
                    pass  # espelhamento no MLflow nunca deve derrubar o treinamento.


def ler_metricas(caminho: str | Path) -> list[dict[str, Any]]:
    linhas = []
    with Path(caminho).open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                linhas.append(json.loads(linha))
    return linhas
