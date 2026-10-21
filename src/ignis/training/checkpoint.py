"""Serialização de checkpoints de treinamento (Fase 3).

Um checkpoint completo registra tudo que é necessário para retomar o
treinamento de forma consistente e para auditar depois, sob quais condições
ele foi gerado:

- pesos do modelo (`model_state_dict`; opcionalmente também exportados em
  Safetensors — ver `salvar_pesos_safetensors` — quando o pacote
  `safetensors` está disponível; isso é OPCIONAL, nunca obrigatório, e nunca
  substitui o arquivo `.pt` principal, que também guarda otimizador,
  scheduler e estado de RNG, algo que o formato Safetensors não suporta);
- estado do otimizador e do "scheduler" (aqui o scheduler de LR é uma função
  pura de `step`, então não há estado próprio de scheduler a salvar além do
  `step` em si — ver `ignis.training.trainer.lr_no_passo`);
- `step` e total de tokens processados até aquele ponto;
- a configuração arquitetural do modelo (`IgnisModelConfig`), para detectar
  incompatibilidade ao retomar com uma arquitetura diferente;
- identificação do tokenizador (caminho e hash SHA-256 do arquivo do
  tokenizador) e do dataset (caminho/identificador dos splits usados), para
  detectar quando alguém tenta retomar um treinamento com tokenizador ou
  dados diferentes dos originais;
- sementes e estados de RNG (python/numpy/torch);
- metadados de versão (versão do pacote `ignis`, versão do PyTorch, horário
  de criação, versão do esquema do próprio checkpoint).

Tratamento seguro do arquivo de estado do otimizador: o estado do otimizador
(momentos do AdamW) é salvo DENTRO do mesmo arquivo `.pt` que os pesos do
modelo, nunca em um arquivo separado sem controle de acesso — e a escrita é
atômica (grava em um arquivo temporário no mesmo diretório e troca com
`os.replace`, que é atômico no mesmo sistema de arquivos), para que uma
interrupção a meio da escrita (queda de energia, processo morto) nunca deixe
um checkpoint parcialmente escrito e corrompido sendo lido como válido por
engano — na pior hipótese, o arquivo final simplesmente não é atualizado.

Atenção: `load_checkpoint` usa `weights_only=False` porque o checkpoint
guarda estados de RNG e metadados (objetos Python além de tensores), não
suportados pelo modo `weights_only=True` do PyTorch. Por isso, carregue
apenas checkpoints gerados por este próprio projeto, nunca arquivos de
origem não confiável — `torch.load` com `weights_only=False` pode executar
código arbitrário ao desserializar um arquivo malicioso.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

VERSAO_ESQUEMA_CHECKPOINT = 1


def _versao_pacote_ignis() -> str:
    try:
        import importlib.metadata as m

        return m.version("ignis")
    except Exception:
        return "desconhecida (pacote não instalado via metadata)"


def sha256_arquivo(caminho: str | Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


@dataclass
class CheckpointMetadata:
    """Identificação do contexto em que o checkpoint foi gerado — além do
    estado do modelo/otimizador em si. Tudo opcional porque smoke tests e
    testes unitários podem não ter um tokenizador/dataset real disponível,
    mas DEVE ser preenchido em qualquer execução real de treinamento."""

    config_modelo: dict[str, Any] | None = None
    tokenizer_caminho: str | None = None
    tokenizer_sha256: str | None = None
    dataset_identificador: str | None = None
    dataset_sha256: str | None = None

    @classmethod
    def construir(
        cls,
        config_modelo: dict[str, Any] | None = None,
        tokenizer_caminho: str | Path | None = None,
        dataset_caminho: str | Path | None = None,
        dataset_identificador: str | None = None,
    ) -> "CheckpointMetadata":
        """Calcula os hashes de tokenizador/dataset a partir dos caminhos de
        arquivo, quando fornecidos (evita exigir que quem chama calcule o
        hash manualmente)."""
        tokenizer_sha = sha256_arquivo(tokenizer_caminho) if tokenizer_caminho else None
        dataset_sha = sha256_arquivo(dataset_caminho) if dataset_caminho else None
        return cls(
            config_modelo=config_modelo,
            tokenizer_caminho=str(tokenizer_caminho) if tokenizer_caminho else None,
            tokenizer_sha256=tokenizer_sha,
            dataset_identificador=dataset_identificador or (str(dataset_caminho) if dataset_caminho else None),
            dataset_sha256=dataset_sha,
        )


def _escrever_atomicamente(caminho: Path, escrever_em: "callable[[Path], None]") -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, caminho_tmp_str = tempfile.mkstemp(dir=str(caminho.parent), prefix=f".{caminho.name}.", suffix=".tmp")
    os.close(fd)
    caminho_tmp = Path(caminho_tmp_str)
    try:
        escrever_em(caminho_tmp)
        os.replace(caminho_tmp, caminho)  # atômico no mesmo filesystem
    finally:
        if caminho_tmp.exists():
            caminho_tmp.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
    metadata: CheckpointMetadata | None = None,
    exportar_safetensors: bool = True,
) -> None:
    estado: dict[str, Any] = {
        "versao_esquema_checkpoint": VERSAO_ESQUEMA_CHECKPOINT,
        "criado_em_utc": datetime.now(timezone.utc).isoformat(),
        "versao_ignis": _versao_pacote_ignis(),
        "versao_torch": torch.__version__,
        "step": step,
        "model_state_dict": model.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_initialized():
        estado["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        estado["optimizer_state_dict"] = optimizer.state_dict()
    if metadata is not None:
        estado["metadata"] = asdict(metadata)
    if extra:
        estado["extra"] = extra

    caminho = Path(path)
    _escrever_atomicamente(caminho, lambda tmp: torch.save(estado, tmp))

    if exportar_safetensors:
        try:
            salvar_pesos_safetensors(caminho.with_suffix(".safetensors"), model)
        except ImportError:
            pass  # safetensors é opcional — ver docstring do módulo.


def salvar_pesos_safetensors(path: str | Path, model: torch.nn.Module) -> None:
    """Exporta SOMENTE os pesos do modelo (sem otimizador/RNG/metadados) em
    formato Safetensors, quando o pacote `safetensors` está instalado.
    Levanta `ImportError` se o pacote não estiver disponível — chamadores
    internos (`save_checkpoint`) tratam isso como uma exportação opcional.

    BUG REAL encontrado e corrigido (não apenas hipotético): o Ignis usa
    `tie_word_embeddings=True` por padrão, ou seja, `embed_tokens.weight` e
    `lm_head.weight` são o MESMO tensor na memória (compartilham storage).
    `safetensors.torch.save_file` recusa explicitamente salvar um
    `state_dict` com tensores que compartilham memória (levanta
    `RuntimeError: Some tensors share memory...`), porque duplicá-los no
    arquivo faria o arquivo maior e poderia dessincronizar os dois ao
    recarregar. Isto só foi detectado agora, rodando de verdade no Windows
    com `safetensors` de fato instalado — antes disso, os testes usavam
    `pytest.importorskip("safetensors")` e eram pulados, mascarando o bug.
    Corrigido usando `safetensors.torch.save_model`/`load_model` (em vez de
    `save_file`/`load_file` com um dicionário de tensores montado à mão),
    que tratam tensores compartilhados corretamente: `save_model` detecta e
    remove duplicatas antes de salvar, e `load_model` reconstrói os pesos
    compartilhados ao carregar, sem exigir nenhuma lógica extra aqui."""
    from safetensors.torch import save_model  # ImportError se ausente — proposital.

    caminho = Path(path)
    _escrever_atomicamente(caminho, lambda tmp: save_model(model, str(tmp)))


def carregar_pesos_safetensors(
    path: str | Path, model: torch.nn.Module, map_location: str = "cpu", strict: bool = True
) -> None:
    """Carrega de volta no `model` os pesos exportados por
    `salvar_pesos_safetensors`. Complementa a exportação com a recuperação —
    sem isto, o arquivo `.safetensors` só podia ser escrito, nunca lido de
    volta pelo próprio projeto, o que não é suficiente para considerar a
    funcionalidade validada (ver Fase 3 da skill `ignis-llm-engineer`:
    "validar exportação E recuperação dos pesos").

    Esta função carrega APENAS os pesos — nunca otimizador, RNG, step ou
    metadados, que não existem no formato Safetensors por desenho (ver
    docstring do módulo). Para retomar um treinamento, use `load_checkpoint`
    com o arquivo `.pt` correspondente, não este arquivo `.safetensors`.
    Levanta `ImportError` se o pacote `safetensors` não estiver instalado —
    mesma convenção de `salvar_pesos_safetensors`.

    Usa `safetensors.torch.load_model` (não `load_file` +
    `model.load_state_dict`) justamente para lidar com pesos compartilhados
    (`tie_word_embeddings=True`) salvos por `salvar_pesos_safetensors` sem
    duplicação — `load_model` sabe reconstruir o tensor compartilhado que
    não foi fisicamente duplicado no arquivo, mesmo com `strict=True`."""
    from safetensors.torch import load_model  # ImportError se ausente — proposital.

    load_model(model, str(path), strict=strict, device=map_location)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
    restaurar_rng: bool = True,
    verificar_metadata: CheckpointMetadata | None = None,
) -> dict[str, Any]:
    estado = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(estado["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in estado:
        optimizer.load_state_dict(estado["optimizer_state_dict"])
    if restaurar_rng and "rng_state" in estado:
        rng = estado["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    if verificar_metadata is not None and "metadata" in estado:
        salva = estado["metadata"]
        if (
            verificar_metadata.tokenizer_sha256
            and salva.get("tokenizer_sha256")
            and verificar_metadata.tokenizer_sha256 != salva["tokenizer_sha256"]
        ):
            raise ValueError(
                "retomando com um tokenizador DIFERENTE do usado para gerar este "
                f"checkpoint (hash salvo={salva['tokenizer_sha256'][:12]}..., "
                f"hash atual={verificar_metadata.tokenizer_sha256[:12]}...) — isto "
                "corromperia o treinamento (IDs de token deixam de corresponder ao "
                "vocabulário original). Use o mesmo arquivo de tokenizador."
            )
        if (
            verificar_metadata.dataset_sha256
            and salva.get("dataset_sha256")
            and verificar_metadata.dataset_sha256 != salva["dataset_sha256"]
        ):
            raise ValueError(
                "retomando com um dataset/splits DIFERENTE do usado para gerar este "
                "checkpoint — isto pode causar vazamento entre splits ou repetição "
                "inconsistente de dados. Use o mesmo arquivo de splits."
            )

    return estado
