"""Manifesto de proveniência de documentos do corpus.

Cada documento ingerido recebe uma entrada de manifesto com proveniência,
licença, hash e status de triagem. Nenhum documento é aceito automaticamente
sem licença e permissão de treinamento declaradas — ver `StatusDocumento`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class StatusDocumento(str, Enum):
    ACEITO = "aceito"
    REJEITADO = "rejeitado"
    QUARENTENA = "quarentena"  # possível PII ou outro risco — revisar manualmente
    LICENCA_DESCONHECIDA = "licenca_desconhecida"


class EntradaManifesto(BaseModel):
    documento_id: str
    caminho_origem: str
    fonte: str
    licenca: str | None = None
    permissao_treinamento: bool | None = None
    sha256: str
    tamanho_caracteres: int
    idioma_assumido: str = "pt-BR"
    status: StatusDocumento = StatusDocumento.LICENCA_DESCONHECIDA
    motivo: str = ""
    grupo_quase_duplicata: str | None = None
    unidade_agrupamento: str | None = None
    """Unidade de agrupamento para fins de split (Sprint 5) — ver docstring
    de `ignis.data.ingest` e `ignis.data.splits`. DIFERENTE de `fonte`:
    `fonte` é proveniência/atribuição (pode ser constante para um provedor
    inteiro); `unidade_agrupamento` é a chave que impede documentos
    relacionados de caírem em splits diferentes. Quando None, o consumidor
    (`ignis.data.splits._componentes_conexas`) usa `fonte` como fallback,
    preservando o comportamento anterior ao Sprint 5."""
