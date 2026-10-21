"""Ingestão de documentos TXT, MD e JSONL em UTF-8.

Distinção PROVEDOR vs. UNIDADE DE AGRUPAMENTO (Sprint 5): `fonte` é um
rótulo de PROVENIÊNCIA/ATRIBUIÇÃO (ex.: "wikipedia-pt"), usado para
auditoria e distribuição no relatório (`ignis.cli report`) — pode (e deve)
ser o MESMO valor para um provedor inteiro, por maior que seja. Já
`unidade_agrupamento` é a chave usada por `ignis.data.splits.gerar_splits`
(quando `agrupar_por_fonte=True`) para decidir quais documentos NUNCA podem
ser separados entre treino/validação/teste — ela deve identificar um grupo
de documentos efetivamente relacionados (ex.: um mesmo shard/lote de
ingestão, ou revisões do mesmo artigo), nunca o provedor inteiro.

Se `unidade_agrupamento` não for informada explicitamente (nem via `.jsonl`
nem via parâmetro), o padrão é o nome da PASTA IMEDIATA que contém o
arquivo — isto é proposital: ao ingerir um provedor grande (ex.: um dump da
Wikipédia), organizar os arquivos em VÁRIAS subpastas (shards) sob um único
`--fonte wikipedia-pt` explícito garante, por padrão, muitas unidades de
agrupamento independentes em vez de uma só. Colocar todos os arquivos de um
provedor grande em uma única pasta (ou usar `fonte` como se fosse a unidade)
é exatamente o erro que esta distinção existe para evitar — ver
`docs/plano-corpus-piloto.md`, seção sobre granularidade de splits.

CONTRATO de arquivos auxiliares (Sprint 6.1, bug real encontrado com
`scripts/coletar_wikipedia_pt.py`): um diretório de corpus pode conter
arquivos de apoio que NÃO são documentos (ex.: `_metadados.jsonl` com
proveniência, `_estado_coleta.json` para retomada de um coletor). Qualquer
arquivo cujo NOME comece com `_` (sublinhado) é tratado como auxiliar e
NUNCA é lido como conteúdo, mesmo que tenha extensão `.jsonl`/`.txt`/`.md`
reconhecida -- essa é a convenção usada por todo o projeto para distinguir
"documento" de "metadado ao lado do documento". Antes desta correção,
`ingerir_diretorio` tentava ler `_metadados.jsonl` como se cada linha fosse
um documento, falhando com "não possui a chave obrigatória 'texto'" (as
linhas de metadados têm `titulo`/`url`/`revisao`, não `texto`)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

EXTENSOES_SUPORTADAS = {".txt", ".md", ".jsonl"}


@dataclass
class DocumentoBruto:
    documento_id: str
    caminho_origem: str
    fonte: str
    texto: str
    licenca: str | None = None
    permissao_treinamento: bool | None = None
    unidade_agrupamento: str | None = None


def _ler_txt_md(caminho: Path, fonte: str, unidade_agrupamento: str) -> Iterator[DocumentoBruto]:
    texto = caminho.read_text(encoding="utf-8", errors="strict")
    yield DocumentoBruto(
        documento_id=caminho.stem,
        caminho_origem=str(caminho),
        fonte=fonte,
        texto=texto,
        unidade_agrupamento=unidade_agrupamento,
    )


def _ler_jsonl(caminho: Path, fonte: str, unidade_agrupamento: str) -> Iterator[DocumentoBruto]:
    """Cada linha deve ser um objeto JSON com a chave obrigatória `texto`.
    Chaves opcionais reconhecidas: `id`, `licenca`, `permissao_treinamento`,
    `unidade_agrupamento` (quando presente na linha, tem PRIORIDADE sobre o
    padrão derivado da pasta — permite, por exemplo, marcar explicitamente
    que duas linhas de arquivos/pastas diferentes são a mesma unidade, como
    duas revisões do mesmo artigo)."""
    with caminho.open("r", encoding="utf-8", errors="strict") as f:
        for i, linha in enumerate(f):
            linha = linha.strip()
            if not linha:
                continue
            registro = json.loads(linha)
            if "texto" not in registro:
                raise ValueError(f"{caminho}:{i + 1} não possui a chave obrigatória 'texto'")
            yield DocumentoBruto(
                documento_id=registro.get("id", f"{caminho.stem}-{i}"),
                caminho_origem=f"{caminho}:{i}",
                fonte=fonte,
                texto=registro["texto"],
                licenca=registro.get("licenca"),
                permissao_treinamento=registro.get("permissao_treinamento"),
                unidade_agrupamento=registro.get("unidade_agrupamento", unidade_agrupamento),
            )


def ingerir_arquivo(caminho: Path, fonte: str | None = None) -> list[DocumentoBruto]:
    caminho = Path(caminho)
    if caminho.suffix not in EXTENSOES_SUPORTADAS:
        raise ValueError(f"extensão não suportada: {caminho.suffix} (suportadas: {EXTENSOES_SUPORTADAS})")
    fonte_resolvida = fonte or caminho.parent.name or "desconhecida"
    # Padrão da unidade de agrupamento: nome da pasta imediata do arquivo —
    # DIFERENTE de fonte_resolvida quando `fonte` foi passado explicitamente
    # (ver docstring do módulo). Quando `fonte` não foi passado, coincide com
    # fonte_resolvida, preservando o comportamento anterior ao Sprint 5.
    unidade_resolvida = caminho.parent.name or "desconhecida"
    if caminho.suffix in {".txt", ".md"}:
        return list(_ler_txt_md(caminho, fonte_resolvida, unidade_resolvida))
    return list(_ler_jsonl(caminho, fonte_resolvida, unidade_resolvida))


def eh_arquivo_auxiliar(caminho: Path) -> bool:
    """Um arquivo cujo NOME (não o caminho inteiro) começa com `_` é
    auxiliar (metadados, estado de retomada, etc.), nunca conteúdo -- ver
    docstring do módulo. Verificado pelo nome do arquivo em si, não por
    estar dentro de uma pasta cujo nome comece com `_` (uma pasta assim
    ainda teria seus arquivos normais ingeridos normalmente)."""
    return caminho.name.startswith("_")


def ingerir_diretorio(diretorio: Path, fonte: str | None = None) -> list[DocumentoBruto]:
    diretorio = Path(diretorio)
    documentos: list[DocumentoBruto] = []
    for caminho in sorted(diretorio.rglob("*")):
        if not caminho.is_file() or caminho.suffix not in EXTENSOES_SUPORTADAS:
            continue
        if eh_arquivo_auxiliar(caminho):
            continue
        documentos.extend(ingerir_arquivo(caminho, fonte))
    return documentos
