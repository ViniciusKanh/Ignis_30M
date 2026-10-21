"""Pipeline de ingestão, limpeza, triagem e geração de manifesto do corpus.

Ordem de triagem por documento: duplicata exata > qualidade/tamanho > PII
(quarentena) > licença/permissão de treinamento. Nenhum documento é aceito
sem licença e permissão de treinamento declaradas explicitamente.

INGESTÃO INCREMENTAL E RETOMÁVEL (Sprint 5): `processar_diretorio` aceita
`manifesto_existente` (o manifesto de uma execução anterior) e
`limite_novos_documentos` (um teto de quantos documentos NOVOS ou
ATUALIZADOS processar nesta chamada). Isso permite ingerir um diretório que
cresce aos poucos (ex.: um corpus sendo baixado em lotes) chamando o mesmo
comando repetidamente, sem reprocessar do zero nem baixar/processar tudo de
uma vez — ver docstring de `processar_diretorio` para o mecanismo exato
(comparação de hash SHA-256 por `caminho_origem`), com comparação de
quase-duplicatas entre lotes do diretório local.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dedup import agrupar_quase_duplicatas, sha256_texto
from .filtros import FiltrosQualidade, avaliar_qualidade
from .ingest import DocumentoBruto, ingerir_diretorio
from .manifest import EntradaManifesto, StatusDocumento
from .normalize import normalizar_texto
from .pii import detectar_pii


def _classificar_documento(
    doc: DocumentoBruto,
    texto_norm: str,
    sha: str,
    hashes_vistos: dict[str, str],
    filtros: FiltrosQualidade,
) -> tuple[StatusDocumento, str]:
    if sha in hashes_vistos:
        return StatusDocumento.REJEITADO, f"duplicata exata de {hashes_vistos[sha]}"

    ok_qualidade, motivo_qualidade = avaliar_qualidade(texto_norm, filtros)
    if not ok_qualidade:
        return StatusDocumento.REJEITADO, motivo_qualidade

    achados_pii = detectar_pii(texto_norm)
    if achados_pii:
        return StatusDocumento.QUARENTENA, f"possível PII detectada: {achados_pii}"

    if doc.licenca and doc.permissao_treinamento:
        return StatusDocumento.ACEITO, "licença e permissão de treinamento declaradas"

    motivo = "licença não declarada" if not doc.licenca else "permissão de treinamento não declarada"
    return StatusDocumento.LICENCA_DESCONHECIDA, motivo


@dataclass
class ResultadoPipeline:
    manifesto: list[EntradaManifesto]
    textos_aceitos: dict[str, str]
    """Textos normalizados dos documentos ACEITOS processados NESTA chamada.
    Em execução incremental (`manifesto_existente` informado), NÃO inclui o
    texto de documentos aceitos em chamadas anteriores e pulados por não
    terem mudado — apenas `manifesto` cobre o histórico completo; o texto
    pleno de um documento antigo só é recuperável relendo o arquivo original
    em disco (ver `_carregar_textos_do_split` em `ignis.cli`)."""
    n_novos_processados: int = 0
    """Quantos documentos foram efetivamente processados NESTA chamada (novos
    ou atualizados) -- distinto de `len(manifesto)`, que inclui também
    documentos carregados de `manifesto_existente` e pulados por não terem
    mudado (ver `processar_diretorio`). Em uma execução não-incremental
    (sem `manifesto_existente`), é igual ao total de documentos brutos
    ingeridos."""

    def relatorio(self) -> dict:
        contagens = {status.value: 0 for status in StatusDocumento}
        for entrada in self.manifesto:
            contagens[entrada.status.value] += 1
        return {
            "total_documentos": len(self.manifesto),
            "por_status": contagens,
            "novos_processados_nesta_execucao": self.n_novos_processados,
        }


def processar_diretorio(
    diretorio: Path,
    fonte: str | None = None,
    filtros: FiltrosQualidade | None = None,
    limiar_quase_duplicata: float = 0.8,
    manifesto_existente: list[EntradaManifesto] | None = None,
    limite_novos_documentos: int | None = None,
) -> ResultadoPipeline:
    """Ingere, limpa, triagem e gera o manifesto de um diretório de corpus.

    Com `manifesto_existente` (tipicamente carregado via `carregar_manifesto`
    de uma execução anterior deste mesmo diretório), a execução é
    INCREMENTAL: um documento cujo `caminho_origem` já está no manifesto
    existente E cujo hash SHA-256 (do texto normalizado) não mudou é pulado
    — não é relido nem reclassificado. Um documento cujo `caminho_origem` já
    existia mas cujo hash mudou é tratado como atualização (reprocessado,
    substituindo a entrada antiga). Documentos cujo `caminho_origem` nunca
    apareceu antes são tratados como novos.

    `limite_novos_documentos`, quando informado, processa no máximo esse
    número de documentos novos/atualizados NESTA chamada — os demais
    simplesmente não são incluídos no manifesto retornado e serão
    considerados "novos" novamente em uma chamada futura (nada é marcado
    como processado até realmente ser processado). Isso permite ingerir um
    diretório muito grande em lotes controlados, chamando este pipeline
    repetidamente.

    Quase-duplicatas são recalculadas sobre todos os aceitos disponíveis no
    diretório, incluindo os inalterados, para evitar vazamento entre lotes.
    """
    filtros = filtros or FiltrosQualidade()
    brutos = ingerir_diretorio(Path(diretorio), fonte)

    manifesto_existente = manifesto_existente or []
    manifesto: dict[str, EntradaManifesto] = {e.documento_id: e for e in manifesto_existente}
    sha_por_caminho: dict[str, str] = {e.caminho_origem: e.sha256 for e in manifesto_existente}
    hashes_vistos: dict[str, str] = {
        e.sha256: e.documento_id for e in manifesto_existente if e.status != StatusDocumento.REJEITADO
    }

    textos_para_dedup: dict[str, str] = {}
    normalizados: dict[str, str] = {}
    n_novos_processados = 0

    for doc in brutos:
        texto_norm = normalizar_texto(doc.texto)
        sha = sha256_texto(texto_norm)

        sha_anterior = sha_por_caminho.get(doc.caminho_origem)
        inalterado = (
            sha_anterior is not None and sha_anterior == sha and doc.documento_id in manifesto
        )
        if inalterado:
            if manifesto[doc.documento_id].status == StatusDocumento.ACEITO:
                textos_para_dedup[doc.documento_id] = texto_norm
            continue  # execução incremental: já processado e sem mudanças

        if limite_novos_documentos is not None and n_novos_processados >= limite_novos_documentos:
            continue  # respeita o limite desta chamada; será pego em execução futura

        status, motivo = _classificar_documento(doc, texto_norm, sha, hashes_vistos, filtros)
        if status != StatusDocumento.REJEITADO:
            hashes_vistos.setdefault(sha, doc.documento_id)

        manifesto[doc.documento_id] = EntradaManifesto(
            documento_id=doc.documento_id,
            caminho_origem=doc.caminho_origem,
            fonte=doc.fonte,
            licenca=doc.licenca,
            permissao_treinamento=doc.permissao_treinamento,
            sha256=sha,
            tamanho_caracteres=len(texto_norm),
            status=status,
            motivo=motivo,
            unidade_agrupamento=doc.unidade_agrupamento,
        )
        n_novos_processados += 1
        if status == StatusDocumento.ACEITO:
            normalizados[doc.documento_id] = texto_norm
            textos_para_dedup[doc.documento_id] = texto_norm

    if textos_para_dedup:
        grupos = agrupar_quase_duplicatas(textos_para_dedup, limiar=limiar_quase_duplicata)
        for doc_id, grupo_id in grupos.items():
            manifesto[doc_id].grupo_quase_duplicata = grupo_id

    return ResultadoPipeline(
        manifesto=list(manifesto.values()),
        textos_aceitos=normalizados,
        n_novos_processados=n_novos_processados,
    )


def salvar_manifesto(resultado: ResultadoPipeline, caminho: Path) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for entrada in resultado.manifesto:
            f.write(entrada.model_dump_json() + "\n")


def carregar_manifesto(caminho: Path) -> list[EntradaManifesto]:
    entradas = []
    with Path(caminho).open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                entradas.append(EntradaManifesto.model_validate_json(linha))
    return entradas
