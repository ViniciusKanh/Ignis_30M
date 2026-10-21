"""Particionamento reproduzível em treino/validação/teste.

Agrupa documentos em "unidades" indivisíveis antes de atribuir splits, por
COMPONENTES CONEXAS de um grafo onde dois documentos são ligados quando: (a)
pertencem ao mesmo grupo de quase-duplicata (SEMPRE, incondicionalmente), ou
(b) pertencem à mesma UNIDADE DE AGRUPAMENTO (apenas quando
`agrupar_por_fonte=True`). Isso garante transitivamente que:

- Documentos quase-duplicados NUNCA ficam em splits diferentes, mesmo que
  venham de fontes diferentes (ex.: um texto plagiado/copiado entre duas
  fontes distintas do corpus).
- Com `agrupar_por_fonte=True` (padrão), TODOS os documentos de uma mesma
  unidade de agrupamento caem no mesmo split — a justificativa usual é
  evitar vazamento por correlação (não faz sentido ter metade de um mesmo
  livro/artigo/shard no treino e a outra metade no teste).

PROVEDOR (`fonte`) vs. UNIDADE DE AGRUPAMENTO (`unidade_agrupamento`)
— distinção introduzida na Sprint 5, ver `ignis.data.ingest`: a chave usada
aqui é `entrada.unidade_agrupamento`, com fallback para `entrada.fonte`
quando `unidade_agrupamento` não foi informada (compatibilidade retroativa
total — todo manifesto gerado antes da Sprint 5 não tem esse campo, então o
comportamento é idêntico ao anterior). A razão da distinção: agrupar
estritamente por `fonte` fazia sentido para fontes pequenas e coesas, mas
colapsa um PROVEDOR inteiro (ex.: toda a Wikipédia em português, ingerida
com `--fonte wikipedia-pt`) em uma única unidade indivisível — mesmo que o
provedor contenha milhares de artigos independentes entre si. Isso
impossibilita splits de validação/teste realistas para corpora dominados
por um provedor grande, e esconde o viés de ter 100% de um split vindo de
uma só fonte. `unidade_agrupamento` resolve isso: ao ingerir um provedor
grande em várias subpastas (shards), cada subpasta vira uma unidade
independente por padrão (nome da pasta), enquanto `fonte` continua
constante para fins de auditoria/atribuição — ver
`ignis.data.splits.contar_unidades_por_fonte` para a auditoria por fonte, e
`docs/plano-corpus-piloto.md`.

Uma implementação anterior usava a chave `f"{fonte}::{grupo}"`, que NÃO
garantia o agrupamento por fonte para documentos que não fossem também
quase-duplicatas entre si (cada documento distinto de uma mesma fonte virava
uma unidade independente, na prática ignorando `agrupar_por_fonte`) — foi
substituída por esta abordagem baseada em componentes conexas.

A atribuição usa um hash determinístico (SHA-256), não o `hash()` nativo do
Python (que varia entre execuções via PYTHONHASHSEED) — por isso é
reproduzível entre máquinas e execuções.

Algoritmo de atribuição às três partições: as unidades (componentes conexas)
são ordenadas de forma determinística-porém-pseudoaleatória via hash com
seed; em seguida:

1. Se houver unidades suficientes (pelo menos uma por split com proporção
   > 0), cada split com proporção > 0 recebe primeiro uma unidade "garantida"
   (as unidades de menor proporção têm prioridade na reserva, para não ficar
   sem representação quando as proporções são muito desiguais, ex. 90/5/5).
2. As unidades restantes são distribuídas por um método de maior resto
   (apportionment), favorecendo a cada passo o split cujo número de
   documentos já atribuídos está mais abaixo da proporção alvo.
3. Se NÃO houver unidades suficientes para a garantia do passo 1, a função
   levanta `GruposInsuficientesError` em vez de devolver silenciosamente um
   split vazio — isto existe especificamente para impedir uma "aprovação
   enganosa" de um pipeline que nunca exercitou validação/teste de verdade.

Documentos relacionados (mesma unidade) NUNCA são separados entre splits
apenas para preencher uma proporção — a unidade inteira vai para um único
split.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, model_validator

from .manifest import EntradaManifesto, StatusDocumento

NOMES_SPLITS = ("treino", "validacao", "teste")


class GruposInsuficientesError(ValueError):
    """Não há grupos (unidades) independentes suficientes no corpus aceito
    para preencher, com pelo menos um documento cada, todos os splits que
    têm proporção > 0. Levantada em vez de devolver splits vazios, para que
    uma validação ponta a ponta não seja aprovada de forma enganosa."""

    def __init__(self, n_unidades: int, splits_necessarios: list[str], proporcoes: "ProporcoesSplit"):
        self.n_unidades = n_unidades
        self.splits_necessarios = splits_necessarios
        self.proporcoes = proporcoes
        super().__init__(
            f"grupos independentes insuficientes para gerar todos os splits solicitados: "
            f"{n_unidades} grupo(s)/unidade(s) disponível(is) (após agrupar por fonte e/ou "
            f"quase-duplicata), mas {len(splits_necessarios)} split(s) com proporção > 0 "
            f"precisam de pelo menos 1 unidade cada ({', '.join(splits_necessarios)}; "
            f"proporções solicitadas: treino={proporcoes.treino}, validacao={proporcoes.validacao}, "
            f"teste={proporcoes.teste}). Aumente a diversidade do corpus aceito (mais documentos de "
            f"fontes e/ou grupos de quase-duplicata distintos) ou ajuste as proporções/zere a "
            f"proporção de um split antes de gerar os splits. Documentos relacionados não são "
            f"forçados a conjuntos diferentes apenas para preencher proporções."
        )


class ProporcoesSplit(BaseModel):
    treino: float = 0.9
    validacao: float = 0.05
    teste: float = 0.05

    @model_validator(mode="after")
    def _validar_soma(self) -> "ProporcoesSplit":
        soma = self.treino + self.validacao + self.teste
        if abs(soma - 1.0) > 1e-6:
            raise ValueError(f"as proporções devem somar 1.0; somaram {soma}")
        return self

    def ativos(self) -> list[str]:
        """Nomes dos splits com proporção estritamente positiva."""
        return [nome for nome in NOMES_SPLITS if getattr(self, nome) > 0.0]


class AtribuicaoSplits(BaseModel):
    seed: int
    proporcoes: ProporcoesSplit
    agrupar_por_fonte: bool
    n_unidades: int
    treino: list[str]
    validacao: list[str]
    teste: list[str]

    def proporcoes_realizadas(self) -> dict[str, float]:
        """Proporção real de DOCUMENTOS (não de unidades) em cada split,
        para comparar com `proporcoes` (a proporção solicitada). Pode
        divergir da proporção solicitada quando o número de unidades é
        pequeno ou quando unidades têm tamanhos desiguais — isso é esperado
        e documentado, nunca escondido."""
        total = len(self.treino) + len(self.validacao) + len(self.teste)
        if total == 0:
            return {"treino": 0.0, "validacao": 0.0, "teste": 0.0}
        return {
            "treino": len(self.treino) / total,
            "validacao": len(self.validacao) / total,
            "teste": len(self.teste) / total,
        }

    def relatorio(self) -> dict:
        return {
            "seed": self.seed,
            "agrupar_por_fonte": self.agrupar_por_fonte,
            "n_unidades_independentes": self.n_unidades,
            "contagem_documentos": {
                "treino": len(self.treino),
                "validacao": len(self.validacao),
                "teste": len(self.teste),
            },
            "proporcoes_solicitadas": self.proporcoes.model_dump(),
            "proporcoes_realizadas": self.proporcoes_realizadas(),
        }


def _hash_estavel(chave: str, seed: int) -> float:
    import hashlib

    h = hashlib.sha256(f"{seed}:{chave}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(16**16)


def _componentes_conexas(
    aceitos: list[EntradaManifesto], agrupar_por_fonte: bool
) -> dict[str, list[str]]:
    """Une documentos em componentes conexas via union-find: sempre por
    grupo de quase-duplicata, e também por fonte quando `agrupar_por_fonte`
    é verdadeiro. Retorna {chave_da_componente: [documento_ids]}."""
    pai: dict[str, str] = {e.documento_id: e.documento_id for e in aceitos}

    def achar(x: str) -> str:
        while pai[x] != x:
            x = pai[x]
        return x

    def unir(a: str, b: str) -> None:
        ra, rb = achar(a), achar(b)
        if ra != rb:
            pai[ra] = rb

    por_grupo_dup: dict[str, list[str]] = defaultdict(list)
    por_unidade: dict[str, list[str]] = defaultdict(list)
    for entrada in aceitos:
        grupo = entrada.grupo_quase_duplicata or entrada.documento_id
        por_grupo_dup[grupo].append(entrada.documento_id)
        # `unidade_agrupamento` tem prioridade; cai para `fonte` quando
        # ausente (manifestos anteriores à Sprint 5) — ver docstring do
        # módulo para a distinção provedor vs. unidade de agrupamento.
        chave_unidade = entrada.unidade_agrupamento or entrada.fonte
        por_unidade[chave_unidade].append(entrada.documento_id)

    for ids_do_grupo in por_grupo_dup.values():
        for outro in ids_do_grupo[1:]:
            unir(ids_do_grupo[0], outro)

    if agrupar_por_fonte:
        for ids_da_unidade in por_unidade.values():
            for outro in ids_da_unidade[1:]:
                unir(ids_da_unidade[0], outro)

    componentes: dict[str, list[str]] = defaultdict(list)
    for entrada in aceitos:
        componentes[achar(entrada.documento_id)].append(entrada.documento_id)
    return componentes


def contar_unidades_por_fonte(
    manifesto: list[EntradaManifesto], agrupar_por_fonte: bool = True
) -> dict[str, int]:
    """Para cada `fonte` presente entre os documentos ACEITOS, conta quantas
    unidades de agrupamento (componentes conexas, ver `_componentes_conexas`)
    independentes ela contribui. Uso: auditoria de concentração em
    `ignis.cli report` (`distribuicao_por_fonte`) para detectar o caso em
    que uma fonte com muitos documentos, na prática, forma uma ÚNICA unidade
    indivisível — sinal de que a granularidade de agrupamento está errada
    (todo documento daquela fonte sempre cairá no mesmo split, por maior que
    seja). Quando uma unidade contém documentos de mais de uma fonte (via
    quase-duplicata entre fontes), ela é contada para cada fonte envolvida."""
    aceitos = [e for e in manifesto if e.status == StatusDocumento.ACEITO]
    componentes = _componentes_conexas(aceitos, agrupar_por_fonte)

    doc_para_componente: dict[str, str] = {}
    for chave_componente, ids in componentes.items():
        for doc_id in ids:
            doc_para_componente[doc_id] = chave_componente

    unidades_por_fonte: dict[str, set[str]] = defaultdict(set)
    for entrada in aceitos:
        chave = doc_para_componente.get(entrada.documento_id)
        if chave is not None:
            unidades_por_fonte[entrada.fonte].add(chave)

    return {fonte: len(chaves) for fonte, chaves in unidades_por_fonte.items()}


def gerar_splits(
    manifesto: list[EntradaManifesto],
    seed: int = 42,
    proporcoes: ProporcoesSplit | None = None,
    agrupar_por_fonte: bool = True,
) -> AtribuicaoSplits:
    proporcoes = proporcoes or ProporcoesSplit()

    aceitos = [e for e in manifesto if e.status == StatusDocumento.ACEITO]

    grupos = _componentes_conexas(aceitos, agrupar_por_fonte)

    # Ordem determinística-porém-pseudoaleatória das unidades, via hash com seed.
    unidades_ordenadas = sorted(grupos.keys(), key=lambda u: _hash_estavel(u, seed))
    n_unidades = len(unidades_ordenadas)

    splits_ativos = proporcoes.ativos()

    if n_unidades == 0:
        # Nenhum documento aceito: não há o que particionar. Isto não é um
        # erro de "grupos insuficientes" (não há ambiguidade sobre o que
        # fazer com zero documentos), apenas resulta em três splits vazios.
        return AtribuicaoSplits(
            seed=seed,
            proporcoes=proporcoes,
            agrupar_por_fonte=agrupar_por_fonte,
            n_unidades=0,
            treino=[],
            validacao=[],
            teste=[],
        )

    if n_unidades < len(splits_ativos):
        raise GruposInsuficientesError(n_unidades, splits_ativos, proporcoes)

    atribuicao_unidades: dict[str, str] = {}

    # Passo 1 — reserva: garante pelo menos 1 unidade para cada split ativo.
    # As unidades de MENOR proporção reservam primeiro (maior risco de ficar
    # sem nenhuma unidade quando as proporções são muito desiguais).
    splits_por_proporcao_crescente = sorted(splits_ativos, key=lambda nome: getattr(proporcoes, nome))
    indice_unidade = 0
    for nome_split in splits_por_proporcao_crescente:
        atribuicao_unidades[unidades_ordenadas[indice_unidade]] = nome_split
        indice_unidade += 1

    # Passo 2 — maior resto / deficit round-robin para as unidades restantes.
    contagem_unidades: dict[str, int] = {nome: 0 for nome in NOMES_SPLITS}
    for nome_split in atribuicao_unidades.values():
        contagem_unidades[nome_split] += 1

    unidades_restantes = unidades_ordenadas[indice_unidade:]
    total_processado = len(atribuicao_unidades)
    for unidade in unidades_restantes:
        total_processado += 1
        # déficit = quanto o split "deveria" ter até agora menos quanto já tem.
        deficits = {
            nome: (getattr(proporcoes, nome) * total_processado) - contagem_unidades[nome]
            for nome in splits_ativos
        }
        escolhido = max(splits_ativos, key=lambda nome: deficits[nome])
        atribuicao_unidades[unidade] = escolhido
        contagem_unidades[escolhido] += 1

    resultado: dict[str, list[str]] = {"treino": [], "validacao": [], "teste": []}
    for unidade, nome_split in atribuicao_unidades.items():
        resultado[nome_split].extend(grupos[unidade])

    return AtribuicaoSplits(
        seed=seed,
        proporcoes=proporcoes,
        agrupar_por_fonte=agrupar_por_fonte,
        n_unidades=n_unidades,
        treino=sorted(resultado["treino"]),
        validacao=sorted(resultado["validacao"]),
        teste=sorted(resultado["teste"]),
    )


def verificar_contaminacao(atribuicao: AtribuicaoSplits) -> list[str]:
    """Retorna os documento_ids que aparecem em mais de um split (deveria
    ser sempre uma lista vazia; é usada como verificação de segurança)."""
    contagem: dict[str, int] = defaultdict(int)
    for lista in (atribuicao.treino, atribuicao.validacao, atribuicao.teste):
        for doc_id in lista:
            contagem[doc_id] += 1
    return [doc_id for doc_id, n in contagem.items() if n > 1]


def salvar_splits(atribuicao: AtribuicaoSplits, caminho: Path) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(atribuicao.model_dump_json(indent=2), encoding="utf-8")


def carregar_splits(caminho: Path) -> AtribuicaoSplits:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return AtribuicaoSplits.model_validate(dados)
