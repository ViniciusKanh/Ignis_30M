"""Gera um corpus sintético pequeno, em português, com MÚLTIPLAS FONTES
INDEPENDENTES, para validar ponta a ponta o pipeline de dados da Fase 2 e a
infraestrutura de treinamento da Fase 3 (ingestão, dedup, splits não-vazios,
tokenizador, preparo de dataset).

Escrito em Python (não em PowerShell) deliberadamente: montar JSON e texto
acentuado dentro de um script .ps1 é frágil (problemas de codificação entre
UTF-8 e a code page padrão do Windows PowerShell 5.1 já causaram falhas de
parsing reais neste projeto — ver docs/experiments/fase1-fundacao.md). Gerar
o corpus em Python, com `encoding="utf-8"` explícito (nunca "utf-8-sig", que
adicionaria um BOM), evita essa classe inteira de problema.

Estrutura gerada (diretório de saída):
    <saida>/fonte_0/doc-fonte0-0.jsonl   (um documento por arquivo .jsonl)
    <saida>/fonte_0/doc-fonte0-1.jsonl
    ...
    <saida>/fonte_5/...
    <saida>/casos-especiais/doc-*.jsonl  (duplicata exata, quase-duplicata,
                                           curto demais, PII, sem licença,
                                           sem permissão)

Cada subpasta vira uma "fonte" distinta na ingestão (`ingerir_diretorio` usa
o nome da pasta-mãe de cada arquivo como fonte quando nenhuma fonte é
passada explicitamente) — múltiplas fontes independentes são o que permite
`gerar_splits` produzir treino/validação/teste todos não-vazios de forma
genuína, em vez de depender de um único grupo que pode cair inteiro em um
só split por sorte do hash.

Nenhum destes documentos é dado real: são textos sintéticos, curtos,
genéricos e deliberadamente distintos entre si, usados apenas para exercitar
mecanicamente o pipeline (nunca para treinar um tokenizador ou modelo
"oficial" — ver docs/data-card.md e docs/plano-corpus-piloto.md para o
procedimento de um corpus real).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Seis temas distintos, usados para gerar parágrafos com vocabulário
# suficientemente diferente entre si (baixa similaridade de Jaccard), para
# que não sejam agrupados como quase-duplicatas por engano.
_TEMAS = [
    (
        "Transformers e atencao",
        "A arquitetura Transformer revolucionou o processamento de linguagem "
        "natural ao substituir recorrencia por mecanismos de atencao, "
        "permitindo capturar dependencias de longo alcance em paralelo, o "
        "que acelera o treinamento em hardware moderno com varios nucleos.",
    ),
    (
        "Diversidade linguistica do Brasil",
        "O Brasil possui uma diversidade linguistica e cultural enorme, com "
        "variacoes regionais de vocabulario, sotaque e expressoes idiomaticas "
        "entre o sul, o nordeste, o sudeste, o norte e o centro-oeste do "
        "pais, o que torna a lingua portuguesa falada aqui bastante rica.",
    ),
    (
        "Reprodutibilidade cientifica",
        "Reprodutibilidade em experimentos de aprendizado de maquina exige "
        "registrar sementes aleatorias, versoes de bibliotecas, hardware "
        "utilizado e hiperparametros escolhidos, para que outro pesquisador "
        "consiga obter resultados equivalentes repetindo o mesmo processo.",
    ),
    (
        "Deduplicacao de corpora",
        "Deduplicacao de documentos e uma etapa essencial na preparacao de "
        "corpora para pre-treinamento, pois textos repetidos podem levar um "
        "modelo a memorizar trechos especificos em vez de aprender padroes "
        "gerais da lingua, alem de distorcer metricas de avaliacao.",
    ),
    (
        "Pre-treinamento versus instrucoes",
        "A separacao entre pre-treinamento e ajuste por instrucoes e "
        "fundamental: um modelo pode apresentar baixa perplexidade em um "
        "corpus de validacao sem necessariamente ser capaz de seguir "
        "instrucoes de forma util, coerente ou segura em uma conversa real.",
    ),
    (
        "Historia da culinaria regional",
        "A culinaria regional brasileira combina influencias indigenas, "
        "africanas, portuguesas e de imigrantes europeus e asiaticos, "
        "resultando em pratos tipicos muito diferentes entre o litoral, o "
        "interior, a floresta amazonica e as regioes de fronteira do pais.",
    ),
]


def _paragrafo(tema_idx: int, variacao_idx: int) -> str:
    titulo, base = _TEMAS[tema_idx % len(_TEMAS)]
    sufixo = (
        f" Este e o paragrafo numero {variacao_idx} sobre o tema '{titulo}', "
        "escrito com uma frase final distinta para cada variacao, de modo "
        "que documentos do mesmo tema permanecam claramente diferentes entre "
        "si e nao sejam agrupados como quase-duplicatas pelo pipeline."
    )
    return base + sufixo


def _documentos_fontes(n_fontes: int, docs_por_fonte: int) -> dict[str, list[dict]]:
    """Retorna {nome_da_fonte: [documentos]} com textos distintos entre si
    (cada fonte usa um tema diferente, e cada documento dentro da fonte tem
    uma variacao diferente do mesmo tema)."""
    por_fonte: dict[str, list[dict]] = {}
    for fonte_idx in range(n_fontes):
        nome_fonte = f"fonte_{fonte_idx}"
        docs = []
        for doc_idx in range(docs_por_fonte):
            docs.append(
                {
                    "id": f"{nome_fonte}-doc{doc_idx}",
                    "texto": _paragrafo(fonte_idx, doc_idx),
                    "licenca": "CC-BY-4.0-exemplo",
                    "permissao_treinamento": True,
                }
            )
        por_fonte[nome_fonte] = docs
    return por_fonte


def _documentos_casos_especiais() -> list[dict]:
    """Documentos desenhados para exercitar cada status de triagem do
    pipeline (ver src/ignis/data/pipeline.py): aceito, rejeitado (duplicata
    exata e qualidade), quarentena (PII) e licenca_desconhecida (sem licenca
    e sem permissao, em casos separados para nao colidir por dedup exata)."""
    texto_base = _paragrafo(0, 0)  # mesmo texto do primeiro doc de fonte_0
    return [
        {
            "id": "caso-duplicata-exata",
            "texto": texto_base,  # duplicata exata de fonte_0-doc0
            "licenca": "CC-BY-4.0-exemplo",
            "permissao_treinamento": True,
        },
        {
            "id": "caso-quase-duplicata",
            "texto": _paragrafo(1, 0) + " Um trecho final extra apenas para teste de similaridade alta.",
            "licenca": "CC-BY-4.0-exemplo",
            "permissao_treinamento": True,
        },
        {
            "id": "caso-curto-demais",
            "texto": "Texto muito curto para passar no filtro de qualidade.",
            "licenca": "CC-BY-4.0-exemplo",
            "permissao_treinamento": True,
        },
        {
            "id": "caso-com-pii",
            "texto": (
                _paragrafo(2, 0)
                + " Para mais informacoes, entre em contato pelo e-mail "
                "exemplo.teste@dominio-ficticio.com.br ou pelo CPF 123.456.789-09 "
                "cadastrado no sistema de teste."
            ),
            "licenca": "CC-BY-4.0-exemplo",
            "permissao_treinamento": True,
        },
        {
            "id": "caso-sem-licenca",
            "texto": (
                "Este documento de teste nao possui nenhuma licenca declarada "
                "na origem, portanto deve ser classificado como licenca "
                "desconhecida pelo pipeline de triagem, mesmo que o conteudo "
                "em si nao apresente nenhum outro problema de qualidade, "
                "duplicacao ou presenca de dados pessoais identificaveis."
            ),
            "licenca": None,
            "permissao_treinamento": True,
        },
        {
            "id": "caso-sem-permissao",
            "texto": (
                "Este outro documento de teste possui uma licenca declarada, "
                "porem a permissao explicita de uso para treinamento de "
                "modelos de linguagem nao foi concedida pela fonte, logo o "
                "pipeline deve classifica-lo como licenca desconhecida ate "
                "que essa permissao seja obtida e registrada corretamente."
            ),
            "licenca": "CC-BY-4.0-exemplo",
            "permissao_treinamento": False,
        },
    ]


def gerar(diretorio_saida: Path, n_fontes: int = 6, docs_por_fonte: int = 4) -> dict[str, int]:
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    contagens: dict[str, int] = {}

    por_fonte = _documentos_fontes(n_fontes=n_fontes, docs_por_fonte=docs_por_fonte)
    for nome_fonte, docs in por_fonte.items():
        pasta_fonte = diretorio_saida / nome_fonte
        pasta_fonte.mkdir(parents=True, exist_ok=True)
        caminho = pasta_fonte / f"{nome_fonte}.jsonl"
        # encoding="utf-8" explicito (nunca "utf-8-sig") garante UTF-8 sem BOM.
        with open(caminho, "w", encoding="utf-8", newline="\n") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False))
                f.write("\n")
        contagens[nome_fonte] = len(docs)

    pasta_especiais = diretorio_saida / "casos-especiais"
    pasta_especiais.mkdir(parents=True, exist_ok=True)
    casos = _documentos_casos_especiais()
    caminho_especiais = pasta_especiais / "casos-especiais.jsonl"
    with open(caminho_especiais, "w", encoding="utf-8", newline="\n") as f:
        for doc in casos:
            f.write(json.dumps(doc, ensure_ascii=False))
            f.write("\n")
    contagens["casos-especiais"] = len(casos)

    return contagens


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gera um corpus sintetico multi-fonte (diretorios por fonte, "
            "UTF-8 sem BOM) para validar as Fases 2 e 3 de ponta a ponta."
        )
    )
    parser.add_argument("--saida", required=True, help="Diretorio de saida do corpus")
    parser.add_argument("--n-fontes", type=int, default=6, help="Numero de fontes distintas")
    parser.add_argument("--docs-por-fonte", type=int, default=4, help="Documentos distintos por fonte")
    args = parser.parse_args()

    contagens = gerar(Path(args.saida), n_fontes=args.n_fontes, docs_por_fonte=args.docs_por_fonte)
    total = sum(contagens.values())
    print(f"Corpus sintetico gerado em: {args.saida} ({total} documentos em {len(contagens)} fontes/pastas)")
    for nome, n in contagens.items():
        print(f"  {nome}: {n} documento(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
