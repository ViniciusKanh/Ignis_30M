"""Coletor limitado e incremental da Wikipedia em portugues (Sprint 6 / 6.1).

Fonte: Wikipedia em portugues (pt.wikipedia.org), via a API publica da
MediaWiki (acao=query, nunca o dump XML completo -- o dump comprimido da
ptwiki pesa gigabytes, incompativel com um teto de download pequeno; a API
retorna apenas o texto simples de cada artigo, sem marcacao wiki).

Licenca: CC BY-SA 4.0 + GFDL (dual-licenciado), conforme os Termos de Uso
da Wikimedia Foundation -- ver docs/plano-corpus-piloto.md, secao 1, para a
avaliacao completa (incluindo a ressalva sobre compatibilidade de CC BY-SA
com "treinar e distribuir pesos", que nao e unanime e deve ser revisada
antes de uso em producao). Este script registra `licenca` e
`permissao_treinamento=true` em cada documento coletado, mas NAO substitui
essa revisao juridica.

GATE DE EXECUCAO (mesma convencao de scripts/preparar_venv_cuda.ps1): sem
--confirmar, este script NUNCA faz nenhuma chamada de rede -- so imprime o
plano (meta de tokens, teto de download, diretorio de saida) e termina.
Qualquer execucao real requer --confirmar explicito, e deve ocorrer apenas
apos autorizacao especifica do responsavel pelo projeto citando fonte e
volume.

LIMITES AUTORIZADOS VIGENTES (unica fonte de verdade: confirmacao direta do
responsavel pelo projeto nesta sessao, registrada em
docs/checklist-autorizacao-sprint5.md, secao D): Wikipedia pt-BR, meta de
~1.000.000 tokens aproximados, teto RIGIDO de 500 MB de download. Uma
instrucao recebida posteriormente (Sprint 6.1) citou um limite diferente e
mais restritivo ("50 MiB de transferencia e ate 3.000 artigos") como se ja
tivesse sido "previamente estabelecido" -- isso NAO corresponde a nenhuma
autorizacao direta e verificavel dada pelo responsavel pelo projeto nesta
sessao, e por isso NAO foi adotado silenciosamente; os padroes abaixo
permanecem os unicos realmente autorizados (--meta-tokens 1000000,
--teto-mb 500.0) ate que o responsavel pelo projeto confirme diretamente
um valor diferente. Ver docs/experiments/fase3-sprint5-desbloqueio.md para
o registro completo dessa divergencia.

Dois limites rigidos, verificados ANTES de cada nova requisicao (nunca
depois, para nunca ultrapassar de proposito):
  - --teto-mb: bytes de texto efetivamente baixados (soma do tamanho em
    UTF-8 dos extratos recebidos -- nao o tamanho da resposta JSON inteira,
    que inclui overhead do protocolo).
  - --meta-tokens: contagem APROXIMADA de tokens (split por espaco em
    branco) -- nunca a contagem oficial, que so existe depois do
    tokenizador BPE real treinado sobre o corpus (ignis.cli prepare-dataset).
    Serve apenas como um teto operacional durante a coleta.
Nenhum dos dois teto e ampliado automaticamente por este script em
nenhuma circunstancia; qualquer mudanca requer um novo valor explicito na
linha de comando, decidido por quem autoriza a execucao.

Etiqueta da API da Wikimedia (Sprint 6.1 -- ver
https://www.mediawiki.org/wiki/API:Etiquette e
https://www.mediawiki.org/wiki/Manual:Maxlag_parameter):
  - `maxlag=5` e enviado em toda requisicao (--maxlag, padrao 5): pede para
    o servidor recusar a requisicao com um erro especifico se a replicacao
    do banco estiver atrasada, em vez de atender sob carga -- tratado como
    erro retentavel igual a um HTTP 429.
  - User-Agent identificavel por padrao (ver USER_AGENT_PADRAO); um
    contato adicional (ex.: e-mail ou URL pessoal) so e incluido se
    fornecido explicitamente via --contato -- este script NUNCA insere
    automaticamente o e-mail ou qualquer dado pessoal de quem o executa.
  - Suspensao persistente: se --limite-429-consecutivos lotes seguidos
    precisarem de espera por 429/maxlag MESMO apos a espera recomendada, a
    coleta para sozinha (estado salvo) em vez de insistir ate o limite de
    seguranca de requisicoes -- um sinal de que a API esta sob throttling
    sustentado, nao uma falha passageira.

Saida: arquivos .jsonl em --saida-dir, prontos para
`ignis.cli validate-corpus --corpus-dir <saida-dir> --fonte wikipedia-pt`.
Cada linha ja inclui `unidade_agrupamento` explicito por artigo
(wikipedia-pt-<pageid>), para nunca colapsar este provedor grande em uma
unica unidade de split (ver docs/plano-corpus-piloto.md, secao sobre
"Provedor vs. unidade de agrupamento"). Metadados completos de proveniencia
(titulo, url, revisao, data de coleta, hash) vao para um arquivo paralelo
`_metadados.jsonl`, para alimentar docs/data-card.md -- esses campos nao
sao lidos por ignis.data.ingest (que so reconhece id/texto/licenca/
permissao_treinamento/unidade_agrupamento e ignora qualquer arquivo cujo
NOME comece com `_`, incluindo este), entao nao interferem na ingestao.

Resumivel: o estado (paginas ja coletadas, bytes/tokens acumulados) e
salvo em --estado apos cada lote; rodar de novo com --confirmar continua
de onde parou, nunca baixa a mesma pagina duas vezes.

Verificacao de integridade (Sprint 6.1, sem rede, pode rodar a qualquer
momento): `--verificar` conferre, para o --saida-dir dado, que todo id em
_metadados.jsonl tem exatamente um artigo correspondente nos shards (e
vice-versa), que o sha256 registrado bate com o texto realmente salvo, e
que nao ha pageid duplicado -- sem fazer nenhuma chamada de rede e sem
modificar nada.

NAO baixa o dump completo, NAO ultrapassa os tetos configurados, NAO
amplia tetos sozinho, NAO decide sozinho a licenca de uso final (isso e
documentado, nao arbitrado por este script), e NAO executa nenhuma
chamada de rede sem --confirmar.

Uso (plano, sem baixar nada):
    python scripts/coletar_wikipedia_pt.py

Uso (execucao real, REQUER autorizacao previa especifica):
    python scripts/coletar_wikipedia_pt.py --confirmar --meta-tokens 1000000 --teto-mb 500

Uso (verificacao de integridade, sem rede):
    python scripts/coletar_wikipedia_pt.py --verificar --saida-dir corpus_real/wikipedia_pt
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://pt.wikipedia.org/w/api.php"
LICENCA_WIKIPEDIA_PT = "CC BY-SA 4.0 + GFDL (Wikimedia Foundation Terms of Use)"
MIN_CARACTERES_ARTIGO_PADRAO = 1200  # mais alto que o limiar generico de FiltrosQualidade (200) de proposito:
# a amostragem aleatoria de toda a Wikipedia em portugues inclui uma cauda enorme de
# "permastubs" (asteroides, pequenos lugares, etc.) de 1-3 frases; exigir mais
# caracteres aqui reduz o numero de artigos/requisicoes necessarios para a meta de
# tokens E melhora a qualidade/diversidade do corpus (menos "sopa de stub").
ARTIGOS_POR_ARQUIVO = 200  # apenas para nao gerar um unico .jsonl gigante; nao afeta unidade_agrupamento
PAUSA_ENTRE_REQUISICOES_S_PADRAO = 1.5  # cortesia com a API publica da Wikimedia -- ver nota sobre 429 abaixo
PAUSA_APOS_429_S = 30.0  # usada quando o servidor responde 429/maxlag sem um cabecalho Retry-After utilizavel
MAXLAG_PADRAO = 5  # ver https://www.mediawiki.org/wiki/Manual:Maxlag_parameter
LIMITE_429_CONSECUTIVOS_PADRAO = 15  # lotes seguidos com 429/maxlag -> suspensao persistente (ver docstring)
USER_AGENT_PADRAO = (
    "Ignis-30M-corpus-collector/0.1 "
    "(projeto de pesquisa em SLM pt-BR, uso nao comercial, baixo volume; "
    "ver docs/plano-corpus-piloto.md no repositorio do projeto)"
)


def _agora_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _carregar_estado(caminho: Path) -> dict:
    if caminho.exists():
        with caminho.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"pageids_coletados": [], "bytes_baixados": 0, "tokens_aproximados": 0, "indice_arquivo_atual": 0}


def _salvar_estado(caminho: Path, estado: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)


class ErroRetentavel(Exception):
    """Erro de rede/limite de taxa/maxlag ja tratado com espera -- sinaliza
    ao chamador que a tentativa seguinte pode ter sucesso, mas que esta
    especifica nao produziu dados."""


def _requisitar_api(params: dict, user_agent: str, contadores: collections.Counter, tentativas: int = 5) -> dict:
    """Chama a API da MediaWiki com um User-Agent identificavel (etiqueta da
    API da Wikimedia) e trata erros de rede/limite de taxa/maxlag com
    poucas tentativas -- nunca em loop indefinido.

    Tratamento especifico de HTTP 429 (Too Many Requests, observado na
    pratica com --tamanho-lote/--pausa-s padrao antigos) e do erro
    `maxlag` (que a API pode devolver com HTTP 200 e um campo `error` no
    corpo, em vez de um HTTPError): honra o cabecalho `Retry-After` quando
    presente (em segundos, conforme a especificacao HTTP), ou usa uma
    pausa fixa e generosa (PAUSA_APOS_429_S) quando o cabecalho nao vem ou
    nao e um numero -- nunca um backoff curto, que so re-dispara o mesmo
    limite de taxa. Cada ocorrencia de 429/maxlag/erro de rede incrementa
    o contador correspondente em `contadores`, usado depois para as
    metricas de eficiencia e para a deteccao de throttling persistente."""
    query = urllib.parse.urlencode(params)
    url = f"{API_URL}?{query}"
    ultimo_erro: Exception | None = None
    for tentativa in range(1, tentativas + 1):
        try:
            requisicao = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(requisicao, timeout=30) as resposta:
                corpo = resposta.read()
            resultado = json.loads(corpo)
            if isinstance(resultado, dict) and "error" in resultado:
                codigo_erro = resultado["error"].get("code", "")
                if codigo_erro == "maxlag":
                    contadores["maxlag"] += 1
                    print(
                        f"  AVISO: tentativa {tentativa}/{tentativas} recebeu erro 'maxlag' da API "
                        f"(replicacao atrasada no servidor) -- aguardando {PAUSA_APOS_429_S:.0f}s."
                    )
                    time.sleep(PAUSA_APOS_429_S)
                    ultimo_erro = RuntimeError(f"maxlag: {resultado['error']}")
                    continue
                raise RuntimeError(f"API retornou erro: {resultado['error']}")
            return resultado
        except urllib.error.HTTPError as exc:
            ultimo_erro = exc
            if exc.code == 429:
                contadores["http_429"] += 1
                retry_after_bruto = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    espera = float(retry_after_bruto) if retry_after_bruto is not None else PAUSA_APOS_429_S
                except ValueError:
                    espera = PAUSA_APOS_429_S
                print(
                    f"  AVISO: tentativa {tentativa}/{tentativas} recebeu HTTP 429 (limite de taxa da API) -- "
                    f"aguardando {espera:.0f}s antes de tentar de novo (nunca um backoff curto para 429)."
                )
                time.sleep(espera)
            elif 500 <= exc.code < 600:
                contadores["http_5xx"] += 1
                print(f"  AVISO: tentativa {tentativa}/{tentativas} recebeu HTTP {exc.code}; aguardando antes de tentar de novo.")
                time.sleep(3 * tentativa)
            else:
                contadores["http_outro"] += 1
                print(f"  AVISO: tentativa {tentativa}/{tentativas} falhou ({exc}); aguardando antes de tentar de novo.")
                time.sleep(3 * tentativa)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            ultimo_erro = exc
            contadores["erro_rede"] += 1
            print(f"  AVISO: tentativa {tentativa}/{tentativas} falhou ({exc}); aguardando antes de tentar de novo.")
            time.sleep(3 * tentativa)
    raise RuntimeError(f"Falha ao consultar a API da Wikipedia apos {tentativas} tentativas: {ultimo_erro}")


def _buscar_lote_aleatorio(tamanho_lote: int, user_agent: str, maxlag: int, contadores: collections.Counter) -> list[dict]:
    """Solicita um artigo completo por consulta: TextExtracts permite varios
    extratos apenas com exintro, que descartaria o corpo do artigo.
    tamanho_lote e mantido na assinatura por compatibilidade com a CLI.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": "1",
        "exlimit": "1",
        "prop": "extracts|info|pageprops|revisions",
        "explaintext": "1",
        "exsectionformat": "plain",
        "inprop": "url",
        "rvprop": "ids",
        "ppprop": "disambiguation",
        "maxlag": str(maxlag),
    }
    resultado = _requisitar_api(params, user_agent, contadores)
    paginas = resultado.get("query", {}).get("pages", {})
    if not paginas:
        contadores["lotes_sem_paginas"] += 1
    artigos = []
    for pagina in paginas.values():
        if "missing" in pagina or "extract" not in pagina:
            continue
        eh_desambiguacao = "disambiguation" in pagina.get("pageprops", {})
        artigos.append(
            {
                "pageid": pagina["pageid"],
                "titulo": pagina.get("title", ""),
                "texto": pagina.get("extract", ""),
                "url": pagina.get("fullurl", ""),
                "revisao": (pagina.get("revisions") or [{}])[0].get("revid"),
                "eh_redirecionamento": "redirect" in pagina,
                "eh_desambiguacao": eh_desambiguacao,
            }
        )
    return artigos


def verificar_integridade(saida_dir: Path) -> dict:
    """Confere, SEM nenhuma chamada de rede e sem modificar nada, que o
    corpus ja coletado em `saida_dir` esta consistente: todo id em
    _metadados.jsonl tem exatamente um artigo correspondente em algum
    shard_*.jsonl (e vice-versa), o sha256 registrado bate com o texto
    realmente salvo, e nao ha pageid duplicado. Retorna um dict com o
    relatorio; levanta FileNotFoundError se saida_dir ou _metadados.jsonl
    nao existirem."""
    saida_dir = Path(saida_dir)
    caminho_metadados = saida_dir / "_metadados.jsonl"
    if not caminho_metadados.exists():
        raise FileNotFoundError(f"Nao encontrado: {caminho_metadados}")

    metadados_por_id: dict[str, dict] = {}
    pageids_vistos: list[int] = []
    with caminho_metadados.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            m = json.loads(linha)
            metadados_por_id[m["id"]] = m
            pageids_vistos.append(m["pageid"])

    textos_por_id: dict[str, str] = {}
    for caminho_shard in sorted(saida_dir.glob("shard_*.jsonl")):
        with caminho_shard.open("r", encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                d = json.loads(linha)
                textos_por_id[d["id"]] = d["texto"]

    ids_shards = set(textos_por_id)
    ids_metadados = set(metadados_por_id)
    so_em_shards = sorted(ids_shards - ids_metadados)
    so_em_metadados = sorted(ids_metadados - ids_shards)

    hashes_inconsistentes = []
    for doc_id, texto in textos_por_id.items():
        meta = metadados_por_id.get(doc_id)
        if meta is None:
            continue
        hash_real = hashlib.sha256(texto.encode("utf-8")).hexdigest()
        if hash_real != meta.get("sha256_texto"):
            hashes_inconsistentes.append(doc_id)

    contagem_pageids = collections.Counter(pageids_vistos)
    duplicados = sorted(pid for pid, n in contagem_pageids.items() if n > 1)

    relatorio = {
        "n_ids_shards": len(ids_shards),
        "n_ids_metadados": len(ids_metadados),
        "ids_so_em_shards": so_em_shards,
        "ids_so_em_metadados": so_em_metadados,
        "hashes_inconsistentes": hashes_inconsistentes,
        "pageids_duplicados": duplicados,
        "n_pageids_unicos": len(contagem_pageids),
        "n_pageids_total": len(pageids_vistos),
        "integro": not (so_em_shards or so_em_metadados or hashes_inconsistentes or duplicados),
    }
    return relatorio


def _imprimir_relatorio_integridade(relatorio: dict) -> None:
    print("==== Verificacao de integridade (sem rede) ====")
    print(f"  IDs em shards: {relatorio['n_ids_shards']}")
    print(f"  IDs em metadados: {relatorio['n_ids_metadados']}")
    print(f"  IDs so em shards (sem metadado): {relatorio['ids_so_em_shards']}")
    print(f"  IDs so em metadados (sem shard): {relatorio['ids_so_em_metadados']}")
    print(f"  Hashes inconsistentes: {len(relatorio['hashes_inconsistentes'])} {relatorio['hashes_inconsistentes']}")
    print(f"  pageids unicos: {relatorio['n_pageids_unicos']} de {relatorio['n_pageids_total']}")
    if relatorio["pageids_duplicados"]:
        print(f"  pageids DUPLICADOS: {relatorio['pageids_duplicados']}")
    print(f"  INTEGRIDADE: {'OK' if relatorio['integro'] else 'FALHOU -- ver detalhes acima'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--confirmar",
        action="store_true",
        help="Sem esta flag, o script so imprime o plano e NAO faz nenhuma chamada de rede.",
    )
    parser.add_argument(
        "--verificar",
        action="store_true",
        help="So verifica a integridade do --saida-dir existente (sem rede, sem --confirmar) e termina.",
    )
    parser.add_argument("--meta-tokens", type=int, default=1_000_000, help="Teto aproximado de tokens a coletar")
    parser.add_argument("--teto-mb", type=float, default=500.0, help="Teto RIGIDO de bytes de texto baixados, em MB")
    parser.add_argument("--saida-dir", default="corpus_real/wikipedia_pt", help="Diretorio de saida (.jsonl)")
    parser.add_argument("--estado", default=None, help="Arquivo de estado para retomada (padrao: <saida-dir>/_estado_coleta.json)")
    parser.add_argument("--tamanho-lote", type=int, default=1, help="Compatibilidade: extratos completos usam um artigo por consulta")
    parser.add_argument("--max-requisicoes", type=int, default=500, help="Limite de seguranca de requisicoes nesta execucao")
    parser.add_argument(
        "--limite-429-consecutivos",
        type=int,
        default=LIMITE_429_CONSECUTIVOS_PADRAO,
        help="Numero de lotes SEGUIDOS que precisaram esperar por 429/maxlag antes de suspender a coleta "
        "(throttling persistente) -- estado e salvo normalmente, so execute de novo mais tarde.",
    )
    parser.add_argument(
        "--min-caracteres",
        type=int,
        default=MIN_CARACTERES_ARTIGO_PADRAO,
        help="Tamanho minimo do extrato (em caracteres) para aceitar um artigo -- mais alto reduz 'permastubs' e o numero de requisicoes necessarios",
    )
    parser.add_argument(
        "--pausa-s",
        type=float,
        default=PAUSA_ENTRE_REQUISICOES_S_PADRAO,
        help="Pausa entre requisicoes bem-sucedidas, em segundos (cortesia com a API publica)",
    )
    parser.add_argument("--maxlag", type=int, default=MAXLAG_PADRAO, help="Valor de maxlag enviado a API (etiqueta da Wikimedia)")
    parser.add_argument("--seed", type=int, default=None, help="Seed para a amostragem aleatoria (reprodutibilidade parcial -- a API 'random' nao garante reprodutibilidade exata entre execucoes)")
    parser.add_argument("--user-agent", default=USER_AGENT_PADRAO, help="User-Agent base enviado a API da Wikimedia (etiqueta da API)")
    parser.add_argument(
        "--contato",
        default=None,
        help="Contato adicional (e-mail ou URL) a anexar ao User-Agent, por etiqueta da API da Wikimedia. "
        "Opcional e NUNCA preenchido automaticamente com dados pessoais de quem executa o script -- "
        "so e enviado se fornecido explicitamente aqui.",
    )
    args = parser.parse_args(argv)
    if args.tamanho_lote != 1:
        print("AVISO: TextExtracts completo suporta um artigo por consulta; tamanho-lote ajustado para 1.")
        args.tamanho_lote = 1

    saida_dir = Path(args.saida_dir)

    if args.verificar:
        try:
            relatorio = verificar_integridade(saida_dir)
        except FileNotFoundError as exc:
            print(f"ERRO: {exc}")
            return 1
        _imprimir_relatorio_integridade(relatorio)
        return 0 if relatorio["integro"] else 1

    user_agent = args.user_agent if not args.contato else f"{args.user_agent} contato: {args.contato}"
    caminho_estado = Path(args.estado) if args.estado else saida_dir / "_estado_coleta.json"

    print("==== Plano desta execucao ====")
    print(f"  Fonte: Wikipedia em portugues (pt.wikipedia.org), via API publica")
    print(f"  Licenca a ser registrada em cada documento: {LICENCA_WIKIPEDIA_PT}")
    print(f"  Meta aproximada de tokens: {args.meta_tokens:,}")
    print(f"  Teto RIGIDO de download: {args.teto_mb:.1f} MB")
    print(f"  Diretorio de saida: {saida_dir}")
    print(f"  Arquivo de estado (retomada): {caminho_estado}")
    print(f"  Tamanho do lote por requisicao: {args.tamanho_lote} (extrato completo, sem limitar a introducao)")
    print(f"  Minimo de caracteres por artigo aceito: {args.min_caracteres}")
    print(f"  Pausa entre requisicoes: {args.pausa_s}s (mais, automaticamente, apos qualquer HTTP 429/maxlag)")
    print(f"  maxlag enviado a API: {args.maxlag}")
    print(f"  Suspensao persistente apos: {args.limite_429_consecutivos} lotes seguidos com 429/maxlag")
    print(f"  User-Agent: {user_agent}")
    print(f"  Modo: {'EXECUCAO REAL (vai baixar)' if args.confirmar else 'SOMENTE PLANO (nada sera baixado)'}")

    if not args.confirmar:
        print()
        print("Nenhuma chamada de rede foi realizada (faltou --confirmar).")
        print("So execute com --confirmar apos autorizacao explicita e especifica do responsavel pelo projeto,")
        print("citando fonte, meta de tokens e teto de download (ja documentada em")
        print("docs/checklist-autorizacao-sprint5.md para a autorizacao vigente: ~1.000.000 tokens, 500 MB).")
        return 0

    if args.seed is not None:
        random.seed(args.seed)  # nao afeta a amostragem da propria API, so qualquer logica local futura

    estado = _carregar_estado(caminho_estado)
    pageids_coletados = set(estado["pageids_coletados"])
    bytes_baixados = estado["bytes_baixados"]
    tokens_aproximados = estado["tokens_aproximados"]
    tokens_iniciais = tokens_aproximados
    indice_arquivo = estado["indice_arquivo_atual"]

    teto_bytes = args.teto_mb * 1024 * 1024
    saida_dir.mkdir(parents=True, exist_ok=True)
    caminho_metadados = saida_dir / "_metadados.jsonl"

    print()
    print(f"Estado inicial: {len(pageids_coletados)} artigos ja coletados, "
          f"{bytes_baixados / (1024 * 1024):.2f} MB, ~{tokens_aproximados:,} tokens aproximados.")

    def _caminho_arquivo_atual() -> Path:
        return saida_dir / f"shard_{indice_arquivo:04d}.jsonl"

    artigos_no_arquivo_atual = 0
    if _caminho_arquivo_atual().exists():
        with _caminho_arquivo_atual().open("r", encoding="utf-8") as f:
            artigos_no_arquivo_atual = sum(1 for _ in f)

    # Contadores estruturados por desfecho (Sprint 6.1, secao de instrumentacao) --
    # usados no resumo final e na deteccao de throttling persistente.
    contadores: collections.Counter = collections.Counter()
    tempo_inicio = time.monotonic()
    tempo_perdido_em_espera_s = 0.0
    lotes_429_consecutivos = 0
    suspenso_por_throttling = False

    n_requisicoes = 0
    while True:
        if bytes_baixados >= teto_bytes:
            print(f"\nTeto de download atingido ({bytes_baixados / (1024 * 1024):.2f} MB >= {args.teto_mb} MB). Parando.")
            break
        if tokens_aproximados >= args.meta_tokens:
            print(f"\nMeta de tokens atingida (~{tokens_aproximados:,} >= {args.meta_tokens:,}). Parando.")
            break
        if n_requisicoes >= args.max_requisicoes:
            print(f"\nLimite de seguranca de {args.max_requisicoes} requisicoes atingido nesta execucao. Parando "
                  f"(rode de novo para continuar -- o estado foi salvo).")
            break

        n_requisicoes += 1
        contadores_antes = dict(contadores)
        tempo_antes_lote = time.monotonic()
        try:
            artigos = _buscar_lote_aleatorio(args.tamanho_lote, user_agent, args.maxlag, contadores)
        except RuntimeError as exc:
            print(f"\nERRO: {exc}")
            print("Parando com o estado salvo ate aqui -- rode de novo mais tarde para continuar.")
            contadores["lotes_com_falha_final"] += 1
            break

        # um lote "precisou esperar" se algum 429/maxlag ocorreu durante ele
        # (mesmo que tenha dado certo na tentativa seguinte) -- e a base da
        # deteccao de throttling persistente (nao de erro isolado).
        teve_429_ou_maxlag_neste_lote = (
            contadores["http_429"] + contadores["maxlag"]
            > contadores_antes.get("http_429", 0) + contadores_antes.get("maxlag", 0)
        )
        tempo_perdido_em_espera_s += max(0.0, (time.monotonic() - tempo_antes_lote) - args.pausa_s) if teve_429_ou_maxlag_neste_lote else 0.0
        if teve_429_ou_maxlag_neste_lote:
            lotes_429_consecutivos += 1
        else:
            lotes_429_consecutivos = 0

        if lotes_429_consecutivos >= args.limite_429_consecutivos:
            print(
                f"\nSUSPENSAO PERSISTENTE: {lotes_429_consecutivos} lotes seguidos precisaram esperar por "
                f"429/maxlag mesmo apos a espera recomendada -- isso indica throttling sustentado da API, nao "
                f"uma falha passageira. Parando com o estado salvo; rode de novo mais tarde (ou com "
                f"--tamanho-lote menor / --pausa-s maior)."
            )
            suspenso_por_throttling = True
            break

        if not artigos:
            contadores["lotes_sem_artigos_uteis"] += 1

        aceitos_neste_lote = 0
        for artigo in artigos:
            if bytes_baixados >= teto_bytes or tokens_aproximados >= args.meta_tokens:
                break
            if artigo["pageid"] in pageids_coletados:
                contadores["duplicados"] += 1
                continue
            if artigo["eh_redirecionamento"]:
                contadores["redirecionamentos"] += 1
                continue
            if artigo["eh_desambiguacao"]:
                contadores["desambiguacoes"] += 1
                continue
            texto = artigo["texto"].strip()
            if len(texto) < args.min_caracteres:
                contadores["abaixo_minimo"] += 1
                continue

            texto_bytes = len(texto.encode("utf-8"))
            if bytes_baixados + texto_bytes > teto_bytes:
                print(f"  Pulando '{artigo['titulo']}': ultrapassaria o teto de download. Parando esta execucao.")
                bytes_baixados = teto_bytes  # forca a parada no proximo teste do loop externo
                break

            pageids_coletados.add(artigo["pageid"])
            bytes_baixados += texto_bytes
            tokens_aproximados += len(texto.split())
            contadores["aceitos"] += 1

            unidade = f"wikipedia-pt-{artigo['pageid']}"
            linha_corpus = {
                "id": unidade,
                "texto": texto,
                "licenca": LICENCA_WIKIPEDIA_PT,
                "permissao_treinamento": True,
                "unidade_agrupamento": unidade,
            }
            linha_metadados = {
                "id": unidade,
                "pageid": artigo["pageid"],
                "titulo": artigo["titulo"],
                "url": artigo["url"],
                "revisao": artigo["revisao"],
                "licenca": LICENCA_WIKIPEDIA_PT,
                "data_coleta_utc": _agora_iso(),
                "sha256_texto": hashlib.sha256(texto.encode("utf-8")).hexdigest(),
                "n_caracteres": len(texto),
            }

            if artigos_no_arquivo_atual >= ARTIGOS_POR_ARQUIVO:
                indice_arquivo += 1
                artigos_no_arquivo_atual = 0

            with _caminho_arquivo_atual().open("a", encoding="utf-8") as f:
                f.write(json.dumps(linha_corpus, ensure_ascii=False) + "\n")
            with caminho_metadados.open("a", encoding="utf-8") as f:
                f.write(json.dumps(linha_metadados, ensure_ascii=False) + "\n")
            artigos_no_arquivo_atual += 1
            aceitos_neste_lote += 1

        estado = {
            "pageids_coletados": sorted(pageids_coletados),
            "bytes_baixados": bytes_baixados,
            "tokens_aproximados": tokens_aproximados,
            "indice_arquivo_atual": indice_arquivo,
        }
        _salvar_estado(caminho_estado, estado)
        # Resumo PERIODICO por lote (nao por artigo -- Sprint 6.2, para nao
        # poluir a saida com uma linha por artigo quando --tamanho-lote e
        # grande; o detalhe completo de cada artigo continua em
        # _metadados.jsonl, nunca so no log do terminal).
        print(
            f"  lote {n_requisicoes}: +{aceitos_neste_lote} aceitos neste lote "
            f"(acumulado: {len(pageids_coletados)} artigos, {bytes_baixados / (1024 * 1024):.2f} MB, "
            f"~{tokens_aproximados:,} tokens aprox.)"
        )
        time.sleep(args.pausa_s)

    tempo_total_s = time.monotonic() - tempo_inicio
    tokens_novos = tokens_aproximados - tokens_iniciais
    n_tentativas_totais = sum(contadores[k] for k in ("http_429", "http_5xx", "http_outro", "erro_rede", "maxlag")) + n_requisicoes

    print()
    print("==== Resumo desta execucao ====")
    print(f"  Artigos coletados no total (acumulado): {len(pageids_coletados)}")
    print(f"  Bytes de texto baixados (acumulado): {bytes_baixados / (1024 * 1024):.2f} MB (teto: {args.teto_mb} MB)")
    print(f"  Tokens aproximados (acumulado, NAO e a contagem oficial): ~{tokens_aproximados:,} (meta: {args.meta_tokens:,})")
    print(f"  Arquivos de corpus em: {saida_dir}/shard_*.jsonl")
    print(f"  Metadados de proveniencia em: {caminho_metadados}")
    if suspenso_por_throttling:
        print(f"  ATENCAO: execucao suspensa por throttling persistente (429/maxlag), nao por meta/teto atingidos.")
    print()
    print("  -- Estatisticas por desfecho (nesta execucao) --")
    print(f"  requisicoes enviadas: {n_requisicoes}")
    print(f"  artigos aceitos: {contadores['aceitos']}")
    print(f"  rejeitados por tamanho minimo: {contadores['abaixo_minimo']}")
    print(f"  redirecionamentos ignorados: {contadores['redirecionamentos']}")
    print(f"  desambiguacoes ignoradas: {contadores['desambiguacoes']}")
    print(f"  duplicados (ja coletados antes): {contadores['duplicados']}")
    print(f"  lotes sem artigos uteis: {contadores['lotes_sem_artigos_uteis']}")
    print(f"  HTTP 429 (limite de taxa): {contadores['http_429']}")
    print(f"  erro 'maxlag' da API: {contadores['maxlag']}")
    print(f"  HTTP 5xx: {contadores['http_5xx']}")
    print(f"  outros erros HTTP: {contadores['http_outro']}")
    print(f"  erros de rede/timeout: {contadores['erro_rede']}")
    print()
    print("  -- Metricas de eficiencia (nesta execucao; estimativas, nao garantias) --")
    if n_requisicoes > 0:
        print(f"  artigos aceitos por requisicao: {contadores['aceitos'] / n_requisicoes:.2f}")
    total_triados = contadores["aceitos"] + contadores["abaixo_minimo"] + contadores["redirecionamentos"] + contadores["desambiguacoes"] + contadores["duplicados"]
    if total_triados > 0:
        taxa_rejeicao = 1 - (contadores["aceitos"] / total_triados)
        print(f"  taxa de rejeicao (nao-aceitos / total triado): {taxa_rejeicao:.1%}")
    if tempo_total_s > 0:
        if tokens_novos and tempo_total_s >= 1.0:
            print(f"  tokens aproximados por minuto (esta execucao): {tokens_novos / (tempo_total_s / 60):.0f}")
        else:
            print("  tokens aproximados por minuto: n/d (execucao rapida demais ou nenhum artigo novo para uma medicao confiavel)")
    print(f"  tempo total da execucao: {tempo_total_s:.1f}s; tempo estimado perdido em espera por 429/maxlag: ~{tempo_perdido_em_espera_s:.1f}s")
    if tokens_novos > 0 and tokens_aproximados < args.meta_tokens and tempo_total_s > 0:
        taxa_tokens_s = tokens_novos / tempo_total_s
        if taxa_tokens_s > 0:
            restante_s = (args.meta_tokens - tokens_aproximados) / taxa_tokens_s
            print(
                f"  tempo estimado restante para a meta de tokens, extrapolando a taxa desta execucao "
                f"(ALTA incerteza -- throttling e disponibilidade de artigos longos variam muito): "
                f"~{restante_s / 60:.0f} min"
            )
    print()
    print("PROXIMO PASSO: ingerir de verdade no pipeline (triagem de qualidade, PII, dedup):")
    print(f"  python -m ignis.cli validate-corpus --corpus-dir {saida_dir} --fonte wikipedia-pt "
          f"--manifesto-saida data/manifesto.jsonl --incremental")
    print()
    print("Lembrete: a contagem de tokens acima e uma APROXIMACAO (split por espaco), nao a contagem oficial --")
    print("essa so existe depois de 'ignis.cli prepare-dataset' com o tokenizador BPE real treinado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
