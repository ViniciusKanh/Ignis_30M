"""Testes de regressao (Sprint 6.1) para scripts/coletar_wikipedia_pt.py --
NENHUM destes testes faz uma chamada de rede real: toda interacao com a API
da Wikimedia e simulada via monkeypatch de urllib.request.urlopen (ou,
quando mais simples, de _requisitar_api diretamente). O gate --confirmar
em si ja e verificado por inspecao de codigo/execucao manual (dry-run
real, documentada no relatorio da Sprint 6.1) e nao precisa de rede."""

import collections
import importlib
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
coletor = importlib.import_module("coletar_wikipedia_pt")


def test_eh_puramente_ascii_todo_o_arquivo():
    """Convencao do projeto (mesma dos scripts .ps1): nenhum acento ou
    caractere nao-ascii no codigo-fonte."""
    caminho = Path(coletor.__file__)
    conteudo = caminho.read_text(encoding="utf-8")
    nao_ascii = [c for c in conteudo if ord(c) > 127]
    assert nao_ascii == [], f"caracteres nao-ascii encontrados: {nao_ascii[:10]}"


def test_requisitar_api_retorna_resultado_em_sucesso_direto(monkeypatch):
    chamadas = []

    class _RespostaFalsa:
        def __init__(self, corpo: bytes):
            self._corpo = corpo

        def read(self):
            return self._corpo

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen_falso(requisicao, timeout=30):
        chamadas.append(requisicao)
        return _RespostaFalsa(json.dumps({"query": {"pages": {}}}).encode("utf-8"))

    monkeypatch.setattr(coletor.urllib.request, "urlopen", _urlopen_falso)
    contadores = collections.Counter()
    resultado = coletor._requisitar_api({"action": "query"}, "ua-teste", contadores)
    assert resultado == {"query": {"pages": {}}}
    assert len(chamadas) == 1
    assert sum(contadores.values()) == 0


def test_requisitar_api_trata_429_com_retry_after_e_depois_sucesso(monkeypatch):
    """O caso real observado na pratica: primeira tentativa recebe 429 com
    Retry-After; a segunda tentativa tem sucesso. Nao deve usar o backoff
    curto generico -- deve respeitar o Retry-After."""
    esperas = []
    monkeypatch.setattr(coletor.time, "sleep", lambda s: esperas.append(s))

    tentativas_feitas = {"n": 0}

    class _RespostaFalsa:
        def __init__(self, corpo: bytes):
            self._corpo = corpo

        def read(self):
            return self._corpo

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen_falso(requisicao, timeout=30):
        tentativas_feitas["n"] += 1
        if tentativas_feitas["n"] == 1:
            raise urllib.error.HTTPError(
                url="x", code=429, msg="Too Many Requests", hdrs={"Retry-After": "7"}, fp=None
            )
        return _RespostaFalsa(json.dumps({"query": {"pages": {}}}).encode("utf-8"))

    monkeypatch.setattr(coletor.urllib.request, "urlopen", _urlopen_falso)
    contadores = collections.Counter()
    resultado = coletor._requisitar_api({"action": "query"}, "ua-teste", contadores)

    assert resultado == {"query": {"pages": {}}}
    assert contadores["http_429"] == 1
    assert esperas == [7.0]  # usou o Retry-After (7s), nunca um backoff curto fixo


def test_requisitar_api_trata_429_sem_retry_after_usa_pausa_padrao(monkeypatch):
    esperas = []
    monkeypatch.setattr(coletor.time, "sleep", lambda s: esperas.append(s))
    tentativas_feitas = {"n": 0}

    class _RespostaFalsa:
        def __init__(self, corpo):
            self._corpo = corpo

        def read(self):
            return self._corpo

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen_falso(requisicao, timeout=30):
        tentativas_feitas["n"] += 1
        if tentativas_feitas["n"] == 1:
            raise urllib.error.HTTPError(url="x", code=429, msg="Too Many Requests", hdrs={}, fp=None)
        return _RespostaFalsa(json.dumps({"query": {"pages": {}}}).encode("utf-8"))

    monkeypatch.setattr(coletor.urllib.request, "urlopen", _urlopen_falso)
    contadores = collections.Counter()
    coletor._requisitar_api({"action": "query"}, "ua-teste", contadores)
    assert esperas == [coletor.PAUSA_APOS_429_S]


def test_requisitar_api_trata_erro_maxlag_no_corpo_json(monkeypatch):
    """A API pode devolver HTTP 200 com um campo 'error' no corpo quando
    maxlag e excedido, em vez de um HTTPError -- precisa ser detectado e
    tratado como retentavel, igual a 429."""
    esperas = []
    monkeypatch.setattr(coletor.time, "sleep", lambda s: esperas.append(s))
    tentativas_feitas = {"n": 0}

    class _RespostaFalsa:
        def __init__(self, corpo):
            self._corpo = corpo

        def read(self):
            return self._corpo

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen_falso(requisicao, timeout=30):
        tentativas_feitas["n"] += 1
        if tentativas_feitas["n"] == 1:
            corpo = json.dumps({"error": {"code": "maxlag", "info": "lag atrasado"}}).encode("utf-8")
        else:
            corpo = json.dumps({"query": {"pages": {}}}).encode("utf-8")
        return _RespostaFalsa(corpo)

    monkeypatch.setattr(coletor.urllib.request, "urlopen", _urlopen_falso)
    contadores = collections.Counter()
    resultado = coletor._requisitar_api({"action": "query"}, "ua-teste", contadores)
    assert resultado == {"query": {"pages": {}}}
    assert contadores["maxlag"] == 1
    assert esperas == [coletor.PAUSA_APOS_429_S]


def test_requisitar_api_esgota_tentativas_e_levanta_erro(monkeypatch):
    monkeypatch.setattr(coletor.time, "sleep", lambda s: None)

    def _urlopen_sempre_falha(requisicao, timeout=30):
        raise urllib.error.HTTPError(url="x", code=429, msg="Too Many Requests", hdrs={}, fp=None)

    monkeypatch.setattr(coletor.urllib.request, "urlopen", _urlopen_sempre_falha)
    contadores = collections.Counter()
    with pytest.raises(RuntimeError, match="apos 3 tentativas"):
        coletor._requisitar_api({"action": "query"}, "ua-teste", contadores, tentativas=3)
    assert contadores["http_429"] == 3


def test_buscar_lote_aleatorio_filtra_redirecionamento_e_desambiguacao_e_envia_maxlag(monkeypatch):
    params_capturados = {}

    def _requisitar_falso(params, user_agent, contadores, tentativas=5):
        params_capturados.update(params)
        return {
            "query": {
                "pages": {
                    "1": {"pageid": 1, "title": "Normal", "extract": "texto valido " * 20, "fullurl": "u1"},
                    "2": {"pageid": 2, "title": "Redireciona", "extract": "x", "redirect": "", "fullurl": "u2"},
                    "3": {
                        "pageid": 3,
                        "title": "Desambigua",
                        "extract": "x",
                        "pageprops": {"disambiguation": ""},
                        "fullurl": "u3",
                    },
                    "4": {"pageid": 4, "title": "Sem extrato"},
                }
            }
        }

    monkeypatch.setattr(coletor, "_requisitar_api", _requisitar_falso)
    contadores = collections.Counter()
    artigos = coletor._buscar_lote_aleatorio(50, "ua-teste", 5, contadores)

    assert params_capturados["maxlag"] == "5"
    assert params_capturados["grnlimit"] == "1"
    assert params_capturados["exlimit"] == "1"
    assert "exintro" not in params_capturados
    ids = {a["pageid"] for a in artigos}
    assert 1 in ids
    artigo_normal = next(a for a in artigos if a["pageid"] == 1)
    assert not artigo_normal["eh_redirecionamento"]
    assert not artigo_normal["eh_desambiguacao"]
    artigo_redirect = next(a for a in artigos if a["pageid"] == 2)
    assert artigo_redirect["eh_redirecionamento"]
    artigo_desambig = next(a for a in artigos if a["pageid"] == 3)
    assert artigo_desambig["eh_desambiguacao"]
    assert 4 not in ids  # sem campo 'extract' -> nunca incluido


def test_main_suspende_apos_429_persistentes_sem_passar_do_limite_de_requisicoes(monkeypatch, tmp_path):
    """Regressao central da secao 'suspensao persistente' do Sprint 6.1: se
    todo lote precisa esperar por 429, a execucao para sozinha bem antes de
    --max-requisicoes, em vez de insistir ate o limite bruto."""
    monkeypatch.setattr(coletor.time, "sleep", lambda s: None)

    def _buscar_lote_sempre_429(tamanho_lote, user_agent, maxlag, contadores):
        contadores["http_429"] += 1
        return []

    monkeypatch.setattr(coletor, "_buscar_lote_aleatorio", _buscar_lote_sempre_429)

    saida_dir = tmp_path / "wikipedia_pt"
    codigo = coletor.main(
        [
            "--confirmar",
            "--saida-dir",
            str(saida_dir),
            "--max-requisicoes",
            "1000",
            "--limite-429-consecutivos",
            "3",
            "--pausa-s",
            "0",
        ]
    )
    assert codigo == 0
    estado = json.loads((saida_dir / "_estado_coleta.json").read_text(encoding="utf-8"))
    assert estado["pageids_coletados"] == []


def test_main_coleta_artigos_simulados_ate_meta_de_tokens(monkeypatch, tmp_path):
    """Fim a fim (sem rede real): simula lotes com artigos validos e confere
    que os arquivos de saida (shard + metadados + estado) sao escritos
    corretamente e que a execucao para ao atingir a meta de tokens."""
    monkeypatch.setattr(coletor.time, "sleep", lambda s: None)
    proximo_pageid = {"n": 1}

    def _buscar_lote_falso(tamanho_lote, user_agent, maxlag, contadores):
        artigos = []
        for _ in range(tamanho_lote):
            pid = proximo_pageid["n"]
            proximo_pageid["n"] += 1
            artigos.append(
                {
                    "pageid": pid,
                    "titulo": f"Artigo {pid}",
                    "texto": ("palavra " * 500),
                    "url": f"https://pt.wikipedia.org/wiki/Artigo_{pid}",
                    "revisao": 1000 + pid,
                    "eh_redirecionamento": False,
                    "eh_desambiguacao": False,
                }
            )
        return artigos

    monkeypatch.setattr(coletor, "_buscar_lote_aleatorio", _buscar_lote_falso)

    saida_dir = tmp_path / "wikipedia_pt"
    codigo = coletor.main(
        [
            "--confirmar",
            "--saida-dir",
            str(saida_dir),
            "--meta-tokens",
            "1000",
            "--teto-mb",
            "500",
            "--tamanho-lote",
            "5",
            "--min-caracteres",
            "10",
            "--pausa-s",
            "0",
        ]
    )
    assert codigo == 0

    relatorio = coletor.verificar_integridade(saida_dir)
    assert relatorio["integro"]
    assert relatorio["n_ids_shards"] > 0

    estado = json.loads((saida_dir / "_estado_coleta.json").read_text(encoding="utf-8"))
    assert estado["tokens_aproximados"] >= 1000


def test_verificar_integridade_detecta_hash_inconsistente(tmp_path):
    saida_dir = tmp_path / "wikipedia_pt"
    saida_dir.mkdir()
    (saida_dir / "shard_0000.jsonl").write_text(
        json.dumps({"id": "wikipedia-pt-1", "texto": "texto original", "licenca": "x", "permissao_treinamento": True})
        + "\n",
        encoding="utf-8",
    )
    (saida_dir / "_metadados.jsonl").write_text(
        json.dumps({"id": "wikipedia-pt-1", "pageid": 1, "titulo": "t", "sha256_texto": "hash-errado-de-proposito"})
        + "\n",
        encoding="utf-8",
    )
    relatorio = coletor.verificar_integridade(saida_dir)
    assert not relatorio["integro"]
    assert relatorio["hashes_inconsistentes"] == ["wikipedia-pt-1"]


def test_verificar_integridade_detecta_pageid_duplicado(tmp_path):
    saida_dir = tmp_path / "wikipedia_pt"
    saida_dir.mkdir()
    (saida_dir / "shard_0000.jsonl").write_text("", encoding="utf-8")
    with (saida_dir / "_metadados.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a", "pageid": 42, "titulo": "A"}) + "\n")
        f.write(json.dumps({"id": "b", "pageid": 42, "titulo": "B"}) + "\n")
    relatorio = coletor.verificar_integridade(saida_dir)
    assert not relatorio["integro"]
    assert relatorio["pageids_duplicados"] == [42]


def test_verificar_integridade_ok_para_corpus_consistente(tmp_path):
    saida_dir = tmp_path / "wikipedia_pt"
    saida_dir.mkdir()
    texto = "conteudo de exemplo"
    import hashlib

    (saida_dir / "shard_0000.jsonl").write_text(
        json.dumps({"id": "wikipedia-pt-9", "texto": texto, "licenca": "x", "permissao_treinamento": True}) + "\n",
        encoding="utf-8",
    )
    (saida_dir / "_metadados.jsonl").write_text(
        json.dumps(
            {
                "id": "wikipedia-pt-9",
                "pageid": 9,
                "titulo": "Exemplo",
                "sha256_texto": hashlib.sha256(texto.encode("utf-8")).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    relatorio = coletor.verificar_integridade(saida_dir)
    assert relatorio["integro"]


def test_contrato_cli_nao_confirmar_nao_chama_rede(monkeypatch, tmp_path, capsys):
    """Garantia de seguranca mais basica do script: sem --confirmar, nenhuma
    chamada de rede e feita -- aqui forcamos urlopen a levantar se for
    chamado, para confirmar que main() nunca chega la."""

    def _urlopen_nao_deveria_ser_chamado(*a, **k):
        raise AssertionError("urlopen nao deveria ser chamado sem --confirmar")

    monkeypatch.setattr(coletor.urllib.request, "urlopen", _urlopen_nao_deveria_ser_chamado)
    codigo = coletor.main(["--saida-dir", str(tmp_path / "saida")])
    assert codigo == 0
    saida = capsys.readouterr().out
    assert "Nenhuma chamada de rede foi realizada" in saida


def test_metricas_na_retomada_usam_so_tokens_novos(monkeypatch, tmp_path, capsys):
    relogio = iter([0, 10, 60])
    monkeypatch.setattr(coletor.time, "monotonic", lambda: next(relogio))
    monkeypatch.setattr(coletor.time, "sleep", lambda _: None)
    coletor._salvar_estado(tmp_path / "_estado_coleta.json", dict(
        pageids_coletados=[], bytes_baixados=100, tokens_aproximados=1000, indice_arquivo_atual=0))
    def buscar(*args):
        return [dict(pageid=1, titulo="Teste", texto="palavra " * 500,
                     url="u", revisao=1, eh_redirecionamento=False, eh_desambiguacao=False)]
    monkeypatch.setattr(coletor, "_buscar_lote_aleatorio", buscar)
    assert coletor.main(["--confirmar", "--saida-dir", str(tmp_path),
                         "--max-requisicoes", "1", "--meta-tokens", "2000"]) == 0
    saida = capsys.readouterr().out
    assert "tokens aproximados por minuto (esta execucao): 500" in saida
