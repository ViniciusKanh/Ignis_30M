"""Validação local reproduzível da Sprint 6.2; sem rede ou treino de pesos."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import torch
from ignis.cli import main
from ignis.data.ingest import ingerir_diretorio
from ignis.data.pipeline import processar_diretorio, salvar_manifesto
from ignis.data.normalize import normalizar_texto
from ignis.data.pii import _PADROES, detectar_pii
from ignis.data.splits import carregar_splits, salvar_splits, verificar_contaminacao
from ignis.data.dedup import agrupar_quase_duplicatas
from ignis.tokenizer.bpe import IgnisTokenizer
from coletar_wikipedia_pt import verificar_integridade


def executar():
    corpus = Path("corpus_real/wikipedia_pt")
    saida = Path("data/sprint62_validacao_20261020")
    saida.mkdir(exist_ok=True)
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in corpus.iterdir() if p.is_file()}
    antes = hashes()
    integridade = verificar_integridade(corpus)
    assert integridade["integro"]
    brutos = ingerir_diretorio(corpus, "wikipedia-pt")
    assert len(brutos) == len({d.documento_id for d in brutos})
    resultado = processar_diretorio(corpus, "wikipedia-pt")
    manifesto = saida / "manifesto.jsonl"
    salvar_manifesto(resultado, manifesto)
    bytes_antes = manifesto.read_bytes()
    incremental = processar_diretorio(corpus, "wikipedia-pt", manifesto_existente=resultado.manifesto)
    salvar_manifesto(incremental, manifesto)
    assert incremental.n_novos_processados == 0 and manifesto.read_bytes() == bytes_antes
    # Mantém o teste histórico congelado, sem refazer o sorteio.
    splits = carregar_splits(Path("data/splits.json"))
    ids = splits.treino + splits.validacao + splits.teste
    aceitos = {e.documento_id for e in resultado.manifesto if e.status.value == "aceito"}
    assert set(ids) == aceitos and len(ids) == len(set(ids)) and not verificar_contaminacao(splits)
    por_split = {id_: nome for nome in ("treino", "validacao", "teste") for id_ in getattr(splits, nome)}
    for atributo in ("sha256", "unidade_agrupamento", "grupo_quase_duplicata"):
        grupos = {}
        for e in resultado.manifesto:
            if e.documento_id in aceitos:
                grupos.setdefault(getattr(e, atributo), set()).add(por_split[e.documento_id])
        assert all(len(g) == 1 for chave, g in grupos.items() if chave)
    salvar_splits(splits, saida / "splits.json")
    auditoria = []
    for d in brutos:
        texto = normalizar_texto(d.texto)
        # Registra contextos mascarados de todas as ocorrências, inclusive exceções.
        for tipo, padrao in _PADROES.items():
            for m in padrao.finditer(texto):
                contexto = texto[max(0,m.start()-45):m.end()+45]
                auditoria.append(dict(id=d.documento_id, tipo=tipo, inicio=m.start(), contexto=re.sub(r"\d", "#", contexto), achados_documento=detectar_pii(texto)))
    (saida / "auditoria_pii_mascarada.json").write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")
    tok = IgnisTokenizer.load(Path("data/tokenizer_dryrun_naooficial.json"))
    falhas = [d.documento_id for d in brutos if tok.decode(tok.encode(normalizar_texto(d.texto))) != normalizar_texto(d.texto)]
    assert not falhas
    comum = ["--corpus-dir", str(corpus), "--manifesto", str(manifesto), "--splits", str(saida / "splits.json"), "--tokenizer", "data/tokenizer_dryrun_naooficial.json"]
    assert main(["prepare-dataset", *comum, "--saida-base", str(saida / "tokenizado")]) == 0
    codigo = main(["audit-tokenizer", *comum, "--dataset-base", str(saida / "tokenizado"), "--textos-externos", "data/textos_externos_dryrun.txt", "--saida", str(saida / "audit-tokenizer.json")])
    assert codigo == 1
    assert antes == hashes()
    resumo = dict(integridade=integridade, corpus=resultado.relatorio(), incremental_novos=incremental.n_novos_processados, manifesto_bytes_identicos=True, corpus_hashes_preservados=antes, splits={n:len(getattr(splits,n)) for n in ("treino","validacao","teste")}, vocab_size=tok.vocab_size, round_trip_documentos=len(brutos), falhas_round_trip=falhas, audit_exit_code=codigo, torch=torch.__version__, cuda=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (saida / "evidencias.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
