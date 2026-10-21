"""Mede títulos editoriais no treino/validação, sem limpar ou retokenizar dados."""

import json
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path

from ignis.data.ingest import ingerir_diretorio
from ignis.data.normalize import normalizar_texto
from ignis.training.checkpoint import sha256_arquivo
from ignis.tokenizer.bpe import IgnisTokenizer

TITULOS = {"referências", "ligações externas", "bibliografia", "notas", "ver também"}


def analisar_texto(texto):
    """Detecta somente títulos isolados; a cauda é candidata, não ruído confirmado."""
    ocorrencias = Counter()
    inicio_cauda = None
    caracteres_titulos = 0
    posicao = 0
    for linha in texto.splitlines(keepends=True):
        titulo = linha.strip().strip("#= \t").casefold()
        if titulo in TITULOS:
            ocorrencias[titulo] += 1
            caracteres_titulos += len(linha)
            if inicio_cauda is None:
                inicio_cauda = posicao
        posicao += len(linha)
    return {"ocorrencias": dict(ocorrencias),
            "caracteres_titulos": caracteres_titulos,
            "inicio_cauda_candidata": inicio_cauda}


def executar():
    base = Path("data/piloto_corpus_v1")
    manifesto_path = Path("data/manifesto_expansao_revisado.jsonl")
    tokenizer_path = base / "tokenizer_naooficial.json"
    splits_path = base / "splits.json"
    manifesto = {r["documento_id"]: r for r in (
        json.loads(linha) for linha in manifesto_path.read_text(encoding="utf-8").splitlines() if linha.strip())}
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    ids_teste = set(splits["teste"])
    ids_por_split = {nome: set(splits[nome]) for nome in ("treino", "validacao")}
    if ids_por_split["treino"] & ids_por_split["validacao"]:
        raise RuntimeError("IDs comuns entre treino e validação.")
    desejados = ids_por_split["treino"] | ids_por_split["validacao"]
    if desejados & ids_teste:
        raise RuntimeError("IDs do teste nos splits auditados.")
    tokenizer = IgnisTokenizer.load(tokenizer_path)
    textos = {}
    origens = set()
    for documento in ingerir_diretorio(Path("corpus_real/wikipedia_pt")):
        if documento.documento_id not in desejados:
            continue
        texto = normalizar_texto(documento.texto)
        registro = manifesto[documento.documento_id]
        if registro["status"] != "aceito":
            raise RuntimeError("Documento não aceito no split auditado.")
        if hashlib.sha256(texto.encode("utf-8")).hexdigest() != registro["sha256"]:
            raise RuntimeError("Texto atual diverge do manifesto: " + documento.documento_id)
        if documento.documento_id in textos:
            raise RuntimeError("ID duplicado na leitura do corpus.")
        textos[documento.documento_id] = texto
        origens.add(Path(documento.caminho_origem.rsplit(":", 1)[0]))
    if textos.keys() != desejados:
        raise RuntimeError("Documentos do split ausentes no corpus atual.")
    relatorio = {"teste_analisado": False, "corpus_alterado": False,
        "metodo": "Títulos isolados. Cauda desde o primeiro título: limite candidato, não remoção aprovada.",
        "hashes": {str(p): sha256_arquivo(p) for p in
                   [manifesto_path, splits_path, tokenizer_path, *sorted(origens)]}, "splits": {}}
    for nome, ids in ids_por_split.items():
        total_tokens = total_chars = tokens_cauda = chars_cauda = chars_titulos = 0
        documentos_com_titulos = repetidos = 0
        contagens = Counter()
        exemplos = []
        for doc_id in sorted(ids):
            texto = textos[doc_id]
            resultado = analisar_texto(texto)
            contagens.update(resultado["ocorrencias"])
            total_tokens += len(tokenizer.encode(texto))
            total_chars += len(texto)
            chars_titulos += resultado["caracteres_titulos"]
            inicio = resultado["inicio_cauda_candidata"]
            if inicio is not None:
                documentos_com_titulos += 1
                repetidos += int(any(n > 1 for n in resultado["ocorrencias"].values()))
                chars_cauda += len(texto[inicio:])
                tokens_cauda += len(tokenizer.encode(texto[inicio:], adicionar_bos_eos=False))
                if len(exemplos) < 10:
                    exemplos.append({"documento_id": doc_id, "titulos": resultado["ocorrencias"]})
        estat_path = base / f"tokenizado_{nome}.estatisticas.json"
        esperado = json.loads(estat_path.read_text(encoding="utf-8"))["n_tokens_total"]
        if total_tokens != esperado:
            raise RuntimeError(f"Tokens atuais de {nome} diferem das estatísticas do piloto.")
        for sufixo in (".sequencias.npy", ".mascaras.npy", ".estatisticas.json"):
            p = base / (f"tokenizado_{nome}" + sufixo)
            relatorio["hashes"][str(p)] = sha256_arquivo(p)
        relatorio["splits"][nome] = {"documentos": len(ids), "tokens_com_especiais": total_tokens,
            "caracteres": total_chars, "documentos_com_titulos": documentos_com_titulos,
            "percentual_documentos_com_titulos": 100 * documentos_com_titulos / len(ids),
            "documentos_com_titulo_repetido": repetidos, "ocorrencias_por_titulo": dict(contagens),
            "caracteres_linhas_titulos": chars_titulos,
            "caracteres_cauda_candidata": chars_cauda,
            "tokens_cauda_candidata_sem_especiais": tokens_cauda,
            "percentual_tokens_cauda_aproximado": 100 * tokens_cauda / total_tokens,
            "nota_tokens": "Cauda codificada separadamente; fronteira BPE pode diferir. Inclui conteúdo legítimo.",
            "exemplos_ids": exemplos}
    destino = Path("docs/experiments") / ("auditoria_editorial_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    destino.mkdir(parents=True, exist_ok=False)
    with (destino / "relatorio.json").open("x", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, indent=2, ensure_ascii=False)
    print(json.dumps(relatorio["splits"], ensure_ascii=True, indent=2))
    print("Relatorio salvo:", destino / "relatorio.json")


if __name__ == "__main__":
    executar()
