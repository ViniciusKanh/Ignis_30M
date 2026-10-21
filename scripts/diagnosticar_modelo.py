"""Diagnóstico sem treinamento: generalização, uso de contexto e indícios temáticos."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ignis.config import IgnisConfig
from ignis.data.ingest import ingerir_diretorio
from ignis.data.normalize import normalizar_texto
from ignis.model import IgnisModel
from ignis.training.checkpoint import load_checkpoint, sha256_arquivo
from ignis.training.dataloader import criar_dataloader


def embaralhar_prefixos(entradas, gerador):
    """Permuta o prefixo anterior ao último token sem tocar nesse token ou no alvo."""
    if entradas.ndim != 2 or entradas.shape[1] < 3:
        raise ValueError("Esperadas entradas [lote, contexto] com pelo menos três tokens.")
    resultado = entradas.clone()
    for i in range(len(entradas)):
        ordem = torch.randperm(entradas.shape[1] - 1, generator=gerador)
        resultado[i, :-1] = entradas[i, :-1][ordem]
    return resultado


@torch.inference_mode()
def avaliar_completo(modelo, loader, device, log_probs_unigrama):
    soma = soma_base = 0.0
    tokens = sequencias = 0
    for entrada, alvo in loader:
        validos = alvo[alvo != -100]
        soma_base -= log_probs_unigrama[validos].sum().item()
        logits = modelo(entrada.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), alvo.to(device).flatten(),
                               ignore_index=-100, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Loss não finita no diagnóstico.")
        soma += loss.item()
        tokens += validos.numel()
        sequencias += len(entrada)
    if not tokens:
        raise ValueError("Split sem tokens válidos.")
    media = soma / tokens
    return {"loss": media, "perplexidade": math.exp(media), "tokens_alvo": tokens,
            "sequencias": sequencias, "loss_unigrama_treino": soma_base / tokens}


@torch.inference_mode()
def avaliar_contexto(modelo, dataset, device, quantidade=512, seed=42):
    # Somente blocos completos: posição e comprimento iguais em todos os pares.
    elegiveis = [i for i in range(len(dataset)) if bool(dataset.mascaras[i].all())]
    if quantidade < 1 or not elegiveis:
        raise ValueError("Quantidade inválida ou validação sem blocos completos.")
    gerador_amostra = torch.Generator().manual_seed(seed)
    ordem = torch.randperm(len(elegiveis), generator=gerador_amostra)[:quantidade].tolist()
    indices = [elegiveis[i] for i in ordem]
    gerador_prefixo = torch.Generator().manual_seed(seed + 1)
    perdas = {nome: [] for nome in ("original", "prefixo_embaralhado", "ultimo_token_apenas")}
    for inicio in range(0, len(indices), 8):
        pares = [dataset[i] for i in indices[inicio:inicio + 8]]
        entradas = torch.stack([x for x, _ in pares])
        # O alvo é sempre o próximo token após o último token da entrada original.
        alvos = torch.stack([y[-1] for _, y in pares]).to(device)
        controles = {"original": entradas,
            "prefixo_embaralhado": embaralhar_prefixos(entradas, gerador_prefixo),
            "ultimo_token_apenas": entradas[:, -1:]}
        for nome, entrada in controles.items():
            logits = modelo(entrada.to(device))[:, -1]
            valores = F.cross_entropy(logits, alvos, reduction="none")
            if not torch.isfinite(valores).all():
                raise RuntimeError("Loss não finita no controle de contexto.")
            perdas[nome].extend(valores.cpu().tolist())
    medias = {nome: sum(valores) / len(valores) for nome, valores in perdas.items()}
    return {"n_alvos": len(indices), "indices_sequencias": indices, "seed": seed,
        "loss_media_por_condicao": medias,
        "delta_embaralhado_menos_original": medias["prefixo_embaralhado"] - medias["original"],
        "delta_ultimo_token_menos_original": medias["ultimo_token_apenas"] - medias["original"],
        "fracao_alvos_melhores_com_ordem_original": sum(
            a < b for a, b in zip(perdas["original"], perdas["prefixo_embaralhado"])) / len(indices),
        "metodo": "Um alvo por bloco completo de validação; prefixo permutado preserva último token, multiconjunto e comprimento.",
        "limitacoes": "Uma amostra/seed; blocos não são documentos independentes. Último-token-apenas muda também o comprimento. Não é benchmark de compreensão."}


INDICADORES = {
    "geografia_demografia": {"cidade", "município", "população", "habitantes", "localizada", "condado"},
    "esporte": {"futebol", "clube", "campeonato", "jogador", "estádio", "temporada"},
    "artes_entretenimento": {"filme", "álbum", "música", "série", "ator", "banda"},
    "historia_politica": {"guerra", "exército", "império", "rei", "presidente", "governo"},
    "matematica": {"matemática", "equação", "teorema", "matriz", "derivada", "integral"},
    "computacao": {"algoritmo", "programação", "software", "computador", "computação", "processador"},
    "energia": {"energia", "solar", "elétrica", "eletricidade", "fotovoltaica", "combustível"},
}


def indicadores_texto(texto):
    """Sinais lexicais sobrepostos, nunca rótulos temáticos validados."""
    palavras = set(re.findall(r"[^\W\d_]+", texto.casefold()))
    return {grupo: sorted(palavras & termos) for grupo, termos in INDICADORES.items()
            if len(palavras & termos) >= 2}


def auditar_composicao(base):
    splits_path = base / "splits.json"
    manifesto_path = Path("data/manifesto_expansao_revisado.jsonl")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    ids = set(splits["treino"])
    if ids & (set(splits["validacao"]) | set(splits["teste"])):
        raise RuntimeError("Sobreposição de IDs no treino.")
    manifesto = {r["documento_id"]: r for r in (
        json.loads(linha) for linha in manifesto_path.read_text(encoding="utf-8").splitlines() if linha.strip())}
    contagens, fontes, amostras, vistos = Counter(), Counter(), {}, set()
    arquivos = set()
    for doc in ingerir_diretorio(Path("corpus_real/wikipedia_pt")):
        if doc.documento_id not in ids:
            continue
        texto = normalizar_texto(doc.texto)
        registro = manifesto[doc.documento_id]
        if doc.documento_id in vistos or registro["status"] != "aceito":
            raise RuntimeError("Documento duplicado ou não aceito no treino.")
        if hashlib.sha256(texto.encode("utf-8")).hexdigest() != registro["sha256"]:
            raise RuntimeError("Texto de treino diferente do manifesto.")
        vistos.add(doc.documento_id)
        arquivos.add(Path(doc.caminho_origem.rsplit(":", 1)[0]))
        fontes[registro["fonte"]] += 1
        sinais = indicadores_texto(texto)
        if not sinais:
            contagens["sem_indicador"] += 1
        for grupo, termos in sinais.items():
            contagens[grupo] += 1
            if len(amostras.setdefault(grupo, [])) < 5:
                amostras[grupo].append({"id": doc.documento_id, "termos": termos})
    if vistos != ids:
        raise RuntimeError("Documentos de treino ausentes no corpus.")
    return {"documentos_treino": len(ids), "fontes": dict(fontes),
        "indicadores_lexicais_documentos": dict(contagens), "exemplos_ids_e_termos": amostras,
        "lexicos": {nome: sorted(termos) for nome, termos in INDICADORES.items()},
        "metodo": "Pelo menos dois termos distintos por grupo; grupos sobrepostos. Não estima proporções temáticas verdadeiras.",
        "hashes_fontes": {str(p): sha256_arquivo(p) for p in sorted(arquivos)}}


def executar():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/pilot-corpus-v1-fase3a-controle/best.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível.")
    base = Path("data/piloto_corpus_v1")
    config_path = Path("configs/ignis-30m.yaml")
    arquivos = [Path(args.checkpoint), config_path, base / "splits.json",
                base / "tokenizer_naooficial.json", Path("data/manifesto_expansao_revisado.jsonl")]
    for split in ("treino", "validacao"):
        arquivos.extend(base / (f"tokenizado_{split}" + sufixo) for sufixo in
                        (".sequencias.npy", ".mascaras.npy", ".estatisticas.json"))
    hashes = {str(p): sha256_arquivo(p) for p in arquivos}
    config = IgnisConfig.from_yaml(config_path).modelo
    modelo = IgnisModel(config)
    estado = load_checkpoint(args.checkpoint, modelo, restaurar_rng=False)
    metadata = estado.get("metadata", {})
    if metadata.get("config_modelo") != config.model_dump():
        raise RuntimeError("Arquitetura incompatível.")
    for chave, caminho in (("tokenizer_sha256", base / "tokenizer_naooficial.json"),
                           ("dataset_sha256", base / "splits.json")):
        if metadata.get(chave) != hashes[str(caminho)]:
            raise RuntimeError("Metadados incompatíveis ou ausentes: " + chave)
    passo = estado["step"]
    del estado
    if modelo.contar_parametros() != 30416384:
        raise RuntimeError("Contagem de parâmetros incorreta.")
    modelo.to(args.device).eval()
    loaders = {s: criar_dataloader(base / ("tokenizado_" + s), batch_size=8,
        embaralhar=False, max_seq_len=config.max_seq_len) for s in ("treino", "validacao")}
    treino = loaders["treino"].dataset
    alvos_treino = treino.sequencias[:, 1:][treino.mascaras[:, 1:] != 0]
    contagens = torch.bincount(alvos_treino, minlength=config.vocab_size).double() + 1.0
    log_probs = (contagens / contagens.sum()).log()
    relatorio = {"checkpoint": args.checkpoint, "passo_local": passo, "device": args.device,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "teste_utilizado": False, "pesos_atualizados": False, "hashes": hashes,
        "baseline": "Unigrama ajustado apenas nos alvos de treino, suavização aditiva alpha=1.", "splits": {}}
    for split, loader in loaders.items():
        resultado = avaliar_completo(modelo, loader, args.device, log_probs)
        relatorio["splits"][split] = resultado
        print(split, json.dumps(resultado), flush=True)
    relatorio["gap_loss_validacao_menos_treino"] = (
        relatorio["splits"]["validacao"]["loss"] - relatorio["splits"]["treino"]["loss"])
    relatorio["contexto"] = avaliar_contexto(modelo, loaders["validacao"].dataset, args.device)
    print("contexto", json.dumps({k: v for k, v in relatorio["contexto"].items()
                                  if k != "indices_sequencias"}), flush=True)
    relatorio["composicao"] = auditar_composicao(base)
    for p in arquivos:
        if hashes[str(p)] != sha256_arquivo(p):
            raise RuntimeError("Arquivo alterado durante diagnóstico: " + str(p))
    pasta = Path("docs/experiments") / ("diagnostico_modelo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    pasta.mkdir(parents=True, exist_ok=False)
    (pasta / "relatorio.json").write_text(json.dumps(relatorio, ensure_ascii=False,
        indent=2, allow_nan=False), encoding="utf-8")
    print("Relatorio salvo:", pasta / "relatorio.json", flush=True)


if __name__ == "__main__":
    executar()
