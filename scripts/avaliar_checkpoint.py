"""Avaliação experimental reproduzível, sem treinamento e sem usar o teste."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ignis.config import IgnisConfig
from ignis.inference.generate import generate
from ignis.model import IgnisModel
from ignis.tokenizer.bpe import ID_BOS, ID_EOS, IgnisTokenizer
from ignis.training.checkpoint import load_checkpoint, sha256_arquivo
from ignis.training.dataloader import criar_dataloader


def verificar_splits(metadata, caminho, permitir_legado=False):
    """Distingue ausência histórica de hash de uma divergência comprovada."""
    atual = sha256_arquivo(caminho)
    salvo = metadata.get("dataset_sha256")
    if salvo is None:
        if not permitir_legado:
            raise RuntimeError(
                "Checkpoint legado sem hash dos splits. Para avaliação apenas, "
                "use --permitir-splits-sem-hash; a origem dos splits ficará não verificada."
            )
        return {"status": "origem_nao_verificada", "sha256_checkpoint": None,
                "sha256_atual": atual}
    if salvo != atual:
        raise RuntimeError("Splits incompatíveis com o checkpoint: hashes diferentes.")
    return {"status": "hash_compativel", "sha256_checkpoint": salvo,
            "sha256_atual": atual}


@torch.inference_mode()
def executar():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/pilot-corpus-v1-fase2/step1000.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--permitir-splits-sem-hash", action="store_true",
                        help="Avalia checkpoint legado sem comprovar seus splits de origem.")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas indisponível.")

    base = Path("data/piloto_corpus_v1")
    caminho_tokenizer = base / "tokenizer_naooficial.json"
    config_path = Path("configs/ignis-30m.yaml")
    config = IgnisConfig.from_yaml(config_path).modelo
    modelo = IgnisModel(config)
    estado = load_checkpoint(args.checkpoint, modelo, restaurar_rng=False)
    metadata = estado.get("metadata", {})
    if metadata.get("tokenizer_sha256") != sha256_arquivo(caminho_tokenizer):
        raise RuntimeError("Tokenizador incompatível com o checkpoint.")
    if metadata.get("config_modelo") != config.model_dump():
        raise RuntimeError("Arquitetura incompatível com o checkpoint.")
    verificacao_splits = verificar_splits(
        metadata, base / "splits.json", args.permitir_splits_sem_hash
    )
    if verificacao_splits["status"] == "origem_nao_verificada":
        print("AVISO: checkpoint sem hash dos splits; origem dos splits não verificada.", flush=True)
    passo = estado["step"]
    del estado
    if modelo.contar_parametros() != 30416384:
        raise RuntimeError("Contagem de parâmetros incompatível.")
    modelo.to(args.device).eval()
    tokenizer = IgnisTokenizer.load(caminho_tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        raise RuntimeError("Vocabulário incompatível.")
    arquivos = [Path(args.checkpoint), config_path, caminho_tokenizer, base / "splits.json"]
    arquivos += [base / ("tokenizado_validacao" + s) for s in (".sequencias.npy", ".mascaras.npy")]
    relatorio = {
        "checkpoint": args.checkpoint, "passo_local": passo,
        "device": args.device, "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "teste_utilizado": False, "pesos_atualizados": False,
        "verificacao_splits": verificacao_splits,
        "hashes": {str(p): sha256_arquivo(p) for p in arquivos},
        "amostras": [],
    }
    loader = criar_dataloader(base / "tokenizado_validacao", batch_size=1,
                             embaralhar=False, max_seq_len=config.max_seq_len)
    soma, tokens, lotes = 0.0, 0, 0
    for entrada, alvo in loader:
        alvo = alvo.to(args.device)
        logits = modelo(entrada.to(args.device))
        loss = F.cross_entropy(logits.flatten(0, 1), alvo.flatten(),
                               ignore_index=-100, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Loss não finita na validação.")
        soma += loss.item()
        tokens += int((alvo != -100).sum().item())
        lotes += 1
    if not tokens:
        raise RuntimeError("Validação sem alvos válidos.")
    relatorio["validacao"] = {"loss": soma / tokens, "perplexidade": math.exp(soma / tokens),
                              "tokens_alvo": tokens, "lotes": lotes}
    prompts = ["O Brasil é um país", "A energia solar pode ser utilizada para",
               "Na matemática, uma função", "Um algoritmo é uma sequência de",
               "A cidade foi fundada"]
    for prompt in prompts:
        ids = [ID_BOS] + tokenizer.encode(prompt, adicionar_bos_eos=False)
        limite = min(64, 128 - len(ids))
        if limite <= 0:
            raise RuntimeError("Prompt excede o contexto experimental.")
        entrada = torch.tensor([ids], dtype=torch.long, device=args.device)
        for modo, parametros in (("greedy", {"temperature": 0.0}),
                                 ("amostragem", {"temperature": 0.8, "top_k": 40, "top_p": 0.9})):
            saida = generate(modelo, entrada, max_new_tokens=limite, seed=42,
                             eos_token_id=ID_EOS, **parametros)
            novos = saida[0, len(ids):].tolist()
            amostra = {"prompt": prompt, "modo": modo, "seed": 42,
                       "parametros": parametros, "continuacao": tokenizer.decode(novos),
                       "tokens_gerados": len(novos)}
            relatorio["amostras"].append(amostra)
            print(json.dumps(amostra, ensure_ascii=True), flush=True)
    # Um diretório novo preserva todas as avaliações anteriores.
    destino = Path("docs/experiments") / ("avaliacao_checkpoint_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    destino.mkdir(parents=True, exist_ok=False)
    with (destino / "relatorio.json").open("x", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(relatorio["validacao"], indent=2), flush=True)
    print("Relatorio salvo:", destino / "relatorio.json", flush=True)


if __name__ == "__main__":
    executar()
