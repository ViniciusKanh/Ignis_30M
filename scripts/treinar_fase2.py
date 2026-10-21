
"""Segunda fase experimental: pesos anteriores, novo otimizador e scheduler."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ignis.cli import main as executar_cli
from ignis.config import IgnisConfig, PilotoConfig
from ignis.model import IgnisModel
from ignis.seeding import set_seed
from ignis.training.checkpoint import (
    CheckpointMetadata, load_checkpoint, sha256_arquivo,
)
from ignis.training.dataloader import criar_dataloader
from ignis.training.trainer import Treinador


def executar():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iniciar-treinamento-real", action="store_true")
    args = parser.parse_args()

    config_path = "configs/train-pilot-corpus-v1-fase2.yaml"
    checkpoint = Path("checkpoints/pilot-corpus-v1/step200.pt")
    hash_checkpoint = (
        "9499591c77661688158617331e8444ac855f08ad6b0ec49615e9b097001fbd3e"
    )

    piloto = PilotoConfig.from_yaml(config_path)
    config_modelo = IgnisConfig.from_yaml(piloto.modelo.config).modelo
    config_treino = piloto.para_treinador_config()
    saida = Path(config_treino.out_dir)

    if saida.exists():
        raise RuntimeError(
            "Diretorio da fase 2 ja existe. Interrompido para preservar resultados."
        )
    if config_treino.max_steps != 1000:
        raise RuntimeError("Orcamento diferente dos 1000 passos propostos.")
    if sha256_arquivo(checkpoint) != hash_checkpoint:
        raise RuntimeError("Checkpoint diferente do avaliado anteriormente.")

    # Reutiliza todas as verifica??es existentes, sem iniciar treinamento.
    codigo = executar_cli(["train", "--config", config_path, "--device", "cuda"])
    if codigo != 0:
        raise RuntimeError("Dry-run da CLI reprovado.")

    set_seed(config_treino.seed)
    modelo = IgnisModel(config_modelo)
    estado = load_checkpoint(
        checkpoint, modelo, map_location="cpu", restaurar_rng=False
    )
    metadata_anterior = estado.get("metadata", {})

    if estado.get("step") != 200:
        raise RuntimeError("Checkpoint nao corresponde ao passo 200.")
    if metadata_anterior.get("tokenizer_sha256") != sha256_arquivo(
        piloto.dados.tokenizer
    ):
        raise RuntimeError("Tokenizador diferente do usado no piloto.")
    if metadata_anterior.get("config_modelo") != config_modelo.model_dump():
        raise RuntimeError("Arquitetura diferente da registrada no checkpoint.")

    del estado
    modelo.to(config_treino.device).eval()

    bases = {
        nome: Path(str(piloto.dados.dataset_base) + "_" + nome)
        for nome in ("treino", "validacao")
    }

    def criar_treino(epoca):
        return criar_dataloader(
            bases["treino"],
            batch_size=piloto.lote.microbatch_size,
            embaralhar=True,
            seed=piloto.execucao.seed,
            epoch=epoca,
            max_seq_len=config_modelo.max_seq_len,
        )

    validacao = criar_dataloader(
        bases["validacao"],
        batch_size=piloto.lote.microbatch_size,
        embaralhar=False,
        max_seq_len=config_modelo.max_seq_len,
    )

    entradas, _ = next(iter(validacao))
    with torch.inference_mode():
        logits = modelo(entradas.to(config_treino.device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Forward do checkpoint carregado nao finito.")

    plano = {
        "fase": 2,
        "checkpoint_origem": str(checkpoint),
        "checkpoint_sha256": hash_checkpoint,
        "passo_origem": 200,
        "passos_adicionais_maximos": config_treino.max_steps,
        "otimizador": "novo",
        "scheduler": "novo",
        "contador_local_inicial": 0,
        "configuracao": asdict(config_treino),
        "hashes": {
            str(p): sha256_arquivo(p)
            for p in (
                Path(config_path),
                Path(piloto.dados.manifesto),
                Path(piloto.dados.splits),
                Path(piloto.dados.tokenizer),
            )
        },
    }
    for base in bases.values():
        for sufixo in (".sequencias.npy", ".mascaras.npy"):
            caminho = Path(str(base) + sufixo)
            plano["hashes"][str(caminho)] = sha256_arquivo(caminho)

    print(json.dumps(plano, ensure_ascii=True, indent=2), flush=True)

    if not args.iniciar_treinamento_real:
        print("DRY-RUN DA FASE 2 APROVADO. Nenhum peso atualizado.")
        return

    saida.mkdir(parents=True, exist_ok=False)
    (saida / "origem_e_configuracao.json").write_text(
        json.dumps(plano, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    metadata = CheckpointMetadata.construir(
        config_modelo=config_modelo.model_dump(),
        tokenizer_caminho=piloto.dados.tokenizer,
        dataset_caminho=piloto.dados.splits,
        dataset_identificador=str(bases["treino"]),
    )
    treinador = Treinador(
        modelo, config_treino, metadata_checkpoint=metadata
    )
    # Nenhum limite parcial: usa toda a valida??o nesta nova fase.
    treinador.config.eval_max_batches = None
    inicial = treinador.avaliar(validacao)
    (saida / "validacao_antes_da_fase2.json").write_text(
        json.dumps(inicial, indent=2), encoding="utf-8"
    )

    resultado = treinador.treinar(criar_treino, validacao)
    resumo = asdict(resultado)
    (saida / "resultado.json").write_text(
        json.dumps(resumo, indent=2), encoding="utf-8"
    )
    print(json.dumps(resumo, indent=2), flush=True)


if __name__ == "__main__":
    executar()
