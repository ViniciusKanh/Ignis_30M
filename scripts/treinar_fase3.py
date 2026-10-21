"""Fase 3A de controle: dry-run padrão; atualização de pesos somente com flag explícita."""

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ignis.config import IgnisConfig, PilotoConfig
from ignis.model import IgnisModel
from ignis.seeding import set_seed
from ignis.tokenizer.bpe import IgnisTokenizer
from ignis.training.checkpoint import CheckpointMetadata, load_checkpoint, sha256_arquivo
from ignis.training.dataloader import criar_dataloader
from ignis.training.trainer import Treinador

CONFIG = Path("configs/train-pilot-corpus-v1-fase3a.yaml")
CHECKPOINT = Path("checkpoints/pilot-corpus-v1-fase2/step1000.pt")
SAIDA = Path("checkpoints/pilot-corpus-v1-fase3a-controle")
HASHES_ESPERADOS = {
    str(CHECKPOINT): "60dbb12aa8727cb1cafbd48cb5c4afdc44886b8f7b7438325acb6ffbccbe566f",
    "configs/ignis-30m.yaml": "a25ac7b621f80df4cd640dc46824a0c804a83c769e8a82c396198e6e0d2a20db",
    "data/manifesto_expansao_revisado.jsonl": "73f68d4bfd36ada638e92d7de0b24f1494f64d8e1ed3f5c0c09354674587e036",
    "data/piloto_corpus_v1/splits.json": "b83af90582995d37a225ffa35f2d52f975ce03b12078b4db2106df8716cdfc59",
    "data/piloto_corpus_v1/tokenizer_naooficial.json": "60bada7e0b694ed8f86c8ffdc98dadef9afa3acf5fcc5f131b38ce2f217524f9",
    "data/piloto_corpus_v1/tokenizado_treino.sequencias.npy": "1d367ad166028d46344fd95e57bacc50010d5a8041e7389a811a39bf23d56e6a",
    "data/piloto_corpus_v1/tokenizado_treino.mascaras.npy": "292e276b66040c84fca62e21047495c4d38581eb53fada1eee5e47e9b7086049",
    "data/piloto_corpus_v1/tokenizado_validacao.sequencias.npy": "301dc6532cbaa3f19a3f2a8c66c6cdf67767c0f7d760c85a23f458d972cdac98",
    "data/piloto_corpus_v1/tokenizado_validacao.mascaras.npy": "e3e17dc83a5ac5a377be2bc99968b1c72ff89bcc2aab54f119888dc89e9c1b77",
}


def validar_config(piloto):
    """Recusa mudanças implícitas no orçamento e nos dados desta proposta."""
    esperado = {
        "lr": 5e-5, "lr_min": 5e-6, "warmup_steps": 50, "max_steps": 1000,
        "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
        "grad_clip": 1.0, "grad_accum_steps": 32, "eval_every": 100,
        "eval_max_batches": None, "checkpoint_every": 100, "log_every": 10,
        "paciencia_avaliacoes": 3, "mixed_precision": "no", "seed": 42,
        "device": "cuda", "parar_em_falha_numerica": True,
        "parar_em_falha_de_memoria": True,
    }
    treino = piloto.para_treinador_config()
    for campo, valor in esperado.items():
        if getattr(treino, campo) != valor:
            raise RuntimeError("Configuração fora da proposta: " + campo)
    if Path(treino.out_dir) != SAIDA or piloto.lote.microbatch_size != 1 or piloto.dados.seq_len != 128:
        raise RuntimeError("Pasta, lote ou contexto fora da proposta.")
    if Path(piloto.modelo.config) != Path("configs/ignis-30m.yaml"):
        raise RuntimeError("Configuração arquitetural fora da proposta.")
    for campo, caminho in {
        "manifesto": "data/manifesto_expansao_revisado.jsonl",
        "splits": "data/piloto_corpus_v1/splits.json",
        "tokenizer": "data/piloto_corpus_v1/tokenizer_naooficial.json",
        "dataset_base": "data/piloto_corpus_v1/tokenizado",
    }.items():
        if Path(getattr(piloto.dados, campo)) != Path(caminho):
            raise RuntimeError("Dados fora da proposta: " + campo)
    treino.salvar_melhor = True
    return treino


@torch.inference_mode()
def avaliar_inicial(modelo, loader, device):
    modelo.eval()
    soma, tokens, lotes = 0.0, 0, 0
    for entrada, alvo in loader:
        alvo = alvo.to(device)
        logits = modelo(entrada.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), alvo.flatten(),
                               ignore_index=-100, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Validação inicial não finita.")
        soma += loss.item()
        tokens += int((alvo != -100).sum().item())
        lotes += 1
    if tokens != 84265 or lotes != 664:
        raise RuntimeError("Cobertura da validação diferente da referência.")
    return {"loss_validacao": soma / tokens, "perplexidade": math.exp(soma / tokens),
            "tokens_alvo": tokens, "n_batches_avaliados": lotes}


def preparar():
    piloto = PilotoConfig.from_yaml(CONFIG)
    treino = validar_config(piloto)
    if SAIDA.exists():
        raise RuntimeError("A pasta da fase 3A já existe; nenhum resultado será sobrescrito.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível nesta sessão.")
    hashes = {}
    for nome, esperado in HASHES_ESPERADOS.items():
        hashes[nome] = sha256_arquivo(nome)
        if hashes[nome] != esperado:
            raise RuntimeError("Arquivo diferente da referência avaliada: " + nome)
    hashes[str(CONFIG)] = sha256_arquivo(CONFIG)
    config_modelo = IgnisConfig.from_yaml(piloto.modelo.config).modelo
    set_seed(treino.seed)
    modelo = IgnisModel(config_modelo)
    estado = load_checkpoint(CHECKPOINT, modelo, restaurar_rng=False)
    metadata = estado.get("metadata", {})
    if estado.get("step") != 1000 or metadata.get("config_modelo") != config_modelo.model_dump():
        raise RuntimeError("Checkpoint de origem incompatível.")
    if metadata.get("tokenizer_sha256") != sha256_arquivo(piloto.dados.tokenizer):
        raise RuntimeError("Tokenizador incompatível com os pesos.")
    if metadata.get("dataset_sha256") != sha256_arquivo(piloto.dados.splits):
        raise RuntimeError("Splits incompatíveis com os pesos.")
    del estado
    if modelo.contar_parametros() != 30416384:
        raise RuntimeError("Contagem arquitetural incompatível.")
    if IgnisTokenizer.load(piloto.dados.tokenizer).vocab_size != config_modelo.vocab_size:
        raise RuntimeError("Vocabulário incompatível.")
    modelo.to(treino.device).eval()
    val = criar_dataloader(str(piloto.dados.dataset_base) + "_validacao",
                           batch_size=1, embaralhar=False, max_seq_len=config_modelo.max_seq_len)

    def criar_treino(epoca):
        return criar_dataloader(str(piloto.dados.dataset_base) + "_treino",
            batch_size=1, embaralhar=True, seed=treino.seed, epoch=epoca,
            max_seq_len=config_modelo.max_seq_len)

    loader_treino = criar_treino(0)
    if len(loader_treino.dataset) != 11867 or loader_treino.dataset.sequencias.shape[1] != 128:
        raise RuntimeError("Shape do treino incompatível.")
    del loader_treino
    inicial = avaliar_inicial(modelo, val, treino.device)
    if abs(inicial["loss_validacao"] - 6.132591157108642) > 1e-4:
        raise RuntimeError("Loss inicial diferente da referência; investigar antes de treinar.")
    plano = {"fase": "3A-controle", "checkpoint_origem": str(CHECKPOINT),
        "otimizador": "novo", "scheduler": "novo", "contador_local_inicial": 0,
        "passos_adicionais_maximos": treino.max_steps, "configuracao": asdict(treino),
        "teste_utilizado": False, "validacao_inicial": inicial, "hashes": hashes,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}
    return piloto, modelo, treino, val, criar_treino, plano


def executar(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iniciar-treinamento-real", action="store_true")
    args = parser.parse_args(argv)
    piloto, modelo, config, val, criar_treino, plano = preparar()
    if not args.iniciar_treinamento_real:
        plano.update({"dry_run_aprovado": True, "pesos_atualizados": False})
        pasta = Path("docs/experiments") / ("fase3a_dryrun_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        pasta.mkdir(parents=True, exist_ok=False)
        (pasta / "relatorio.json").write_text(json.dumps(plano, indent=2), encoding="utf-8")
        print(json.dumps(plano, indent=2), flush=True)
        print("DRY-RUN APROVADO. Nenhum peso atualizado. Relatorio:", pasta / "relatorio.json")
        return

    # Flag reservada à execução real após aprovação específica do responsável.
    SAIDA.mkdir(parents=True, exist_ok=False)
    (SAIDA / "origem_e_configuracao.json").write_text(json.dumps(plano, indent=2), encoding="utf-8")
    metadata = CheckpointMetadata.construir(config_modelo=modelo.config.model_dump(),
        tokenizer_caminho=piloto.dados.tokenizer, dataset_caminho=piloto.dados.splits,
        dataset_identificador=str(piloto.dados.dataset_base) + "_treino")
    treinador = Treinador(modelo, config, metadata_checkpoint=metadata)
    treinador.melhor_loss_validacao = plano["validacao_inicial"]["loss_validacao"]
    treinador.salvar_checkpoint("best.pt")
    resultado = treinador.treinar(criar_treino, val)
    resumo = {**asdict(resultado), "melhor_checkpoint": str(SAIDA / "best.pt")}
    (SAIDA / "resultado.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    print(json.dumps(resumo, indent=2), flush=True)


if __name__ == "__main__":
    executar()
