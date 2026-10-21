#!/usr/bin/env python3
"""Smoke training do pipeline de treinamento da Fase 3 (dataloader + Treinador
+ checkpoint), com um MODELO REDUZIDO e um CORPUS SINTÉTICO gerado em memória.

NÃO é o piloto real do Ignis-30M e NÃO deve ser confundido com ele:
  - o modelo aqui usa uma config minúscula (mesmos componentes arquiteturais
    do Ignis-30M -- RoPE, RMSNorm, GQA, SwiGLU -- mas dimensões muito
    menores), nunca a config oficial de 30.416.384 parâmetros;
  - os "dados" são sequências de tokens aleatórios salvas no formato real da
    Fase 2 (`salvar_dataset_tokenizado`), não texto real nem um corpus
    licenciado;
  - o objetivo é puramente de engenharia: provar que o pipeline real
    (dataloader -> Treinador -> checkpoint) atualiza parâmetros de fato e
    que retomar de um checkpoint funciona, em segundos, antes de gastar
    tempo em um piloto real com dados de verdade.

O que este script verifica e relata:
  1. Os pesos do modelo mudam de fato após `Treinador.treinar` (parâmetros
     não ficam parados -- otimização está realmente ocorrendo).
  2. A loss de treino cai em um experimento controlado de overfit em poucos
     exemplos fixos.
  3. Um treinamento interrompido em uma fronteira de época e retomado de
     checkpoint produz os MESMOS pesos finais que um treinamento contínuo
     equivalente (determinismo em CPU com a mesma seed -- ver limitações
     documentadas em `ignis.training.trainer.Treinador.treinar`).
  4. O checkpoint salvo é carregável e os metadados (hash de tokenizer e de
     dataset, quando fornecidos) são verificados.

Uso:
    python scripts/smoke_training.py
    python scripts/smoke_training.py --max-steps 20 --manter-saida

Este script NÃO baixa nada, NÃO usa dados reais e NÃO inicia nenhum
pré-treinamento de verdade -- ver docs/plano-corpus-piloto.md para o
procedimento (ainda não autorizado) de seleção de um corpus piloto real.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ignis.config import IgnisModelConfig
from ignis.data.pretrain_prep import EstatisticasPreparo, salvar_dataset_tokenizado
from ignis.model import IgnisModel
from ignis.training.checkpoint import load_checkpoint
from ignis.training.dataloader import criar_dataloader
from ignis.training.trainer import Treinador, TreinadorConfig


def _config_reduzida() -> IgnisModelConfig:
    """Mesma config minúscula usada em tests/conftest.py (`config_reduzida`)
    -- reaproveita os componentes arquiteturais do Ignis-30M em escala
    micro, nunca a arquitetura oficial."""
    return IgnisModelConfig(
        vocab_size=64,
        max_seq_len=32,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
    )


def _salvar_dataset_sintetico(
    diretorio: Path, nome: str, vocab_size: int, seq_len: int, n_sequencias: int, seed: int
) -> Path:
    """Gera sequências de tokens ALEATÓRIOS (não texto real) no formato
    exato que a Fase 2 produziria, para exercitar o dataloader/Treinador
    reais sem depender de um corpus/tokenizer de verdade."""
    rng = np.random.default_rng(seed)
    sequencias = rng.integers(low=1, high=vocab_size, size=(n_sequencias, seq_len)).astype(np.int32)
    mascaras = np.ones((n_sequencias, seq_len), dtype=np.int8)
    estatisticas = EstatisticasPreparo(
        n_documentos=n_sequencias,
        n_tokens_total=int(mascaras.sum()),
        n_sequencias=n_sequencias,
        n_sequencias_com_padding=0,
        seq_len=seq_len,
    )
    base = diretorio / nome
    salvar_dataset_tokenizado(base, sequencias, mascaras, estatisticas)
    return base


def _pesos_iguais(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        if not torch.equal(pa, pb):
            return False
    return True


def _verificar_atualizacao_de_parametros(tmp_path: Path, vocab_size: int, seq_len: int) -> None:
    print("\n[1/3] Verificando atualização real de parâmetros após optimizer.step()...")
    torch.manual_seed(0)
    model = IgnisModel(_config_reduzida())
    pesos_antes = {nome: p.detach().clone() for nome, p in model.named_parameters()}

    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino_update", vocab_size, seq_len=seq_len, n_sequencias=16, seed=1
    )
    config_treino = TreinadorConfig(
        lr=1e-2,
        warmup_steps=2,
        max_steps=5,
        eval_every=10_000,
        checkpoint_every=10_000,
        log_every=1,
        seed=42,
        device="cpu",
        out_dir=str(tmp_path / "ckpt_update"),
    )
    treinador = Treinador(model, config_treino)
    treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    mudou_algum_peso = any(
        not torch.equal(pesos_antes[nome], p.detach()) for nome, p in model.named_parameters()
    )
    if not mudou_algum_peso:
        raise RuntimeError("FALHA: nenhum parâmetro mudou após optimizer.step()")
    print("  OK: pelo menos um tensor de parâmetros mudou após o treinamento.")


def _verificar_queda_de_loss(tmp_path: Path, vocab_size: int, seq_len: int, max_steps: int) -> None:
    print("\n[2/3] Verificando queda de loss em overfit controlado...")
    torch.manual_seed(0)
    model = IgnisModel(_config_reduzida())
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino_overfit", vocab_size, seq_len=seq_len, n_sequencias=4, seed=2
    )
    config_treino = TreinadorConfig(
        lr=5e-3,
        warmup_steps=5,
        max_steps=max(max_steps, 50),
        eval_every=10_000,
        checkpoint_every=10_000,
        log_every=1,
        seed=42,
        device="cpu",
        out_dir=str(tmp_path / "ckpt_overfit"),
    )
    treinador = Treinador(model, config_treino)
    treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    from ignis.training.metrics import ler_metricas

    linhas = [l for l in ler_metricas(config_treino.out_dir + "/metricas.jsonl") if "training_loss" in l]
    if len(linhas) < 2:
        raise RuntimeError("FALHA: métricas de treino insuficientes para avaliar queda de loss")
    primeira_loss = linhas[0]["training_loss"]
    ultima_loss = linhas[-1]["training_loss"]
    print(f"  loss inicial: {primeira_loss:.4f}  ->  loss final: {ultima_loss:.4f}")
    if not ultima_loss < primeira_loss:
        raise RuntimeError(f"FALHA: loss não caiu ({primeira_loss:.4f} -> {ultima_loss:.4f})")
    print("  OK: loss de treino caiu no experimento de overfit controlado.")


def _verificar_checkpoint_e_retomada(
    tmp_path: Path, vocab_size: int, seq_len: int, max_steps: int
) -> str:
    print("\n[3/3] Verificando checkpoint e retomada (contínuo vs. interrompido/retomado)...")
    n_sequencias = 16
    batch_size = 4  # 16 / 4 = 4 passos por época -> max_steps par cai em fronteira de época
    # Com n_sequencias=16 e batch_size=4, cada epoca tem exatamente 4 passos.
    # Para que a interrupcao caia numa FRONTEIRA DE EPOCA exata (metade
    # tambem precisa ser multiplo de 4), total_passos e arredondado para
    # cima ate o proximo multiplo de 8 (assim metade = total_passos / 2 e
    # sempre multiplo de 4).
    total_passos = max(max_steps, 8)
    total_passos = ((total_passos + 7) // 8) * 8
    metade = total_passos // 2
    assert metade % 4 == 0, "calculo de fronteira de epoca inconsistente"

    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino_resume", vocab_size, seq_len=seq_len, n_sequencias=n_sequencias, seed=3
    )
    seed_compartilhada = 999

    def _novo_modelo() -> IgnisModel:
        torch.manual_seed(0)
        return IgnisModel(_config_reduzida())

    def _loader_factory(epoca: int):
        return criar_dataloader(
            dataset_base, batch_size=batch_size, embaralhar=True, seed=seed_compartilhada, epoch=epoca
        )

    def _config(out_dir: Path, max_steps: int) -> TreinadorConfig:
        return TreinadorConfig(
            lr=1e-2,
            warmup_steps=2,
            max_steps=max_steps,
            eval_every=10_000,
            checkpoint_every=10_000,
            log_every=1,
            seed=seed_compartilhada,
            device="cpu",
            out_dir=str(out_dir),
        )

    modelo_continuo = _novo_modelo()
    treinador_continuo = Treinador(modelo_continuo, _config(tmp_path / "ckpt_continuo", total_passos))
    treinador_continuo.treinar(criar_loader_treino=_loader_factory, val_loader=None)

    # IMPORTANTE: o horizonte do scheduler (`max_steps=total_passos`) e o
    # MESMO nas tres fases abaixo -- a interrupcao em `metade` passos e
    # puramente OPERACIONAL, via `parar_no_passo`, nunca via um `max_steps`
    # diferente (isso foi exatamente o bug de equivalencia diagnosticado e
    # corrigido em `Treinador.treinar` -- ver docs/limitations.md: usar um
    # `max_steps` menor para a fase interrompida muda a curva de learning
    # rate do scheduler e quebra a equivalencia com o treinamento continuo).
    modelo_interrompido = _novo_modelo()
    treinador_interrompido = Treinador(modelo_interrompido, _config(tmp_path / "ckpt_interrompido", total_passos))
    resultado_parcial = treinador_interrompido.treinar(
        criar_loader_treino=_loader_factory, val_loader=None, parar_no_passo=metade
    )
    if resultado_parcial.passos_executados != metade:
        raise RuntimeError(
            f"FALHA: esperava {metade} passos antes da interrupção, obteve "
            f"{resultado_parcial.passos_executados}"
        )
    checkpoint_intermediario = resultado_parcial.ultimo_checkpoint
    print(f"  treinamento interrompido em {metade} passos; checkpoint: {checkpoint_intermediario}")

    # IMPORTANTE: out_dir DIFERENTE do usado na fase interrompida. O
    # Treinador salva seu checkpoint final como "final.pt" dentro de
    # `out_dir` -- se o treinador retomado usasse o MESMO out_dir
    # ("ckpt_interrompido"), ao terminar ele sobrescreveria o arquivo
    # "final.pt" que e exatamente o `checkpoint_intermediario` (step=4) que
    # o proprio script verifica mais abaixo de forma independente. Bug real
    # encontrado rodando de verdade no Windows (so apareceu agora porque as
    # duas correcoes anteriores -- max_steps/parar_no_passo e
    # safetensors/tie_word_embeddings -- bloqueavam a execucao antes de
    # chegar aqui).
    treinador_retomado = Treinador(
        modelo_interrompido, _config(tmp_path / "ckpt_interrompido_retomado", total_passos)
    )
    step_retomado = treinador_retomado.retomar_de_checkpoint(checkpoint_intermediario)
    if step_retomado != metade:
        raise RuntimeError(f"FALHA: retomada não restaurou o step esperado ({metade}), obteve {step_retomado}")
    print(f"  retomado de checkpoint com sucesso no step {step_retomado}")
    treinador_retomado.treinar(criar_loader_treino=_loader_factory, val_loader=None)

    if not _pesos_iguais(modelo_continuo, modelo_interrompido):
        raise RuntimeError(
            "FALHA: pesos finais do treinamento contínuo e do "
            "interrompido/retomado divergiram (mesma seed, fronteira de época)"
        )
    print("  OK: pesos finais idênticos entre treinamento contínuo e interrompido/retomado.")

    # Verifica também que o checkpoint final é carregável de forma independente.
    modelo_verificacao = _novo_modelo()
    estado = load_checkpoint(checkpoint_intermediario, modelo_verificacao, optimizer=None)
    if estado["step"] != metade:
        raise RuntimeError("FALHA: step salvo no checkpoint não corresponde ao esperado")
    print("  OK: checkpoint intermediário é carregável de forma independente.")

    return checkpoint_intermediario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-steps", type=int, default=10,
        help="Número aproximado de passos usado nas verificações 2 e 3 (ajustado internamente para múltiplos de 4).",
    )
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument(
        "--manter-saida", action="store_true",
        help="Não apagar o diretório temporário de checkpoints/métricas ao final (para inspeção manual).",
    )
    args = parser.parse_args()

    vocab_size = _config_reduzida().vocab_size
    tmp_dir = Path(tempfile.mkdtemp(prefix="ignis_smoke_training_"))
    print(f"Diretório temporário de trabalho: {tmp_dir}")
    print(
        "ATENÇÃO: este é um smoke test de engenharia com modelo REDUZIDO e "
        "dados SINTÉTICOS -- não é o piloto real do Ignis-30M."
    )

    sucesso = False
    try:
        _verificar_atualizacao_de_parametros(tmp_dir, vocab_size, args.seq_len)
        _verificar_queda_de_loss(tmp_dir, vocab_size, args.seq_len, args.max_steps)
        _verificar_checkpoint_e_retomada(tmp_dir, vocab_size, args.seq_len, args.max_steps)
        sucesso = True
    finally:
        if args.manter_saida:
            print(f"\n--manter-saida ativo: diretório preservado em {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    if sucesso:
        print("=== smoke training: TODAS AS VERIFICAÇÕES PASSARAM ===")
        print(
            "Isto confirma que dataloader + Treinador + checkpoint funcionam "
            "mecanicamente. NÃO valida qualidade de dados, hiperparâmetros do "
            "piloto real nem desempenho em hardware -- ver "
            "scripts/benchmark_training.py e docs/plano-corpus-piloto.md."
        )
        return 0
    print("=== smoke training: FALHOU ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
