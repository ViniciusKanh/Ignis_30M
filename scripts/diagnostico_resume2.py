#!/usr/bin/env python3
"""Diagnostico v2 -- corrige um erro do diagnostico anterior (que usava
chamadas incrementais a `treinar()`, o que mascarava o bug real por
afetar `epoca_atual` igualmente nos dois lados da comparacao).

Aqui reproduzimos EXATAMENTE a estrutura do teste real que falha:
  - caminho continuo: UMA UNICA chamada a treinar(max_steps=8).
  - caminho retomado: UMA chamada a treinar(max_steps=4), depois
    checkpoint + retomar_de_checkpoint, depois UMA UNICA chamada a
    treinar(max_steps=8) no Treinador retomado.

Mas em vez de so comparar o resultado FINAL, replicamos manualmente o
corpo do loop de `Treinador.treinar()` para a EPOCA 1 (passos 5-8) nos
dois caminhos, comparando os pesos apos CADA um dos 4 lotes dessa epoca,
para achar o exato ponto em que a divergencia comeca.

RESSALVA registrada apos a causa raiz ter sido confirmada (ver
docs/limitations.md, secao "Causa raiz real da divergencia em resume"):
a funcao `preparar_continuo_ate_fim_da_epoca0()` abaixo usa
`cfg(tmp / "c2", 4)` -- ou seja, max_steps=4 -- como sua referencia
"continua", o que NAO corresponde ao `t_cont` da Parte A/B (max_steps=8).
Isso significa que os numeros desta Parte C (primeira loss identica,
gradient norm identica, divergencia "apos o optimizer step" no primeiro
lote da epoca 1) comparam dois caminhos que compartilham o MESMO
max_steps=4 incorreto, nao o mismatch real de max_steps=4-vs-8 que as
Partes A/B reproduzem corretamente. A causa raiz real (mismatch de
`max_steps` alterando o scheduler de LR) provavelmente ja se manifesta
ANTES da epoca 1, durante a propria fase interrompida -- ver
docs/limitations.md para a analise completa e a correcao aplicada
(parametro `parar_no_passo` em `Treinador.treinar`). Os resultados da
Parte C abaixo nao devem mais ser usados para localizar o ponto exato de
divergencia.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from ignis.config import IgnisModelConfig
from ignis.data.pretrain_prep import EstatisticasPreparo, salvar_dataset_tokenizado
from ignis.model import IgnisModel
from ignis.training.dataloader import criar_dataloader
from ignis.training.trainer import Treinador, TreinadorConfig


def config_reduzida() -> IgnisModelConfig:
    return IgnisModelConfig(
        vocab_size=64, max_seq_len=32, n_layers=2, d_model=32,
        n_heads=4, n_kv_heads=2, head_dim=8, intermediate_size=64,
    )


def novo_modelo() -> IgnisModel:
    torch.manual_seed(0)
    return IgnisModel(config_reduzida())


def pesos_iguais(a, b) -> tuple[bool, str]:
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        if not torch.equal(pa, pb):
            diff = (pa - pb).abs().max().item()
            return False, f"{na} (diff maximo={diff:.3e})"
    return True, ""


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ignis_diag_resume2_"))
    print(f"diretorio temporario: {tmp}")

    rng = np.random.default_rng(0)
    seq_len, n_sequencias, batch_size, seed = 8, 16, 4, 999
    sequencias = rng.integers(1, 64, size=(n_sequencias, seq_len)).astype(np.int32)
    mascaras = np.ones((n_sequencias, seq_len), dtype=np.int8)
    stats = EstatisticasPreparo(
        n_documentos=n_sequencias, n_tokens_total=int(mascaras.sum()),
        n_sequencias=n_sequencias, n_sequencias_com_padding=0, seq_len=seq_len,
    )
    base = tmp / "treino"
    salvar_dataset_tokenizado(base, sequencias, mascaras, stats)

    def loader(epoca: int):
        return criar_dataloader(base, batch_size=batch_size, embaralhar=True, seed=seed, epoch=epoca)

    def cfg(out: Path, max_steps: int) -> TreinadorConfig:
        return TreinadorConfig(
            lr=1e-2, warmup_steps=2, max_steps=max_steps, eval_every=10_000,
            checkpoint_every=10_000, log_every=1, seed=seed, device="cpu", out_dir=str(out),
        )

    print("\n[A] Caminho CONTINUO: UMA chamada treinar(max_steps=8)")
    m_cont = novo_modelo()
    t_cont = Treinador(m_cont, cfg(tmp / "cont", 8))
    t_cont.treinar(criar_loader_treino=loader, val_loader=None)
    print(f"  step final: {t_cont.step}  epoca_atual final: {t_cont.epoca_atual}")

    print("\n[B] Caminho INTERROMPIDO+RETOMADO: treinar(max_steps=4), depois retomar, depois UMA chamada treinar(max_steps=8)")
    m_int = novo_modelo()
    t_int = Treinador(m_int, cfg(tmp / "int", 4))
    resultado = t_int.treinar(criar_loader_treino=loader, val_loader=None)
    ckpt = resultado.ultimo_checkpoint
    print(f"  apos interromper: step={t_int.step} epoca_atual={t_int.epoca_atual} checkpoint={ckpt}")

    m_resumed = novo_modelo()
    t_resumed = Treinador(m_resumed, cfg(tmp / "resumed", 8))
    t_resumed.retomar_de_checkpoint(ckpt)
    print(f"  apos retomar: step={t_resumed.step} epoca_atual={t_resumed.epoca_atual}")
    t_resumed.treinar(criar_loader_treino=loader, val_loader=None)
    print(f"  step final: {t_resumed.step}  epoca_atual final: {t_resumed.epoca_atual}")

    ok, onde = pesos_iguais(m_cont, m_resumed)
    print("\n[RESULTADO FINAL - reproduz o teste real]")
    print("  identicos:", "OK" if ok else f"DIVERGIU em {onde}")

    if ok:
        print("\nNao reproduzi a divergencia desta vez -- algo não determinístico "
              "entre execuções deste próprio script. Rode de novo para confirmar.")
        return

    print("\n[C] Investigando BATCH A BATCH dentro da epoca 1 (passos 5 a 8)")
    print("    Para isso, refazemos os dois caminhos do zero, mas processando")
    print("    a epoca 1 manualmente, passo a passo, comparando apos cada lote.")

    def preparar_continuo_ate_fim_da_epoca0():
        m = novo_modelo()
        t = Treinador(m, cfg(tmp / "c2", 4))
        t.treinar(criar_loader_treino=loader, val_loader=None)
        return m, t

    def preparar_retomado_apos_checkpoint():
        m0 = novo_modelo()
        t0 = Treinador(m0, cfg(tmp / "i2", 4))
        r = t0.treinar(criar_loader_treino=loader, val_loader=None)
        m1 = novo_modelo()
        t1 = Treinador(m1, cfg(tmp / "r2", 8))
        t1.retomar_de_checkpoint(r.ultimo_checkpoint)
        return m1, t1

    m_a, t_a = preparar_continuo_ate_fim_da_epoca0()
    m_b, t_b = preparar_retomado_apos_checkpoint()

    ok0, onde0 = pesos_iguais(m_a, m_b)
    print(f"  antes da epoca 1 (logo apos preparar): identicos={ok0} {onde0}")

    batches_a = list(loader(t_a.epoca_atual))
    batches_b = list(loader(t_b.epoca_atual))
    print(f"  epoca usada por A: {t_a.epoca_atual}  epoca usada por B: {t_b.epoca_atual}")
    for i, ((ia, ta_), (ib, tb_)) in enumerate(zip(batches_a, batches_b)):
        print(f"  lote {i}: input_ids identicos entre A e B?", torch.equal(ia, ib))

    for i in range(len(batches_a)):
        input_ids_a, target_ids_a = batches_a[i]
        input_ids_b, target_ids_b = batches_b[i]

        loss_a, _ = t_a._passo_de_forward_backward(input_ids_a, target_ids_a)
        gn_a = torch.nn.utils.clip_grad_norm_(t_a.model.parameters(), t_a.config.grad_clip)
        t_a._definir_lr(t_a._lr_atual())
        t_a.optimizer.step()
        t_a.optimizer.zero_grad(set_to_none=True)
        t_a.step += 1

        loss_b, _ = t_b._passo_de_forward_backward(input_ids_b, target_ids_b)
        gn_b = torch.nn.utils.clip_grad_norm_(t_b.model.parameters(), t_b.config.grad_clip)
        t_b._definir_lr(t_b._lr_atual())
        t_b.optimizer.step()
        t_b.optimizer.zero_grad(set_to_none=True)
        t_b.step += 1

        pesos_ok, onde_diverge = pesos_iguais(m_a, m_b)
        print(
            f"  apos lote {i}: loss_a={loss_a.item():.6f} loss_b={loss_b.item():.6f} "
            f"gradnorm_a={float(gn_a):.6f} gradnorm_b={float(gn_b):.6f} "
            f"pesos_identicos={pesos_ok} {onde_diverge}"
        )
        if not pesos_ok:
            print("  >>> DIVERGENCIA ENCONTRADA NESTE LOTE -- parando a investigacao aqui.")
            break


if __name__ == "__main__":
    main()
