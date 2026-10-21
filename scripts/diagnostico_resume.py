#!/usr/bin/env python3
"""Diagnostico PONTUAL e TEMPORARIO para isolar onde diverge a equivalencia
treino-continuo vs interrompido/retomado (ver docs/limitations.md,
test_equivalencia_treinamento_continuo_versus_interrompido_e_retomado).

NAO faz parte da suite de testes nem do pipeline oficial -- e so uma
ferramenta de investigacao para rodar uma vez no Windows nativo e reportar
a saida. Pode ser apagado depois que o problema for entendido/corrigido.

Uso:
    python scripts/diagnostico_resume.py
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


def pesos_iguais(a, b, nome_a="A", nome_b="B") -> bool:
    tudo_igual = True
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        igual = torch.equal(pa, pb)
        if not igual:
            diff = (pa - pb).abs().max().item()
            print(f"    DIVERGE: {na}  (diff maximo = {diff:.3e})")
            tudo_igual = False
    return tudo_igual


def comparar_estado_otimizador(a: dict, b: dict) -> bool:
    sa, sb = a["state"], b["state"]
    if set(sa.keys()) != set(sb.keys()):
        print("    chaves de estado do otimizador diferem:", set(sa.keys()), "vs", set(sb.keys()))
        return False
    ok = True
    for k in sa:
        for campo in ("step", "exp_avg", "exp_avg_sq"):
            va = sa[k].get(campo)
            vb = sb[k].get(campo)
            if va is None or vb is None:
                continue
            ta = torch.as_tensor(va)
            tb = torch.as_tensor(vb)
            if ta.shape != tb.shape or not torch.equal(ta, tb):
                print(f"    DIVERGE optimizer[param={k}][{campo}]: a={ta} b={tb}")
                ok = False
    return ok


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ignis_diag_resume_"))
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

    print("\n[passo 1] Duas corridas FRESCAS e INDEPENDENTES ate o step 4 (sem checkpoint envolvido)")
    m_cont = novo_modelo()
    t_cont = Treinador(m_cont, cfg(tmp / "cont", 4))
    t_cont.treinar(criar_loader_treino=loader, val_loader=None)

    m_int = novo_modelo()
    t_int = Treinador(m_int, cfg(tmp / "int", 4))
    resultado = t_int.treinar(criar_loader_treino=loader, val_loader=None)
    ckpt = resultado.ultimo_checkpoint
    assert ckpt is not None

    ok_passo1 = pesos_iguais(m_cont, m_int)
    print("  resultado:", "OK (identicos)" if ok_passo1 else "DIVERGIU")

    print("\n[passo 2] Carregar o checkpoint do step 4 num modelo/otimizador NOVO, sem treinar mais nada")
    m_resumed = novo_modelo()
    t_resumed = Treinador(m_resumed, cfg(tmp / "resumed", 8))
    step_retomado = t_resumed.retomar_de_checkpoint(ckpt)
    print(f"  step retomado: {step_retomado}  epoca_atual: {t_resumed.epoca_atual}")

    ok_passo2_pesos = pesos_iguais(m_cont, m_resumed)
    print("  pesos identicos apos load (sem treinar mais):", "OK" if ok_passo2_pesos else "DIVERGIU")

    ok_passo2_otim = comparar_estado_otimizador(t_cont.optimizer.state_dict(), t_resumed.optimizer.state_dict())
    print("  estado do otimizador identico apos load:", "OK" if ok_passo2_otim else "DIVERGIU")

    print("\n[passo 3] Continuar AMBOS por exatamente mais 1 passo (epoca 1, primeiro lote) e comparar")
    t_cont.config.max_steps = 5
    t_cont.treinar(criar_loader_treino=loader, val_loader=None)

    t_resumed.config.max_steps = 5
    t_resumed.treinar(criar_loader_treino=loader, val_loader=None)

    ok_passo3 = pesos_iguais(m_cont, m_resumed)
    print("  resultado apos +1 passo:", "OK (identicos)" if ok_passo3 else "DIVERGIU")

    print("\n[passo 4] Continuar AMBOS ate o step 8 (replicando o teste original) e comparar")
    t_cont.config.max_steps = 8
    t_cont.treinar(criar_loader_treino=loader, val_loader=None)

    t_resumed.config.max_steps = 8
    t_resumed.treinar(criar_loader_treino=loader, val_loader=None)

    ok_passo4 = pesos_iguais(m_cont, m_resumed)
    print("  resultado apos step 8 completo:", "OK (identicos)" if ok_passo4 else "DIVERGIU")

    print("\n=== RESUMO ===")
    print("passo 1 (duas corridas frescas, sem checkpoint):", "OK" if ok_passo1 else "DIVERGIU")
    print("passo 2a (pesos apos load, sem treinar):        ", "OK" if ok_passo2_pesos else "DIVERGIU")
    print("passo 2b (estado do otimizador apos load):      ", "OK" if ok_passo2_otim else "DIVERGIU")
    print("passo 3 (apos +1 passo pos-resume):              ", "OK" if ok_passo3 else "DIVERGIU")
    print("passo 4 (apos +4 passos pos-resume, step 8):     ", "OK" if ok_passo4 else "DIVERGIU")


if __name__ == "__main__":
    main()
