"""Benchmark de hardware e desempenho de treinamento do Ignis-30M.

Mede forward, backward e passo de otimizador SEPARADAMENTE, em sequências
curtas (32, 128, 256 tokens por padrão) — NUNCA extrapola esses números
linearmente para o contexto completo de 1.024 tokens: o custo de atenção
cresce de forma superlinear com o comprimento da sequência, então medir em
32 tokens e multiplicar por 32 para estimar 1.024 tokens SUBESTIMARIA o
custo real. Quando possível, meça também em 1.024 diretamente antes de
confiar em qualquer estimativa de tempo de piloto.

Acumulação de gradientes (--grad-accum-steps, Sprint 4): quando
`grad_accum_steps > 1`, este script DISTINGUE explicitamente dois
throughputs diferentes, que NUNCA devem ser confundidos entre si:

  - "throughput por microbatch": tokens/s de um único forward+backward
    (ex.: microbatch_size=1, seq_len=128), sem contar o custo do
    optimizer.step() (que só acontece uma vez a cada `grad_accum_steps`
    microbatches). É a métrica relevante para "quanto tempo leva processar
    UM lote pequeno que cabe na memória".
  - "throughput por passo de otimizador": tokens/s de um PASSO DE
    OTIMIZADOR completo (grad_accum_steps microbatches de forward+backward
    + um único optimizer.step()), que é a unidade real de progresso do
    treinamento (um "step" em `ignis.training.trainer.Treinador`, com o
    lote EFETIVO = microbatch_size * grad_accum_steps). É esta métrica, não
    a de microbatch, que deve ser usada para estimar quanto tempo um piloto
    real (configs/train-pilot.yaml) vai levar.

Mixed precision e memória de GPU (Sprint 6): `--mixed-precision {no,bf16,fp16}`
usa exatamente a mesma lógica de `ignis.training.trainer.Treinador`
(`torch.autocast` + `torch.cuda.amp.GradScaler`, reaproveitados por
importação direta, nunca reimplementados aqui) — "fp16" exige
`--device cuda` (autocast fp16 não é suportado em CPU), mesma validação do
`Treinador`. Em `--device cuda`, cada medição de seq_len também reporta
VRAM alocada/reservada e o PICO de VRAM alocada/reservada durante a
medição (`torch.cuda.reset_peak_memory_stats` antes, `torch.cuda.memory_allocated`/
`memory_reserved`/`max_memory_allocated`/`max_memory_reserved` depois).

Falta de memória (OOM) durante uma medição NUNCA derruba o script inteiro:
a medição daquele seq_len/config específico é marcada como `"oom": true`
no relatório (com a mensagem de erro), a VRAM é liberada
(`torch.cuda.empty_cache()`) e o script continua para a próxima
combinação segura — nunca tenta adivinhar um tamanho menor sozinho.

Este script NÃO treina o modelo de verdade: usa dados sintéticos e nunca
persiste nenhum resultado em disco além do relatório JSON. Requer torch —
não executável na VM Linux de desenvolvimento (ver
docs/experiments/fase1-fundacao.md); rode no Windows nativo:

    python scripts/benchmark_training.py --saida benchmark_resultado.json
    python scripts/benchmark_training.py --grad-accum-steps 32 --seq-lens 128 --saida benchmark_piloto.json
    python scripts/benchmark_training.py --device cuda --mixed-precision fp16 --seq-lens 32 128 256 --saida benchmark_gpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch

from ignis.config import IgnisConfig, IgnisModelConfig
from ignis.model import IgnisModel
from ignis.training.trainer import DTYPES_MIXED_PRECISION, _e_erro_de_memoria

try:
    import psutil

    _PSUTIL_DISPONIVEL = True
except ImportError:
    _PSUTIL_DISPONIVEL = False


@dataclass
class InfoHardware:
    sistema_operacional: str
    processador: str
    n_cpus_logicos: int | None
    ram_total_gb: float | None
    torch_versao: str
    cuda_disponivel: bool
    cuda_versao: str | None
    nome_gpu: str | None


def coletar_info_hardware() -> InfoHardware:
    """Nunca assume GPU pelo modelo do computador — só reporta o que o
    próprio PyTorch/psutil efetivamente detectam neste processo."""
    n_cpus = psutil.cpu_count(logical=True) if _PSUTIL_DISPONIVEL else None
    ram_total_gb = (psutil.virtual_memory().total / (1024**3)) if _PSUTIL_DISPONIVEL else None

    cuda_disponivel = torch.cuda.is_available()
    cuda_versao = torch.version.cuda if cuda_disponivel else None
    nome_gpu = torch.cuda.get_device_name(0) if cuda_disponivel else None

    return InfoHardware(
        sistema_operacional=f"{platform.system()} {platform.release()}",
        processador=platform.processor() or platform.machine(),
        n_cpus_logicos=n_cpus,
        ram_total_gb=ram_total_gb,
        torch_versao=torch.__version__,
        cuda_disponivel=cuda_disponivel,
        cuda_versao=cuda_versao,
        nome_gpu=nome_gpu,
    )


def _ram_mb() -> float | None:
    if not _PSUTIL_DISPONIVEL:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _sincronizar(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def medir_sequencia(
    model: IgnisModel,
    seq_len: int,
    batch_size: int,
    device: str,
    repeticoes: int,
    aquecimento: int,
    grad_accum_steps: int = 1,
    mixed_precision: str = "no",
) -> dict:
    """Mede um PASSO DE OTIMIZADOR completo (grad_accum_steps microbatches de
    forward+backward + um optimizer.step()), repetido `repeticoes` vezes
    após `aquecimento` repetições de descarte. Retorna tanto o throughput
    por microbatch quanto o throughput por passo de otimizador completo —
    ver docstring do módulo para a distinção.

    `mixed_precision` ("no"/"bf16"/"fp16") usa a MESMA lógica de
    `ignis.training.trainer.Treinador` (autocast + GradScaler, nunca
    reimplementada aqui). Se ocorrer uma falha de memória (OOM) em
    qualquer ponto, a exceção NÃO propaga: esta função retorna um dict com
    `"oom": True` em vez de lançar, para que o chamador possa pular esta
    combinação específica e continuar com as demais."""
    vocab_size = model.config.vocab_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision == "fp16"))

    def _autocast_ctx():
        if mixed_precision == "no":
            return nullcontext()
        return torch.autocast(device_type=device, dtype=DTYPES_MIXED_PRECISION[mixed_precision])

    def _microbatch_forward_backward() -> tuple[float, float]:
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        _sincronizar(device)
        t0 = time.perf_counter()
        with _autocast_ctx():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        _sincronizar(device)
        t_forward = time.perf_counter() - t0

        t0 = time.perf_counter()
        scaler.scale(loss / grad_accum_steps).backward()
        _sincronizar(device)
        t_backward = time.perf_counter() - t0

        return t_forward, t_backward

    def _passo_otimizador_completo() -> tuple[float, float, float]:
        optimizer.zero_grad(set_to_none=True)
        t_forward_total = 0.0
        t_backward_total = 0.0
        for _ in range(grad_accum_steps):
            tf, tb = _microbatch_forward_backward()
            t_forward_total += tf
            t_backward_total += tb

        t0 = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        _sincronizar(device)
        t_optimizer = time.perf_counter() - t0

        return t_forward_total, t_backward_total, t_optimizer

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        for _ in range(aquecimento):
            _passo_otimizador_completo()

        tempos_forward, tempos_backward, tempos_optimizer = [], [], []
        ram_antes = _ram_mb()
        for _ in range(repeticoes):
            tf, tb, to = _passo_otimizador_completo()
            tempos_forward.append(tf)
            tempos_backward.append(tb)
            tempos_optimizer.append(to)
        ram_depois = _ram_mb()
    except RuntimeError as exc:
        if not _e_erro_de_memoria(exc):
            raise
        if device == "cuda":
            torch.cuda.empty_cache()
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "lote_efetivo": batch_size * grad_accum_steps,
            "mixed_precision": mixed_precision,
            "oom": True,
            "erro": str(exc),
        }

    tempo_medio_forward_total_passo = sum(tempos_forward) / repeticoes
    tempo_medio_backward_total_passo = sum(tempos_backward) / repeticoes
    tempo_medio_optimizer = sum(tempos_optimizer) / repeticoes
    tempo_medio_passo_completo = tempo_medio_forward_total_passo + tempo_medio_backward_total_passo + tempo_medio_optimizer
    tempo_medio_fwd_bwd_por_microbatch = (tempo_medio_forward_total_passo + tempo_medio_backward_total_passo) / grad_accum_steps

    n_tokens_por_microbatch = batch_size * seq_len
    n_tokens_por_passo_otimizador = n_tokens_por_microbatch * grad_accum_steps

    tokens_por_segundo_microbatch = (
        n_tokens_por_microbatch / tempo_medio_fwd_bwd_por_microbatch if tempo_medio_fwd_bwd_por_microbatch > 0 else None
    )
    tokens_por_segundo_passo_otimizador = (
        n_tokens_por_passo_otimizador / tempo_medio_passo_completo if tempo_medio_passo_completo > 0 else None
    )

    vram = None
    if device == "cuda":
        vram = {
            "alocada_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "reservada_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "pico_alocada_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "pico_reservada_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
        }

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lote_efetivo": batch_size * grad_accum_steps,
        "mixed_precision": mixed_precision,
        "oom": False,
        "repeticoes": repeticoes,
        "aquecimento": aquecimento,
        # Campos mantidos por compatibilidade com relatorios anteriores ao
        # Sprint 4: tempo POR MICROBATCH de forward/backward (coincide com o
        # significado original quando grad_accum_steps=1), e tempo do
        # optimizer.step() (sempre medido uma vez por passo completo).
        "tempo_medio_forward_s": tempo_medio_forward_total_passo / grad_accum_steps,
        "tempo_medio_backward_s": tempo_medio_backward_total_passo / grad_accum_steps,
        "tempo_medio_optimizer_step_s": tempo_medio_optimizer,
        "tokens_por_segundo": tokens_por_segundo_passo_otimizador,
        # Distinção explícita exigida pelo Sprint 4 (ver docstring do módulo).
        "throughput_microbatch": {
            "descricao": "Forward+backward de UM microbatch, sem o custo do optimizer.step() (amortizado entre os microbatches do passo).",
            "tokens_por_microbatch": n_tokens_por_microbatch,
            "tempo_medio_s": tempo_medio_fwd_bwd_por_microbatch,
            "tokens_por_segundo": tokens_por_segundo_microbatch,
        },
        "throughput_passo_otimizador": {
            "descricao": "Um PASSO DE OTIMIZADOR completo: grad_accum_steps microbatches de forward+backward + um optimizer.step() (lote efetivo = batch_size * grad_accum_steps).",
            "tokens_por_passo_otimizador": n_tokens_por_passo_otimizador,
            "tempo_medio_s": tempo_medio_passo_completo,
            "tokens_por_segundo": tokens_por_segundo_passo_otimizador,
        },
        "ram_mb_antes": ram_antes,
        "ram_mb_depois": ram_depois,
        # None em CPU -- VRAM so existe em device="cuda" (Sprint 6).
        "vram": vram,
    }


def estimar_tempo_piloto(resultados_sequencia: list[dict], n_steps_piloto: int, seq_len_piloto: int) -> dict | None:
    """Estima o tempo do piloto A PARTIR DE UMA MEDIÇÃO REAL no MESMO
    `seq_len_piloto` (e no mesmo `grad_accum_steps`, para medir o PASSO DE
    OTIMIZADOR completo, não apenas um microbatch) — nunca extrapola de uma
    sequência curta para outra mais longa. Se nenhuma medição com
    `seq_len_piloto` exata estiver disponível, retorna None em vez de
    inventar uma extrapolação."""
    candidatos = [r for r in resultados_sequencia if r["seq_len"] == seq_len_piloto]
    if not candidatos:
        return None
    r = candidatos[0]
    tempo_por_passo = r["throughput_passo_otimizador"]["tempo_medio_s"]
    return {
        "baseado_em_medicao_com_seq_len": seq_len_piloto,
        "baseado_em_grad_accum_steps": r["grad_accum_steps"],
        "tempo_por_passo_otimizador_s": tempo_por_passo,
        "n_steps_piloto": n_steps_piloto,
        "tempo_total_estimado_s": tempo_por_passo * n_steps_piloto,
        "tempo_total_estimado_min": tempo_por_passo * n_steps_piloto / 60,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="ignis-30m", choices=["ignis-30m"], help="Configuração do modelo")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Microbatches acumulados por passo de otimizador (ver configs/train-pilot.yaml: lote.grad_accum_steps)",
    )
    parser.add_argument("--repeticoes", type=int, default=5)
    parser.add_argument("--aquecimento", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--mixed-precision",
        default="no",
        choices=["no", "bf16", "fp16"],
        help=(
            "Mesma semantica de TreinadorConfig.mixed_precision (Sprint 5): 'fp16' exige "
            "--device cuda (autocast fp16 nao suportado em CPU) -- validado antes de qualquer medicao."
        ),
    )
    parser.add_argument("--n-steps-piloto", type=int, default=None, help="Se informado, estima tempo do piloto")
    parser.add_argument("--seq-len-piloto", type=int, default=128)
    parser.add_argument("--saida", default="benchmark_resultado.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("AVISO: --device cuda pedido, mas torch.cuda.is_available() é False; usando cpu.")
        args.device = "cpu"

    if args.mixed_precision == "fp16" and args.device != "cuda":
        print("ERRO: --mixed-precision fp16 exige --device cuda (autocast fp16 não é suportado em CPU).")
        return 2

    info_hardware = coletar_info_hardware()
    print("=== Hardware detectado ===")
    for chave, valor in asdict(info_hardware).items():
        print(f"  {chave}: {valor}")
    if not info_hardware.cuda_disponivel:
        print("  (nenhum acelerador CUDA detectado — todas as medições serão em CPU)")

    config = IgnisConfig.ignis_30m()
    model = IgnisModel(config.modelo).to(args.device)
    n_params = model.contar_parametros()
    print(f"\n=== Modelo: {config.modelo.nome} ({n_params:,} parâmetros) ===")
    print(f"=== grad_accum_steps={args.grad_accum_steps}  lote_efetivo={args.batch_size * args.grad_accum_steps} ===")

    resultados_sequencia = []
    for seq_len in args.seq_lens:
        if seq_len > config.modelo.max_seq_len:
            print(f"AVISO: seq_len={seq_len} > max_seq_len={config.modelo.max_seq_len}; pulando.")
            continue
        print(f"\nMedindo seq_len={seq_len}, batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}...")
        resultado = medir_sequencia(
            model,
            seq_len,
            args.batch_size,
            args.device,
            args.repeticoes,
            args.aquecimento,
            args.grad_accum_steps,
            args.mixed_precision,
        )
        resultados_sequencia.append(resultado)
        if resultado.get("oom"):
            print(f"  OOM: falta de memoria nesta combinacao -- pulando. Detalhe: {resultado['erro']}")
            continue
        tp_micro = resultado["throughput_microbatch"]["tokens_por_segundo"]
        tp_passo = resultado["throughput_passo_otimizador"]["tokens_por_segundo"]
        print(
            f"  forward/microbatch={resultado['tempo_medio_forward_s']:.4f}s  "
            f"backward/microbatch={resultado['tempo_medio_backward_s']:.4f}s  "
            f"optimizer_step={resultado['tempo_medio_optimizer_step_s']:.4f}s"
        )
        print(
            f"  tokens/s (microbatch, SEM optimizer.step)={tp_micro:.1f}   "
            f"tokens/s (passo de otimizador completo, lote_efetivo={resultado['lote_efetivo']})={tp_passo:.1f}"
        )
        if resultado.get("vram"):
            v = resultado["vram"]
            print(
                f"  VRAM alocada={v['alocada_mb']:.1f}MB (pico={v['pico_alocada_mb']:.1f}MB)  "
                f"reservada={v['reservada_mb']:.1f}MB (pico={v['pico_reservada_mb']:.1f}MB)"
            )

    estimativa_piloto = None
    if args.n_steps_piloto is not None:
        resultados_sem_oom = [r for r in resultados_sequencia if not r.get("oom")]
        estimativa_piloto = estimar_tempo_piloto(resultados_sem_oom, args.n_steps_piloto, args.seq_len_piloto)
        if estimativa_piloto is None:
            print(
                f"\nAVISO: não foi possível estimar o tempo do piloto — nenhuma medição com "
                f"seq_len={args.seq_len_piloto} (inclua esse valor em --seq-lens)."
            )
        else:
            print(
                f"\n=== Estimativa do piloto (baseada em medição real com seq_len="
                f"{args.seq_len_piloto}, grad_accum_steps={args.grad_accum_steps}) ==="
            )
            print(f"  {args.n_steps_piloto} passos de otimizador -> ~{estimativa_piloto['tempo_total_estimado_min']:.1f} min")

    relatorio = {
        "hardware": asdict(info_hardware),
        "modelo": {"nome": config.modelo.nome, "n_parametros": n_params},
        "device_usado": args.device,
        "mixed_precision": args.mixed_precision,
        "grad_accum_steps": args.grad_accum_steps,
        "resultados_por_seq_len": resultados_sequencia,
        "estimativa_piloto": estimativa_piloto,
        "aviso_extrapolacao": (
            "Os tempos medidos em sequências curtas NÃO devem ser extrapolados "
            "linearmente para sequências mais longas (ex.: max_seq_len=1024) — "
            "o custo de atenção cresce de forma superlinear com o comprimento. "
            "Meça diretamente no comprimento de interesse antes de confiar em "
            "qualquer estimativa de tempo de treinamento."
        ),
        "aviso_throughput_microbatch_vs_passo_otimizador": (
            "'throughput_microbatch' (tokens/s de um unico forward+backward) e "
            "'throughput_passo_otimizador' (tokens/s de um passo de otimizador "
            "completo, com lote_efetivo = batch_size * grad_accum_steps) sao "
            "metricas DIFERENTES e NUNCA devem ser confundidas entre si: a "
            "primeira mede a velocidade mecanica de processar um lote pequeno, "
            "a segunda mede o progresso real de treinamento (um 'step' do "
            "Treinador). Use sempre throughput_passo_otimizador para estimar "
            "tempo de um piloto real."
        ),
    }

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório salvo em: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
