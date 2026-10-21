"""Verificação mecânica de que a GPU CUDA deste ambiente (tipicamente
`.venv-cuda`, criado por `scripts/preparar_venv_cuda.ps1`) funciona de
verdade para o Ignis-30M: `torch.cuda.is_available()`, alocação de tensores
na GPU, um forward/backward real (usando o próprio `IgnisModel`, se
disponível, ou um modelo mínimo de fallback), e uma comparação de
desempenho CPU-vs-GPU em uma carga de trabalho modesta.

Este script NÃO instala nem modifica nada — é somente leitura/diagnóstico,
chamado depois que torch com suporte a CUDA já foi instalado (Sprint 5,
seção 1). Pode ser rodado isoladamente:

    python scripts/verificar_gpu_cuda.py [--saida relatorio.json]

Código de saída:
    0 -> GPU CUDA disponível e todos os testes passaram
    1 -> torch.cuda.is_available() é False (sem GPU CUDA utilizável)
    2 -> GPU detectada, mas algum teste falhou (erro real, não apenas ausência de GPU)
    3 -> torch não está instalado neste interpretador
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _secao(titulo: str) -> None:
    print(f"\n==== {titulo} ====")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", default=None, help="Salva o relatório em um arquivo JSON")
    parser.add_argument(
        "--tamanho-matmul", type=int, default=2048, help="Dimensão da matriz quadrada usada na comparação de desempenho"
    )
    parser.add_argument(
        "--repeticoes-matmul", type=int, default=20, help="Quantas multiplicações de matriz medir em cada device"
    )
    args = parser.parse_args(argv)

    relatorio: dict = {"testes": [], "cuda_disponivel": False, "sucesso": False}

    def _registrar(nome: str, ok: bool, detalhe: str) -> None:
        relatorio["testes"].append({"nome": nome, "ok": bool(ok), "detalhe": detalhe})
        marca = "OK" if ok else "FALHOU"
        print(f"  [{marca}] {nome}: {detalhe}")

    try:
        import torch
    except ImportError as exc:
        print(f"ERRO: torch não está instalado neste interpretador: {exc}", file=sys.stderr)
        return 3

    _secao("1. Disponibilidade de CUDA")
    cuda_disponivel = torch.cuda.is_available()
    relatorio["cuda_disponivel"] = cuda_disponivel
    relatorio["torch_version"] = torch.__version__
    relatorio["torch_cuda_version"] = torch.version.cuda
    print(f"  torch.__version__ = {torch.__version__}")
    print(f"  torch.version.cuda (versão com que o torch foi compilado) = {torch.version.cuda}")
    print(f"  torch.cuda.is_available() = {cuda_disponivel}")

    if not cuda_disponivel:
        print(
            "\nSem GPU CUDA utilizável neste interpretador -- isto não é necessariamente um erro "
            "(pode ser intencional, ex.: rodando este script no .venv CPU por engano, ou "
            "torch.cuda.is_available() ainda False por falta de instalação CUDA). Nenhum teste de "
            "GPU foi executado.",
            file=sys.stderr,
        )
        _salvar_relatorio(relatorio, args.saida)
        return 1

    nome_gpu = torch.cuda.get_device_name(0)
    capacidade = torch.cuda.get_device_capability(0)
    relatorio["gpu"] = {"nome": nome_gpu, "compute_capability": f"{capacidade[0]}.{capacidade[1]}"}
    print(f"  GPU: {nome_gpu}  (compute capability {capacidade[0]}.{capacidade[1]})")

    _secao("2. Alocação de tensor na GPU")
    try:
        t = torch.randn(1024, 1024, device="cuda")
        soma = t.sum().item()
        _registrar("alocacao_tensor_gpu", True, f"tensor 1024x1024 alocado e reduzido na GPU (soma={soma:.4f})")
    except Exception as exc:
        _registrar("alocacao_tensor_gpu", False, str(exc))
        _salvar_relatorio(relatorio, args.saida)
        return 2

    _secao("3. Forward/backward real na GPU")
    try:
        try:
            # Tenta usar o próprio IgnisModel (mais representativo do uso
            # real do projeto); cai para um modelo mínimo de fallback se
            # o pacote ignis não estiver instalado neste venv.
            from ignis.config import IgnisModelConfig
            from ignis.model import IgnisModel

            config_minima = IgnisModelConfig(
                vocab_size=64, max_seq_len=32, n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, head_dim=8, intermediate_size=64
            )
            modelo = IgnisModel(config_minima).to("cuda")
            entrada = torch.randint(0, config_minima.vocab_size, (2, 16), device="cuda")
            logits = modelo(entrada)
            loss = logits.float().pow(2).mean()
            loss.backward()
            finito = bool(torch.isfinite(loss))
            _registrar(
                "forward_backward_ignis_model_gpu",
                finito,
                f"IgnisModel (config mínima) forward+backward na GPU; loss={loss.item():.6f}  finito={finito}",
            )
        except ImportError:
            # Fallback: ignis não instalado neste venv -- ainda assim
            # confirma que forward/backward real funciona na GPU.
            camada = torch.nn.Linear(64, 64).to("cuda")
            entrada = torch.randn(8, 64, device="cuda")
            saida = camada(entrada)
            loss = saida.pow(2).mean()
            loss.backward()
            finito = bool(torch.isfinite(loss))
            _registrar(
                "forward_backward_fallback_linear_gpu",
                finito,
                f"pacote ignis não encontrado neste venv; usado nn.Linear de fallback. loss={loss.item():.6f}  finito={finito}",
            )
    except Exception as exc:
        _registrar("forward_backward_gpu", False, str(exc))
        _salvar_relatorio(relatorio, args.saida)
        return 2

    _secao("4. Comparação de desempenho CPU vs GPU (multiplicação de matrizes)")
    try:
        n = args.tamanho_matmul
        reps = args.repeticoes_matmul

        a_cpu = torch.randn(n, n)
        b_cpu = torch.randn(n, n)
        inicio_cpu = time.perf_counter()
        for _ in range(reps):
            _ = a_cpu @ b_cpu
        tempo_cpu = time.perf_counter() - inicio_cpu

        a_gpu = a_cpu.to("cuda")
        b_gpu = b_cpu.to("cuda")
        torch.cuda.synchronize()
        inicio_gpu = time.perf_counter()
        for _ in range(reps):
            _ = a_gpu @ b_gpu
        torch.cuda.synchronize()
        tempo_gpu = time.perf_counter() - inicio_gpu

        aceleracao = tempo_cpu / tempo_gpu if tempo_gpu > 0 else float("inf")
        relatorio["benchmark_matmul"] = {
            "tamanho": n,
            "repeticoes": reps,
            "tempo_cpu_s": tempo_cpu,
            "tempo_gpu_s": tempo_gpu,
            "aceleracao_gpu_sobre_cpu": aceleracao,
        }
        _registrar(
            "benchmark_matmul_cpu_vs_gpu",
            True,
            f"matmul {n}x{n} x{reps}: CPU={tempo_cpu:.4f}s  GPU={tempo_gpu:.4f}s  aceleracao={aceleracao:.1f}x "
            "(nao e um benchmark de treinamento real, apenas um sinal rapido de que a GPU esta sendo usada de fato)",
        )
    except Exception as exc:
        _registrar("benchmark_matmul_cpu_vs_gpu", False, str(exc))
        # Falha no benchmark de desempenho não invalida os testes funcionais
        # anteriores (alocação e forward/backward já confirmados) -- não
        # retorna erro fatal aqui, só registra.

    relatorio["sucesso"] = all(t["ok"] for t in relatorio["testes"])
    _salvar_relatorio(relatorio, args.saida)

    print()
    if relatorio["sucesso"]:
        print("GPU CUDA confirmada como utilizável: alocação, forward/backward e benchmark executados com sucesso.")
        return 0
    else:
        print("ERRO: pelo menos um teste de GPU falhou -- ver detalhes acima.", file=sys.stderr)
        return 2


def _salvar_relatorio(relatorio: dict, caminho: str | None) -> None:
    if caminho:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)
        print(f"\nRelatório salvo em: {caminho}")


if __name__ == "__main__":
    raise SystemExit(main())
