"""Loop de treinamento do Ignis (Fase 3 — Pré-treinamento experimental).

Este módulo é o pipeline COMPLETO de pré-treinamento: scheduler de LR com
warmup, acumulação de gradientes, clipping, detecção de NaN/Inf, validação
periódica e checkpoint. `ignis.training.loop.train_steps` continua existindo
separadamente e é intencionalmente mais simples — usado apenas em smoke
tests e no teste de overfit controlado (Fase 1), não neste pipeline.

Princípios respeitados desta skill/projeto:
- O split de TESTE nunca é usado aqui. `Treinador.treinar` só recebe
  carregadores de treino e validação; qualquer avaliação final em teste é
  responsabilidade de um script separado, executado uma única vez, depois
  que todas as decisões de hiperparâmetros já foram tomadas.
- `max_steps` é obrigatório — nunca treina indefinidamente.
- `max_steps` é o HORIZONTE TOTAL do experimento e é o único valor usado
  pelo scheduler de LR (`lr_no_passo`). Ele NUNCA deve mudar entre uma
  execução contínua e uma interrompida/retomada que devam ser equivalentes:
  mudar `max_steps` muda a curva de cosseno do LR para TODOS os passos,
  inclusive os anteriores ao ponto de interrupção. Para parar uma execução
  antes do horizonte planejado (ex.: um teste parcial), use o parâmetro
  `parar_no_passo` de `Treinador.treinar`, que afeta apenas a condição de
  parada do laço, nunca o cálculo de LR.
- Uma falha numérica (loss ou gradiente não finito) interrompe o
  treinamento explicitamente; nunca é ignorada silenciosamente.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from ..seeding import set_seed
from .checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from .dataloader import IGNORE_INDEX
from .metrics import RegistradorMetricas

CarregadorFactory = Callable[[int], Iterable[tuple[torch.Tensor, torch.Tensor]]]

DTYPES_MIXED_PRECISION = {"bf16": torch.bfloat16, "fp16": torch.float16}
"""Mapeia os valores aceitos de `TreinadorConfig.mixed_precision` (exceto
"no") para o dtype de `torch.autocast`. "fp16" só é suportado em CUDA (ver
`Treinador._autocast_ctx`); "bf16" funciona em CPU e CUDA."""


class FalhaNumericaError(RuntimeError):
    """Levantada internamente quando a loss ou algum gradiente deixa de ser
    finito (NaN ou Inf). `Treinador.treinar` captura isso e encerra o
    treinamento de forma controlada, registrando o motivo — nunca continua
    treinando com estado numericamente corrompido."""


class FalhaDeMemoriaError(RuntimeError):
    """Levantada internamente quando o forward, o backward ou o passo do
    otimizador esgotam a memória do device (CUDA out-of-memory, ou
    `MemoryError` em CPU). `Treinador.treinar` captura isso de forma
    análoga a `FalhaNumericaError`: encerra o treinamento de forma
    controlada e tenta um checkpoint de emergência — essa tentativa de
    checkpoint pode, ela mesma, falhar se a memória estiver criticamente
    esgotada, o que é registrado explicitamente, nunca escondido."""


def _e_erro_de_memoria(exc: BaseException) -> bool:
    """Reconhece um erro de falta de memória (CUDA OOM ou `MemoryError` de
    CPU) a partir de uma exceção genérica levantada por uma chamada do
    PyTorch. Usa tanto o tipo (`torch.cuda.OutOfMemoryError`, quando
    disponível nesta versão do PyTorch) quanto, como rede de segurança,
    correspondência textual em `RuntimeError` (mensagens de OOM do PyTorch
    historicamente contêm "out of memory", mesmo quando a exceção não é da
    classe dedicada)."""
    if isinstance(exc, MemoryError):
        return True
    classe_oom_cuda = getattr(torch.cuda, "OutOfMemoryError", None)
    if classe_oom_cuda is not None and isinstance(exc, classe_oom_cuda):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@dataclass
class TreinadorConfig:
    # Otimização
    lr: float
    lr_min: float = 0.0
    warmup_steps: int = 100
    max_steps: int = 1000  # obrigatório ter um limite explícito
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # Validação / checkpoint / logging
    eval_every: int = 100
    eval_max_batches: int | None = 20
    checkpoint_every: int = 100
    salvar_melhor: bool = False
    log_every: int = 10

    # Early stopping explícito (opcional): nenhuma melhora na loss de
    # validação por `paciencia_avaliacoes` avaliações consecutivas para.
    # None desativa o early stopping por paciência (apenas max_steps e
    # falha numérica continuam valendo).
    paciencia_avaliacoes: int | None = None

    # Precisão mista (Sprint 5): opt-in, desativada por padrão ("no"). "bf16"
    # funciona em CPU e CUDA; "fp16" exige device="cuda" (autocast de CPU não
    # suporta float16) -- validado em `Treinador.__init__`, nunca falha
    # silenciosamente no meio do treinamento. Com "fp16", um `GradScaler` é
    # usado internamente para evitar underflow de gradiente; com "bf16" ou
    # "no", o scaler fica desabilitado (passa os valores adiante sem
    # alteração, conforme a API padrão do PyTorch).
    mixed_precision: str = "no"  # "no" | "bf16" | "fp16"

    # Execução
    seed: int = 42
    device: str = "cpu"
    out_dir: str = "checkpoints/pilot"
    parar_em_falha_numerica: bool = True
    # Análogo a `parar_em_falha_numerica`, mas para esgotamento de memória do
    # device (CUDA OOM ou MemoryError de CPU) -- ver `FalhaDeMemoriaError`.
    # Independente do flag de falha numérica: cada um controla sua própria
    # condição de parada.
    parar_em_falha_de_memoria: bool = True


def lr_no_passo(passo: int, config: TreinadorConfig) -> float:
    """Warmup linear até `warmup_steps`, depois decaimento em cosseno até
    `lr_min` em `max_steps`. Hipótese experimental, não um valor validado
    empiricamente — ver docs/plano-corpus-piloto.md e configs/train-pilot.yaml."""
    if config.warmup_steps > 0 and passo < config.warmup_steps:
        return config.lr * (passo + 1) / config.warmup_steps
    progresso = (passo - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    progresso = min(max(progresso, 0.0), 1.0)
    fator_cosseno = 0.5 * (1.0 + math.cos(math.pi * progresso))
    return config.lr_min + (config.lr - config.lr_min) * fator_cosseno


@dataclass
class ResultadoTreinamento:
    passos_executados: int
    tokens_processados: int
    motivo_parada: str
    melhor_loss_validacao: float | None
    tempo_total_s: float
    ultimo_checkpoint: str | None = None


class Treinador:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TreinadorConfig,
        metadata_checkpoint: CheckpointMetadata | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.metadata_checkpoint = metadata_checkpoint

        if config.mixed_precision not in ("no", *DTYPES_MIXED_PRECISION):
            raise ValueError(
                f"mixed_precision inválido: {config.mixed_precision!r} "
                f"(esperado 'no', {', '.join(repr(k) for k in DTYPES_MIXED_PRECISION)})"
            )
        if config.mixed_precision == "fp16" and config.device != "cuda":
            raise ValueError(
                "mixed_precision='fp16' requer device='cuda' -- o autocast de CPU do PyTorch não "
                "suporta float16 (apenas bfloat16). Use mixed_precision='bf16' em CPU."
            )

        set_seed(config.seed)
        model.to(config.device)

        # GradScaler só é efetivamente usado com mixed_precision='fp16'
        # (onde float16 pode sofrer underflow de gradiente); com 'bf16' ou
        # 'no', fica desabilitado e vira um passthrough (scale/step/update
        # não alteram o comportamento), seguindo a API padrão do PyTorch.
        # Construí-lo incondicionalmente, mesmo em CPU/sem mixed precision,
        # evita bifurcar o loop de treinamento em dois caminhos de código.
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.mixed_precision == "fp16"))

        # foreach=False e fixado explicitamente por cautela de reprodutibilidade:
        # sem isso, o PyTorch escolhe automaticamente entre a implementacao
        # "foreach" (vetorizada) e a "for-loop" com base em heuristicas internas
        # que podem, em tese, variar entre um otimizador recem-criado e um cujo
        # estado foi carregado via load_state_dict. Isto NAO foi confirmado como
        # a causa raiz da divergencia observada (ver nota de max_steps acima,
        # que e a causa raiz confirmada) — e uma medida defensiva adicional,
        # nao uma correcao comprovadamente necessaria.
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            foreach=False,
        )

        self.step = 0
        self.tokens_processados = 0
        self.epoca_atual = 0
        self.microbatches_consumidos = 0
        self._n_batches_epoca = None
        self.melhor_loss_validacao: float | None = None
        self.avaliacoes_sem_melhora = 0

        Path(config.out_dir).mkdir(parents=True, exist_ok=True)
        self.registrador = RegistradorMetricas(Path(config.out_dir) / "metricas.jsonl")

    def retomar_de_checkpoint(
        self, caminho: str | Path, verificar_metadata: CheckpointMetadata | None = None
    ) -> int:
        """Carrega pesos, otimizador, RNG e contadores (step, tokens
        processados, melhor loss de validação) de um checkpoint salvo por
        este próprio Treinador. Novos checkpoints registram o cursor após
        optimizer step, a paciência e o scaler. Retorna o `step` retomado.

        Determinismo: em CPU, retomar e continuar deve produzir os mesmos
        resultados que um treinamento contínuo equivalente, DESDE QUE o
        mesmo loader determinístico seja reconstruído por seed/época,
        com num_workers=0 e sem transformações aleatórias. O cursor permite
        saltar os lotes já consumidos. Checkpoints legados e de emergência
        não garantem equivalência; gradientes parciais não são persistidos.
        Isso não é garantido em um eventual backend de GPU com kernels
        não-determinísticos."""
        estado = load_checkpoint(
            caminho, self.model, optimizer=self.optimizer, verificar_metadata=verificar_metadata
        )
        self.step = estado["step"]
        extra = estado.get("extra", {})
        self.tokens_processados = extra.get("tokens_processados", 0)
        self.melhor_loss_validacao = extra.get("melhor_loss_validacao")
        self.epoca_atual = extra.get("epoca_atual", 0)
        self.microbatches_consumidos = extra.get("microbatches_consumidos", 0)
        self.avaliacoes_sem_melhora = extra.get("avaliacoes_sem_melhora", 0)
        if "scaler_state_dict" in extra:
            self.scaler.load_state_dict(extra["scaler_state_dict"])
        if "microbatches_consumidos" not in extra:
            warnings.warn("Checkpoint legado sem posição do iterador: retomada não exata.", RuntimeWarning)
        return self.step

    def _lr_atual(self) -> float:
        return lr_no_passo(self.step, self.config)

    def _definir_lr(self, lr: float) -> None:
        for grupo in self.optimizer.param_groups:
            grupo["lr"] = lr

    def _autocast_ctx(self):
        """Context manager de `torch.autocast` para `self.config.mixed_precision`,
        ou um no-op (`nullcontext`) quando desativado ("no") -- validado
        eagerly em `__init__`, então aqui já é seguro assumir que, se
        mixed_precision != "no", o dtype é suportado pelo device
        configurado (ver `DTYPES_MIXED_PRECISION`)."""
        if self.config.mixed_precision == "no":
            return nullcontext()
        return torch.autocast(
            device_type=self.config.device, dtype=DTYPES_MIXED_PRECISION[self.config.mixed_precision]
        )

    def _passo_de_forward_backward(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        input_ids = input_ids.to(self.config.device)
        target_ids = target_ids.to(self.config.device)

        try:
            with self._autocast_ctx():
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
        except Exception as exc:
            if _e_erro_de_memoria(exc):
                raise FalhaDeMemoriaError(
                    f"memória do device '{self.config.device}' esgotada no passo {self.step} (forward): {exc}"
                ) from exc
            raise

        n_tokens_validos = int((target_ids != IGNORE_INDEX).sum().item())

        if not torch.isfinite(loss):
            raise FalhaNumericaError(f"loss não finita no passo {self.step}: {loss.item()!r}")

        try:
            self.scaler.scale(loss / self.config.grad_accum_steps).backward()
        except Exception as exc:
            if _e_erro_de_memoria(exc):
                raise FalhaDeMemoriaError(
                    f"memória do device '{self.config.device}' esgotada no passo {self.step} (backward): {exc}"
                ) from exc
            raise

        return loss.detach(), n_tokens_validos

    def _verificar_gradientes_finitos(self) -> bool:
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
        return True

    @torch.no_grad()
    def avaliar(self, val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict:
        """Avalia em `eval()`, sem gradientes. NUNCA recebe o split de
        teste — apenas validação."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        n_batches = 0
        try:
            for input_ids, target_ids in val_loader:
                if self.config.eval_max_batches is not None and n_batches >= self.config.eval_max_batches:
                    break
                input_ids = input_ids.to(self.config.device)
                target_ids = target_ids.to(self.config.device)
                try:
                    with self._autocast_ctx():
                        logits = self.model(input_ids)
                        n_tokens_validos = int((target_ids != IGNORE_INDEX).sum().item())
                        if n_tokens_validos == 0:
                            continue
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1),
                            ignore_index=IGNORE_INDEX,
                            reduction="sum",
                        )
                except Exception as exc:
                    if _e_erro_de_memoria(exc):
                        raise FalhaDeMemoriaError(
                            f"memória do device '{self.config.device}' esgotada durante avaliação "
                            f"no passo {self.step}: {exc}"
                        ) from exc
                    raise
                total_loss += loss.item()
                total_tokens += n_tokens_validos
                n_batches += 1
        finally:
            # Sempre restaura o modo treino, mesmo que uma excecao (incluindo
            # FalhaDeMemoriaError) interrompa o laco de avaliacao -- caso
            # contrario o modelo ficaria preso em eval() apos uma falha.
            self.model.train()

        if total_tokens == 0:
            return {"loss_validacao": None, "perplexidade": None, "n_batches_avaliados": 0}

        loss_media = total_loss / total_tokens
        try:
            perplexidade = math.exp(loss_media)
        except OverflowError:
            perplexidade = float("inf")
        return {
            "loss_validacao": loss_media,
            "perplexidade": perplexidade,
            "n_batches_avaliados": n_batches,
        }

    def salvar_checkpoint(self, nome: str | None = None) -> str:
        nome = nome or f"step{self.step}.pt"
        caminho = Path(self.config.out_dir) / nome
        epoca, cursor = self.epoca_atual, self.microbatches_consumidos
        if self._n_batches_epoca is not None and cursor == self._n_batches_epoca:
            epoca, cursor = epoca + 1, 0
        save_checkpoint(
            caminho,
            self.model,
            optimizer=self.optimizer,
            step=self.step,
            extra={
                "tokens_processados": self.tokens_processados,
                "melhor_loss_validacao": self.melhor_loss_validacao,
                "epoca_atual": epoca,
                "microbatches_consumidos": cursor,
                "avaliacoes_sem_melhora": self.avaliacoes_sem_melhora,
                "scaler_state_dict": self.scaler.state_dict(),
                "treinador_config": asdict(self.config),
            },
            metadata=self.metadata_checkpoint,
        )
        return str(caminho)

    def treinar(
        self,
        criar_loader_treino: CarregadorFactory,
        val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None,
        parar_no_passo: int | None = None,
    ) -> ResultadoTreinamento:
        """`criar_loader_treino(epoca) -> iterável de (input_ids, target_ids)`
        — uma função, não um DataLoader já pronto, para permitir reembaralhar
        de forma determinística a cada nova época (ver `dataloader.py`).

        `parar_no_passo`: ponto de PARADA OPERACIONAL desta chamada (ex.: para
        interromper deliberadamente um experimento maior e retomá-lo depois).
        É distinto de `self.config.max_steps`, que é o horizonte total do
        experimento e nunca é alterado por este parâmetro. O scheduler de LR
        (`_lr_atual` / `lr_no_passo`) sempre usa `self.config.max_steps`,
        nunca `parar_no_passo` — isso é o que garante que uma execução
        interrompida em `parar_no_passo` e depois retomada até `max_steps`
        produza a MESMA sequência de learning rates que uma execução
        contínua de `max_steps` passos. Se `parar_no_passo` for None (padrão),
        o treinamento roda até `max_steps`, comportamento idêntico ao
        anterior a este parâmetro. Se `parar_no_passo >= config.max_steps`,
        não tem efeito (o horizonte real já é o limite)."""
        inicio = time.perf_counter()
        limite_este_treino = (
            self.config.max_steps
            if parar_no_passo is None
            else min(parar_no_passo, self.config.max_steps)
        )
        motivo_parada = "max_steps_atingido"
        ultimo_checkpoint: str | None = None
        falha_tratada = False  # True apos o except de falha numerica/memoria, mesmo se o checkpoint de emergencia falhar

        # Retomada em fronteiras de optimizer step: o loader deve reconstruir
        # a mesma ordem por seed/época, sem transformações aleatórias ou workers.
        # Checkpoints antigos não possuem cursor e continuam sem retomada exata.
        self.model.train()
        try:
            while self.step < limite_este_treino:
                loader = criar_loader_treino(self.epoca_atual)
                self._n_batches_epoca = len(loader) if hasattr(loader, "__len__") else None
                cursor = self.microbatches_consumidos
                if cursor and getattr(loader, "num_workers", 0):
                    raise ValueError("Retomada com cursor requer num_workers=0.")
                # Reconstruir e saltar lotes determinísticos não deve avançar
                # o RNG global do modelo (o DataLoader cria uma base_seed).
                rng = torch.get_rng_state() if cursor else None
                iterator = iter(loader)
                for _ in range(cursor):
                    try:
                        next(iterator)
                    except StopIteration as exc:
                        raise ValueError("Cursor excede o tamanho do loader reconstruído.") from exc
                if rng is not None:
                    torch.set_rng_state(rng)
                acumulado_loss = 0.0
                acumulado_tokens = 0
                microbatches_no_acumulo = 0
                self.optimizer.zero_grad(set_to_none=True)
                tempo_passo_inicio = time.perf_counter()

                for indice, (input_ids, target_ids) in enumerate(iterator, start=cursor):
                    loss, n_tokens = self._passo_de_forward_backward(input_ids, target_ids)
                    acumulado_loss += loss.item()
                    acumulado_tokens += n_tokens
                    microbatches_no_acumulo += 1

                    if microbatches_no_acumulo < self.config.grad_accum_steps:
                        continue

                    try:
                        # unscale_ é um no-op quando o GradScaler está
                        # desabilitado (mixed_precision != "fp16") -- os
                        # gradientes já estão em escala real nesse caso.
                        self.scaler.unscale_(self.optimizer)
                    except Exception as exc:
                        if _e_erro_de_memoria(exc):
                            raise FalhaDeMemoriaError(
                                f"memória do device '{self.config.device}' esgotada no passo "
                                f"{self.step} (unscale de gradientes): {exc}"
                            ) from exc
                        raise

                    if not self._verificar_gradientes_finitos():
                        raise FalhaNumericaError(f"gradiente não finito no passo {self.step}")

                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    except Exception as exc:
                        if _e_erro_de_memoria(exc):
                            raise FalhaDeMemoriaError(
                                f"memória do device '{self.config.device}' esgotada no passo "
                                f"{self.step} (clip_grad_norm_): {exc}"
                            ) from exc
                        raise

                    lr_atual = self._lr_atual()
                    self._definir_lr(lr_atual)

                    try:
                        # scaler.step/update são passthrough (equivalentes a
                        # optimizer.step() puro) quando o GradScaler está
                        # desabilitado -- mesmo caminho de código para os
                        # três modos de mixed_precision ("no", "bf16", "fp16").
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    except Exception as exc:
                        if _e_erro_de_memoria(exc):
                            raise FalhaDeMemoriaError(
                                f"memória do device '{self.config.device}' esgotada no passo "
                                f"{self.step} (optimizer.step): {exc}"
                            ) from exc
                        raise
                    self.optimizer.zero_grad(set_to_none=True)

                    tempo_passo = time.perf_counter() - tempo_passo_inicio
                    self.tokens_processados += acumulado_tokens
                    self.step += 1
                    self.microbatches_consumidos = indice + 1

                    if self.step % self.config.log_every == 0:
                        self.registrador.registrar(
                            {
                                "step": self.step,
                                "epoca": self.epoca_atual,
                                "training_loss": acumulado_loss / microbatches_no_acumulo,
                                "learning_rate": lr_atual,
                                "grad_norm": float(grad_norm),
                                "tokens_acumulados": self.tokens_processados,
                                "tokens_por_segundo": acumulado_tokens / max(tempo_passo, 1e-9),
                                "tempo_por_step_s": tempo_passo,
                            }
                        )

                    if val_loader is not None and self.step % self.config.eval_every == 0:
                        metricas_val = self.avaliar(val_loader)
                        self.registrador.registrar({"step": self.step, **{f"validacao_{k}": v for k, v in metricas_val.items()}})
                        loss_val = metricas_val.get("loss_validacao")
                        if loss_val is not None:
                            if self.melhor_loss_validacao is None or loss_val < self.melhor_loss_validacao:
                                self.melhor_loss_validacao = loss_val
                                self.avaliacoes_sem_melhora = 0
                                if self.config.salvar_melhor:
                                    self.salvar_checkpoint("best.pt")
                            else:
                                self.avaliacoes_sem_melhora += 1
                            if (
                                self.config.paciencia_avaliacoes is not None
                                and self.avaliacoes_sem_melhora >= self.config.paciencia_avaliacoes
                            ):
                                motivo_parada = "early_stopping_sem_melhora_na_validacao"
                                ultimo_checkpoint = self.salvar_checkpoint("early_stopping.pt")
                                self.registrador.registrar({"step": self.step, "evento": motivo_parada})
                                return ResultadoTreinamento(
                                    passos_executados=self.step,
                                    tokens_processados=self.tokens_processados,
                                    motivo_parada=motivo_parada,
                                    melhor_loss_validacao=self.melhor_loss_validacao,
                                    tempo_total_s=time.perf_counter() - inicio,
                                    ultimo_checkpoint=ultimo_checkpoint,
                                )

                    if self.step % self.config.checkpoint_every == 0:
                        ultimo_checkpoint = self.salvar_checkpoint()
                        self.registrador.registrar({"step": self.step, "checkpoint_gerado": ultimo_checkpoint})

                    acumulado_loss = 0.0
                    acumulado_tokens = 0
                    microbatches_no_acumulo = 0
                    tempo_passo_inicio = time.perf_counter()

                    if self.step >= limite_este_treino:
                        break
                else:
                    # Mantém a política histórica de descartar a cauda parcial,
                    # agora explicitamente registrada, sem alterar a otimização.
                    if microbatches_no_acumulo:
                        self.registrador.registrar({"step": self.step,
                            "evento": "cauda_sem_optimizer_step",
                            "microbatches": microbatches_no_acumulo})
                    if self.microbatches_consumidos == cursor and cursor == 0:
                        raise ValueError("Loader não produziu um acúmulo completo de gradientes.")
                    self.epoca_atual += 1
                    self.microbatches_consumidos = 0

            # Distingue duas causas de saida normal do laco (sem falha
            # numerica e sem early stopping, que ja retornaram antes):
            # atingir de fato o horizonte planejado do experimento, ou
            # apenas atingir um ponto de parada operacional pedido via
            # `parar_no_passo` (execucao parcial, a ser retomada depois).
            if self.step >= self.config.max_steps:
                motivo_parada = "max_steps_atingido"
            else:
                motivo_parada = "parada_operacional_solicitada"
        except (FalhaNumericaError, FalhaDeMemoriaError) as exc:
            falha_tratada = True
            motivo_parada = "falha_de_memoria" if isinstance(exc, FalhaDeMemoriaError) else "falha_numerica"
            self.registrador.registrar({"step": self.step, "evento": motivo_parada, "detalhe": str(exc)})

            deve_propagar = (
                not self.config.parar_em_falha_de_memoria
                if motivo_parada == "falha_de_memoria"
                else not self.config.parar_em_falha_numerica
            )
            if deve_propagar:
                raise

            # Checkpoint de emergência em try/except: numa falha de memória
            # em particular, a própria tentativa de salvar pode falhar se a
            # memória estiver criticamente esgotada -- isso é registrado
            # explicitamente (nunca escondido), em vez de deixar uma segunda
            # exceção não tratada mascarar a causa raiz original.
            try:
                if (
                    motivo_parada == "falha_de_memoria"
                    and self.config.device == "cuda"
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()
                nome_checkpoint = "falha_de_memoria.pt" if motivo_parada == "falha_de_memoria" else "falha_numerica.pt"
                ultimo_checkpoint = self.salvar_checkpoint(nome_checkpoint)
            except Exception as exc_checkpoint:
                self.registrador.registrar(
                    {
                        "step": self.step,
                        "evento": "checkpoint_de_emergencia_falhou",
                        "detalhe": str(exc_checkpoint),
                    }
                )
                ultimo_checkpoint = None

        if not falha_tratada and (ultimo_checkpoint is None or self.step % self.config.checkpoint_every):
            ultimo_checkpoint = self.salvar_checkpoint(f"step{self.step}_final.pt")
        # Se `falha_tratada` e `ultimo_checkpoint` ainda e None aqui, o
        # proprio checkpoint de emergencia falhou (ja registrado como
        # "checkpoint_de_emergencia_falhou") -- NAO tentamos salvar de novo
        # com outro nome, pois a causa (memoria/disco esgotados) muito
        # provavelmente ainda se aplica; o resultado reflete isso com
        # ultimo_checkpoint=None em vez de mascarar a falha com uma segunda
        # excecao nao tratada.

        return ResultadoTreinamento(
            passos_executados=self.step,
            tokens_processados=self.tokens_processados,
            motivo_parada=motivo_parada,
            melhor_loss_validacao=self.melhor_loss_validacao,
            tempo_total_s=time.perf_counter() - inicio,
            ultimo_checkpoint=ultimo_checkpoint,
        )
