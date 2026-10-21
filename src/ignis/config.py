"""Configuração tipada e validada do Ignis-30M.

Carrega e valida a hipótese de configuração definida em configs/ignis-30m.yaml
(ver PROMPT_INICIAL_CLAUDE.md, seção 2). A validação garante as restrições
arquiteturais exigidas pela skill ignis-llm-engineer antes de qualquer
instanciação do modelo:

- d_model deve ser divisível por n_heads;
- n_heads deve ser divisível por n_kv_heads (requisito de GQA);
- head_dim * n_heads deve ser igual a d_model (convenção adotada nesta
  implementação para simplificar a projeção de saída da atenção).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class IgnisModelConfig(BaseModel):
    """Hiperparâmetros arquiteturais do modelo Ignis."""

    nome: str = "ignis-30m"
    arquitetura: Literal["decoder_only_transformer"] = "decoder_only_transformer"

    vocab_size: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)

    activation: Literal["swiglu"] = "swiglu"
    normalization: Literal["rmsnorm"] = "rmsnorm"
    norm_eps: float = 1e-5
    positional_encoding: Literal["rope"] = "rope"
    rope_theta: float = 10000.0
    attention: Literal["grouped_query_attention"] = "grouped_query_attention"

    tie_word_embeddings: bool = True
    bias: bool = False
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validar_restricoes_arquiteturais(self) -> "IgnisModelConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) deve ser divisível por "
                f"n_heads ({self.n_heads})."
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) deve ser divisível por "
                f"n_kv_heads ({self.n_kv_heads}) para GQA."
            )
        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) não pode ser maior que "
                f"n_heads ({self.n_heads})."
            )
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(
                f"head_dim * n_heads ({self.head_dim} * {self.n_heads} = "
                f"{self.head_dim * self.n_heads}) deve ser igual a d_model "
                f"({self.d_model})."
            )
        return self

    @property
    def grupo_gqa(self) -> int:
        """Quantas cabeças de consulta compartilham cada cabeça KV."""
        return self.n_heads // self.n_kv_heads


class IgnisExecConfig(BaseModel):
    """Parâmetros de execução (não afetam a contagem de parâmetros)."""

    seed: int = 42
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"


class IgnisConfig(BaseModel):
    """Configuração completa: modelo + execução."""

    modelo: IgnisModelConfig
    execucao: IgnisExecConfig = Field(default_factory=IgnisExecConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IgnisConfig":
        caminho = Path(path)
        with caminho.open("r", encoding="utf-8") as arquivo:
            dados = yaml.safe_load(arquivo)
        return cls.model_validate(dados)

    @classmethod
    def ignis_30m(cls) -> "IgnisConfig":
        """Configuração canônica do Ignis-30M (hipótese aprovada na Fase 0)."""
        return cls(
            modelo=IgnisModelConfig(
                nome="ignis-30m",
                vocab_size=16384,
                max_seq_len=1024,
                n_layers=7,
                d_model=512,
                n_heads=8,
                n_kv_heads=4,
                head_dim=64,
                intermediate_size=1536,
            )
        )


class PilotoModeloRefConfig(BaseModel):
    """Referência à configuração arquitetural usada pelo piloto — NUNCA uma
    cópia duplicada dos hiperparâmetros, para impedir que o piloto divirja
    silenciosamente da arquitetura oficial do Ignis-30M (ver
    configs/ignis-30m.yaml, que esta config apenas referencia)."""

    config: str


class PilotoDadosConfig(BaseModel):
    """Identificação do manifesto/splits/tokenizer/dataset do piloto por
    caminho de arquivo (manifestos), nunca por dados brutos embutidos aqui."""

    manifesto: str
    splits: str
    tokenizer: str
    dataset_base: str
    seq_len: int = Field(gt=1)


class PilotoOtimizacaoConfig(BaseModel):
    """Hiperparâmetros de otimização do piloto. HIPÓTESES iniciais razoáveis
    para um piloto curto — nunca valores otimizados empiricamente."""

    lr: float = Field(gt=0)
    lr_min: float = Field(default=0.0, ge=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = Field(default=1.0, gt=0)


class PilotoLoteConfig(BaseModel):
    microbatch_size: int = Field(gt=0)
    grad_accum_steps: int = Field(default=1, gt=0)

    @property
    def lote_efetivo(self) -> int:
        """Lote efetivo = microbatch_size * grad_accum_steps. Documentado
        junto do resultado real do piloto — nunca citado como valor
        validado antes de uma execução real."""
        return self.microbatch_size * self.grad_accum_steps


class PilotoOrcamentoConfig(BaseModel):
    """`max_steps` é OBRIGATÓRIO (um piloto nunca treina indefinidamente).
    `tokens_alvo_hipotese`, quando presente, é só uma estimativa de
    planejamento (ex.: max_steps * lote_efetivo * (seq_len - 1)) — não é uma
    meta de token budget validada empiricamente."""

    max_steps: int = Field(gt=0)
    tokens_alvo_hipotese: int | None = Field(default=None, gt=0)


class PilotoValidacaoCheckpointConfig(BaseModel):
    eval_every: int = Field(gt=0)
    checkpoint_every: int = Field(gt=0)
    log_every: int = Field(gt=0)
    eval_max_batches: int | None = Field(default=None, gt=0)
    paciencia_avaliacoes: int | None = Field(default=None, gt=0)


class PilotoExecucaoConfig(BaseModel):
    seed: int = 42
    device: Literal["cpu", "cuda"] = "cpu"
    out_dir: str = "checkpoints/pilot-30m"
    parar_em_falha_numerica: bool = True
    parar_em_falha_de_memoria: bool = True
    # Precisao mista opt-in (Sprint 5) -- ver TreinadorConfig.mixed_precision
    # em ignis.training.trainer para a validacao exata (fp16 exige cuda).
    mixed_precision: Literal["no", "bf16", "fp16"] = "no"


class PilotoSmokeTestConfig(BaseModel):
    """Execução reduzida opcional (modelo pequeno + corpus sintético) para
    verificar mecanicamente atualização de parâmetros e retomada de
    checkpoint — ver scripts/smoke_training.py. NUNCA é o piloto real."""

    habilitado: bool = True
    max_steps: int = Field(default=10, gt=0)


class PilotoConfig(BaseModel):
    """Configuração completa de configs/train-pilot.yaml (Fase 3).

    IMPORTANTE: carregar e validar este arquivo NÃO autoriza, por si só, o
    início de um treinamento prolongado — ver docs/plano-corpus-piloto.md e
    a skill ignis-llm-engineer (autorização explícita continua sendo sempre
    exigida antes de qualquer execução real)."""

    modelo: PilotoModeloRefConfig
    dados: PilotoDadosConfig
    otimizacao: PilotoOtimizacaoConfig
    lote: PilotoLoteConfig
    orcamento: PilotoOrcamentoConfig
    validacao_e_checkpoint: PilotoValidacaoCheckpointConfig
    execucao: PilotoExecucaoConfig = Field(default_factory=PilotoExecucaoConfig)
    smoke_test: PilotoSmokeTestConfig = Field(default_factory=PilotoSmokeTestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PilotoConfig":
        caminho = Path(path)
        with caminho.open("r", encoding="utf-8") as arquivo:
            dados = yaml.safe_load(arquivo)
        return cls.model_validate(dados)

    def validar_contexto_contra_modelo(self, modelo: IgnisModelConfig) -> None:
        """Confere que o contexto efetivo do piloto (seq_len - 1, por causa
        do deslocamento next-token em `ignis.training.dataloader`) não
        excede `max_seq_len` do modelo referenciado. Nunca altera a
        arquitetura do modelo — só valida compatibilidade antes de usar."""
        contexto_efetivo = self.dados.seq_len - 1
        if contexto_efetivo > modelo.max_seq_len:
            raise ValueError(
                f"contexto efetivo do piloto ({contexto_efetivo}, derivado de "
                f"dados.seq_len={self.dados.seq_len}) excede max_seq_len do "
                f"modelo ({modelo.max_seq_len})"
            )

    def para_treinador_config(self):
        """Converte para `ignis.training.trainer.TreinadorConfig`. Import
        tardio e propositalmente isolado aqui: `ignis.training` depende de
        torch, mas o restante desta classe (carregar/validar o YAML) não
        depende e deve continuar funcionando mesmo sem torch instalado."""
        from .training.trainer import TreinadorConfig

        return TreinadorConfig(
            lr=self.otimizacao.lr,
            lr_min=self.otimizacao.lr_min,
            warmup_steps=self.otimizacao.warmup_steps,
            max_steps=self.orcamento.max_steps,
            weight_decay=self.otimizacao.weight_decay,
            beta1=self.otimizacao.beta1,
            beta2=self.otimizacao.beta2,
            eps=self.otimizacao.eps,
            grad_clip=self.otimizacao.grad_clip,
            grad_accum_steps=self.lote.grad_accum_steps,
            eval_every=self.validacao_e_checkpoint.eval_every,
            eval_max_batches=self.validacao_e_checkpoint.eval_max_batches,
            checkpoint_every=self.validacao_e_checkpoint.checkpoint_every,
            log_every=self.validacao_e_checkpoint.log_every,
            paciencia_avaliacoes=self.validacao_e_checkpoint.paciencia_avaliacoes,
            seed=self.execucao.seed,
            device=self.execucao.device,
            out_dir=self.execucao.out_dir,
            parar_em_falha_numerica=self.execucao.parar_em_falha_numerica,
            parar_em_falha_de_memoria=self.execucao.parar_em_falha_de_memoria,
            mixed_precision=self.execucao.mixed_precision,
        )
