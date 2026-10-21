# Arquitetura do Ignis-30M

Configuração aprovada (Fase 0), ver `configs/ignis-30m.yaml` e `PROMPT_INICIAL_CLAUDE.md` seção 2.

## Componentes

- **RMSNorm** (`src/ignis/model/rmsnorm.py`): normalização por raiz da média quadrática, sem centralização nem bias — apenas um vetor de escala por dimensão.
- **RoPE** (`src/ignis/model/rope.py`): embeddings posicionais rotativos aplicados a Q e K antes da atenção; `offset` permite deslocar a origem das posições para uso futuro em geração incremental (ainda não implementada — ver limitações).
- **Atenção causal com GQA** (`src/ignis/model/attention.py`): 8 cabeças de consulta, 4 cabeças KV (grupo de 2), projeções Wq/Wk/Wv/Wo sem bias. As cabeças KV são expandidas via `repeat_interleave` antes de `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`, abordagem portátil que não depende do suporte nativo a GQA de versões específicas do PyTorch.
- **SwiGLU** (`src/ignis/model/mlp.py`): `down_proj(silu(gate_proj(x)) * up_proj(x))`, três projeções lineares sem bias.
- **Bloco Transformer** (`src/ignis/model/block.py`): pré-norma (RMSNorm) antes da atenção e antes do MLP, com conexões residuais.
- **Modelo completo** (`src/ignis/model/transformer.py`): embedding de entrada, 7 blocos, RMSNorm final, `lm_head` com pesos compartilhados com o embedding de entrada (`tie_word_embeddings: true`).

## Contagem de parâmetros

Derivação analítica em `src/ignis/model/param_count.py::contagem_parametros_esperada`, validada por `scripts/count_params.py` e pela suíte de testes (`tests/test_param_count.py`) contra `IgnisModel.contar_parametros()` (equivalente a `sum(p.numel() for p in model.parameters())`).

Por camada (×7):
- Atenção: Wq (512×512) + Wk (512×256) + Wv (512×256) + Wo (512×512) = 786.432
- MLP SwiGLU: 3 × (512×1536) = 2.359.296
- 2× RMSNorm: 2×512 = 1.024
- Subtotal por camada: 3.146.752 × 7 camadas = 22.027.264

Mais:
- Embedding compartilhado (entrada/saída): 16.384 × 512 = 8.388.608
- RMSNorm final: 512

**Total: 30.416.384 parâmetros treináveis.**

Esse número ainda não foi confirmado por execução real (`sum(p.numel())`) porque o PyTorch não pôde ser instalado neste ambiente — ver `docs/limitations.md`.

## Validações de configuração

`src/ignis/config.py` (Pydantic) impõe antes de qualquer instanciação:
- `d_model % n_heads == 0`
- `n_heads % n_kv_heads == 0`
- `n_kv_heads <= n_heads`
- `head_dim * n_heads == d_model`

## Geração autoregressiva

`src/ignis/inference/generate.py` recalcula o forward completo a cada token gerado (sem cache de K/V) — suficiente para smoke tests, mas não eficiente para geração longa. Suporta decodificação greedy (`temperature <= 0`), temperatura, top-k e top-p.
