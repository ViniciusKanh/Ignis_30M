"""Testes de checkpoint de treinamento (Fase 3 — `ignis.training.checkpoint`).

Requer torch — não executável na VM Linux de desenvolvimento. Rode no
Windows nativo: `pytest tests/test_training_checkpoint.py -v`.
"""

import json

import pytest
import torch

from ignis.config import IgnisModelConfig
from ignis.model import IgnisModel
from ignis.training.checkpoint import (
    CheckpointMetadata,
    carregar_pesos_safetensors,
    load_checkpoint,
    salvar_pesos_safetensors,
    save_checkpoint,
    sha256_arquivo,
)


def test_checkpoint_round_trip_preserva_pesos_e_otimizador(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # dá um passo real para o otimizador ter estado (momentos do AdamW) para salvar.
    x = torch.randint(0, config_reduzida.vocab_size, (2, 4))
    loss = model(x).float().sum()
    loss.backward()
    optimizer.step()

    caminho = tmp_path / "ckpt.pt"
    save_checkpoint(caminho, model, optimizer=optimizer, step=7, extra={"nota": "teste"})
    assert caminho.exists()

    model_novo = IgnisModel(config_reduzida)
    optimizer_novo = torch.optim.AdamW(model_novo.parameters(), lr=1e-3)
    estado = load_checkpoint(caminho, model_novo, optimizer=optimizer_novo)

    assert estado["step"] == 7
    assert estado["extra"]["nota"] == "teste"
    for (nome_a, pa), (nome_b, pb) in zip(model.named_parameters(), model_novo.named_parameters()):
        assert nome_a == nome_b
        assert torch.equal(pa, pb)


def test_checkpoint_sem_otimizador_funciona(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    caminho = tmp_path / "ckpt.pt"
    save_checkpoint(caminho, model, step=0)

    model_novo = IgnisModel(config_reduzida)
    estado = load_checkpoint(caminho, model_novo)
    assert "optimizer_state_dict" not in estado


def test_checkpoint_nao_deixa_arquivo_temporario_apos_salvar(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    caminho = tmp_path / "ckpt.pt"
    save_checkpoint(caminho, model, step=0, exportar_safetensors=False)

    arquivos = list(tmp_path.iterdir())
    assert all(not nome.name.endswith(".tmp") for nome in arquivos), f"sobrou arquivo temporário: {arquivos}"


def test_checkpoint_metadata_detecta_tokenizador_diferente(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)

    arquivo_tokenizer_a = tmp_path / "tok_a.json"
    arquivo_tokenizer_a.write_text(json.dumps({"versao": "a"}), encoding="utf-8")
    arquivo_tokenizer_b = tmp_path / "tok_b.json"
    arquivo_tokenizer_b.write_text(json.dumps({"versao": "b"}), encoding="utf-8")

    metadata_salva = CheckpointMetadata.construir(
        config_modelo=config_reduzida.model_dump(),
        tokenizer_caminho=arquivo_tokenizer_a,
    )
    caminho = tmp_path / "ckpt.pt"
    save_checkpoint(caminho, model, step=0, metadata=metadata_salva)

    metadata_diferente = CheckpointMetadata.construir(tokenizer_caminho=arquivo_tokenizer_b)
    model_novo = IgnisModel(config_reduzida)
    with pytest.raises(ValueError, match="tokenizador"):
        load_checkpoint(caminho, model_novo, verificar_metadata=metadata_diferente)

    # com o MESMO tokenizador, a verificação deve passar sem erro.
    metadata_igual = CheckpointMetadata.construir(tokenizer_caminho=arquivo_tokenizer_a)
    load_checkpoint(caminho, model_novo, verificar_metadata=metadata_igual)


def test_sha256_arquivo_e_estavel(tmp_path):
    caminho = tmp_path / "arquivo.txt"
    caminho.write_text("conteudo de teste", encoding="utf-8")
    h1 = sha256_arquivo(caminho)
    h2 = sha256_arquivo(caminho)
    assert h1 == h2
    assert len(h1) == 64


def test_checkpoint_round_trip_preserva_estado_completo_do_adamw(config_reduzida: IgnisModelConfig, tmp_path):
    """Nao basta o `load_state_dict` do otimizador nao levantar erro: os
    MOMENTOS internos do AdamW (`exp_avg`, `exp_avg_sq`) e o contador de
    `step` por parametro precisam ser exatamente os mesmos apos o round
    trip, senao a retomada produziria atualizacoes de peso diferentes no
    proximo passo mesmo com os pesos do modelo idênticos no momento da
    retomada."""
    model = IgnisModel(config_reduzida)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # dois passos reais, para o AdamW acumular momentos nao triviais.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randint(0, config_reduzida.vocab_size, (2, 4))
        loss = model(x).float().sum()
        loss.backward()
        optimizer.step()

    caminho = tmp_path / "ckpt.pt"
    save_checkpoint(caminho, model, optimizer=optimizer, step=2, exportar_safetensors=False)

    model_novo = IgnisModel(config_reduzida)
    optimizer_novo = torch.optim.AdamW(model_novo.parameters(), lr=1e-3)
    load_checkpoint(caminho, model_novo, optimizer=optimizer_novo)

    estado_original = optimizer.state_dict()["state"]
    estado_restaurado = optimizer_novo.state_dict()["state"]
    assert set(estado_original.keys()) == set(estado_restaurado.keys())
    for chave in estado_original:
        original = estado_original[chave]
        restaurado = estado_restaurado[chave]
        assert original["step"] == restaurado["step"]
        assert torch.equal(original["exp_avg"], restaurado["exp_avg"])
        assert torch.equal(original["exp_avg_sq"], restaurado["exp_avg_sq"])


def test_safetensors_exportado_quando_disponivel(config_reduzida: IgnisModelConfig, tmp_path):
    pytest.importorskip("safetensors")
    model = IgnisModel(config_reduzida)
    caminho = tmp_path / "ckpt.pt"
    save_checkpoint(caminho, model, step=0, exportar_safetensors=True)
    assert caminho.with_suffix(".safetensors").exists()


def test_safetensors_exportado_e_recuperado_preserva_pesos(config_reduzida: IgnisModelConfig, tmp_path):
    """Validacao de export E RECUPERACAO (nao so export) do formato
    Safetensors, conforme exigido pela skill do projeto: carrega os pesos
    de volta em um modelo com inicializacao DIFERENTE e verifica que ficam
    identicos aos pesos originais, tensor a tensor."""
    pytest.importorskip("safetensors")

    model = IgnisModel(config_reduzida)
    caminho = tmp_path / "pesos.safetensors"
    salvar_pesos_safetensors(caminho, model)
    assert caminho.exists()

    torch.manual_seed(12345)  # garante inicializacao DIFERENTE da original
    model_novo = IgnisModel(config_reduzida)
    algum_diferente_antes = any(
        not torch.equal(pa, pb) for pa, pb in zip(model.parameters(), model_novo.parameters())
    )
    assert algum_diferente_antes, "pre-condicao do teste falhou: os dois modelos ja nasceram iguais"

    carregar_pesos_safetensors(caminho, model_novo)

    for (nome_a, pa), (nome_b, pb) in zip(model.named_parameters(), model_novo.named_parameters()):
        assert nome_a == nome_b
        assert torch.equal(pa, pb), f"parametro '{nome_a}' nao foi recuperado corretamente do Safetensors"
