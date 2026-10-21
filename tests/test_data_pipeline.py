from pathlib import Path

from ignis.data.manifest import StatusDocumento
from ignis.data.pipeline import carregar_manifesto, processar_diretorio, salvar_manifesto


def _texto_longo(sufixo: str = "") -> str:
    return ("Este e um texto suficientemente longo para passar nos filtros de qualidade. " * 10) + sufixo


def test_documento_sem_licenca_nao_e_aceito(tmp_path: Path):
    fonte = tmp_path / "fonte"
    fonte.mkdir()
    (fonte / "sem_licenca.txt").write_text(_texto_longo(), encoding="utf-8")

    resultado = processar_diretorio(tmp_path)
    entrada = next(e for e in resultado.manifesto if e.documento_id == "sem_licenca")
    assert entrada.status == StatusDocumento.LICENCA_DESCONHECIDA
    assert entrada.motivo  # motivo sempre registrado


def test_documento_com_licenca_e_permissao_e_aceito(tmp_path: Path):
    import json

    fonte = tmp_path / "fonte"
    fonte.mkdir()
    jsonl = fonte / "docs.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "id": "com_licenca",
                "texto": _texto_longo(),
                "licenca": "CC-BY-4.0",
                "permissao_treinamento": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    resultado = processar_diretorio(tmp_path)
    entrada = next(e for e in resultado.manifesto if e.documento_id == "com_licenca")
    assert entrada.status == StatusDocumento.ACEITO
    assert "com_licenca" in resultado.textos_aceitos


def test_duplicata_exata_e_rejeitada_sem_remover_a_primeira(tmp_path: Path):
    fonte = tmp_path / "fonte"
    fonte.mkdir()
    texto = _texto_longo()
    (fonte / "original.txt").write_text(texto, encoding="utf-8")
    (fonte / "copia.txt").write_text(texto, encoding="utf-8")

    resultado = processar_diretorio(tmp_path)
    status = {e.documento_id: e.status for e in resultado.manifesto}
    # exatamente um dos dois deve ser rejeitado como duplicata; o outro mantém seu status normal
    rejeitados = [doc_id for doc_id, s in status.items() if s == StatusDocumento.REJEITADO]
    assert len(rejeitados) == 1


def test_documento_curto_e_rejeitado_por_qualidade(tmp_path: Path):
    fonte = tmp_path / "fonte"
    fonte.mkdir()
    (fonte / "curto.txt").write_text("muito curto", encoding="utf-8")

    resultado = processar_diretorio(tmp_path)
    entrada = next(e for e in resultado.manifesto if e.documento_id == "curto")
    assert entrada.status == StatusDocumento.REJEITADO
    assert "curto" in entrada.motivo or "caracteres" in entrada.motivo


def test_documento_com_pii_vai_para_quarentena_nao_e_excluido(tmp_path: Path):
    fonte = tmp_path / "fonte"
    fonte.mkdir()
    texto = _texto_longo(" Contato: pessoa@example.com")
    (fonte / "com_pii.txt").write_text(texto, encoding="utf-8")

    resultado = processar_diretorio(tmp_path)
    entrada = next(e for e in resultado.manifesto if e.documento_id == "com_pii")
    assert entrada.status == StatusDocumento.QUARENTENA
    assert "PII" in entrada.motivo
    # quarentena nao e exclusao silenciosa: o documento permanece no manifesto
    assert entrada.documento_id not in resultado.textos_aceitos


def test_salvar_e_carregar_manifesto_preserva_dados(tmp_path: Path):
    fonte = tmp_path / "fonte"
    fonte.mkdir()
    (fonte / "doc.txt").write_text(_texto_longo(), encoding="utf-8")

    resultado = processar_diretorio(tmp_path)
    caminho = tmp_path / "manifesto.jsonl"
    salvar_manifesto(resultado, caminho)
    recarregado = carregar_manifesto(caminho)

    assert len(recarregado) == len(resultado.manifesto)
    assert {e.documento_id for e in recarregado} == {e.documento_id for e in resultado.manifesto}
