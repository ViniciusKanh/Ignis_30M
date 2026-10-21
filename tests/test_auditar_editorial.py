"""Evita confundir conteúdo científico com títulos editoriais isolados."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "auditar_editorial", Path(__file__).parents[1] / "scripts/auditar_editorial.py")
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)


def test_nao_marca_mencoes_em_prosa_ou_formulas():
    texto = "As referências de tensão são V=RI.\nBibliografia científica é relevante."
    assert modulo.analisar_texto(texto)["inicio_cauda_candidata"] is None


def test_marca_titulo_isolado_e_preserva_posicao():
    texto = "Equação: E=mc²\n\n== Referências ==\nFonte legítima\n## Ligações externas\nURL"
    resultado = modulo.analisar_texto(texto)
    assert resultado["ocorrencias"] == {"referências": 1, "ligações externas": 1}
    assert texto[resultado["inicio_cauda_candidata"]:].startswith("== Referências")
