"""Filtros configuráveis de tamanho e qualidade de documentos."""

from __future__ import annotations

from pydantic import BaseModel


class FiltrosQualidade(BaseModel):
    min_caracteres: int = 200
    max_caracteres: int = 2_000_000
    razao_minima_alfabetica: float = 0.5


def avaliar_qualidade(texto: str, filtros: FiltrosQualidade) -> tuple[bool, str]:
    n = len(texto)
    if n < filtros.min_caracteres:
        return False, f"muito curto ({n} < {filtros.min_caracteres} caracteres)"
    if n > filtros.max_caracteres:
        return False, f"muito longo ({n} > {filtros.max_caracteres} caracteres)"
    alfabeticos = sum(1 for c in texto if c.isalpha())
    # Razão calculada sobre caracteres NÃO-ESPAÇO (Sprint 6.2), não sobre o
    # tamanho total do texto: documentos legítimos com fórmulas matemáticas
    # (ex.: artigos de álgebra linear extraídos da Wikipédia) podem ter uma
    # fração anormalmente alta de espaços/quebras de linha vindos da
    # renderização em texto simples de fórmulas (cada símbolo da fórmula
    # vira uma "linha" curta cercada de espaços) -- isso derruba a razão
    # calculada sobre o total sem refletir qualidade de prosa real. Dividir
    # pelos caracteres não-espaço mede a mesma coisa (fração alfabética do
    # conteúdo que realmente carrega informação) sem penalizar texto
    # legítimo por causa de formatação espaçada. Para a imensa maioria dos
    # textos normais (onde espaços são ~15-20% do texto, como de costume em
    # prosa em português) a mudança tem efeito desprezível; o filtro
    # continua rejeitando texto genuinamente degenerado (ex.: sopa de
    # números/símbolos com espaçamento normal).
    caracteres_nao_espaco = sum(1 for c in texto if not c.isspace())
    razao = alfabeticos / caracteres_nao_espaco if caracteres_nao_espaco else 0.0
    if razao < filtros.razao_minima_alfabetica:
        return False, f"baixa razão de caracteres alfabéticos ({razao:.2f} < {filtros.razao_minima_alfabetica})"
    return True, ""
