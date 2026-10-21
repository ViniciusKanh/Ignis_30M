"""Triagem conservadora de possíveis informações pessoais (PII).

Heurística baseada em regex para sinalizar documentos para quarentena —
NUNCA exclui silenciosamente: toda detecção é registrada no manifesto com o
motivo e o tipo de padrão encontrado, para revisão humana posterior.

ISBN e DOI são excluídos somente dentro de seus intervalos sintáticos.
CNPJ permanece em quarentena por precaução. A triagem não comprova PII.
"""

from __future__ import annotations

import re

# Exceções restritas ao intervalo do identificador, nunca à vizinhança.
_RE_IDENTIFICADOR_BIBLIOGRAFICO = re.compile(
    r"\bISBN(?:-1[03])?\s*:?\s*(?:[0-9][ -]?){9,12}[0-9Xx](?![0-9])"
    r"|\bdoi\s*:\s*10\.\d{4,9}/[^\s]+", re.IGNORECASE
)

_PADROES: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "cnpj": re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
    "telefone_br": re.compile(r"\b(?:\+55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b"),
    "cartao_credito": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
}

# Padroes para os quais um contexto bibliografico ("ISBN" ou "doi")
# imediatamente anterior ao numero e tratado como falso positivo conhecido
# (ver docstring do modulo) -- nao se aplica a email/cpf/cnpj, que nao
# sofreram esse problema no corpus real auditado.
_PADROES_COM_EXCECAO_ISBN = frozenset({"telefone_br", "cartao_credito"})



# Exce??es restritas ao identificador reconhecido, n?o ? URL inteira.
_RE_IDENTIFICADORES_ADICIONAIS = (
    (
        re.compile(
            r"\b(?:https?://)?(?:dx\.)?doi\.org/10\.\d{4,9}/[^\s<>]+",
            re.IGNORECASE,
        ),
        frozenset({"telefone_br", "cartao_credito"}),
    ),
    (
        re.compile(r"\bOCLC\s*:?\s*\d{1,12}(?!\d)", re.IGNORECASE),
        frozenset({"telefone_br", "cartao_credito"}),
    ),
    (
        re.compile(
            r"https?://web\.archive\.org/web/"
            r"(?P<data>\d{14})(?:[a-z_]+)?/",
            re.IGNORECASE,
        ),
        frozenset({"cpf", "cnpj", "telefone_br", "cartao_credito"}),
    ),
    (
        re.compile(
            r"https?://www\.livrariascuritiba\.com\.br/"
            r"BaseCD/\d{13}/capa\.jpg\b",
            re.IGNORECASE,
        ),
        frozenset({"cartao_credito"}),
    ),
)


def detectar_pii(texto: str) -> dict[str, int]:
    """Conta candidatos, excluindo apenas identificadores reconhecidos."""
    from datetime import datetime

    intervalos = [
        (m.start(), m.end(), _PADROES_COM_EXCECAO_ISBN)
        for m in _RE_IDENTIFICADOR_BIBLIOGRAFICO.finditer(texto)
    ]

    for padrao, tipos in _RE_IDENTIFICADORES_ADICIONAIS:
        for ocorrencia in padrao.finditer(texto):
            if "data" in ocorrencia.groupdict():
                try:
                    datetime.strptime(ocorrencia.group("data"), "%Y%m%d%H%M%S")
                except ValueError:
                    continue
                # Exclui somente o timestamp, n?o o endere?o arquivado.
                inicio, fim = ocorrencia.span("data")
            else:
                inicio, fim = ocorrencia.span()
            intervalos.append((inicio, fim, tipos))

    achados = {}
    for nome, padrao in _PADROES.items():
        quantidade = 0
        for ocorrencia in padrao.finditer(texto):
            inicio = ocorrencia.start()
            fim = inicio + len(ocorrencia.group().rstrip(" -"))
            identificador_conhecido = any(
                nome in tipos and a <= inicio and fim <= b
                for a, b, tipos in intervalos
            )
            if not identificador_conhecido:
                quantidade += 1
        if quantidade:
            achados[nome] = quantidade
    return achados
