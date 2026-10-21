"""Normalização Unicode e limpeza de artefatos de texto."""

from __future__ import annotations

import re
import unicodedata

_RE_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_ESPACOS_MULTIPLOS = re.compile(r"[ \t]{2,}")
_RE_LINHAS_MULTIPLAS = re.compile(r"\n{3,}")
def normalizar_texto(texto: str, forma: str = "NFC") -> str:
    """Normaliza Unicode e espaços, preservando fórmulas e conteúdo técnico."""
    texto = unicodedata.normalize(forma, texto)
    texto = _RE_CONTROLE.sub("", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = _RE_ESPACOS_MULTIPLOS.sub(" ", texto)
    texto = _RE_LINHAS_MULTIPLAS.sub("\n\n", texto)
    return texto.strip()
