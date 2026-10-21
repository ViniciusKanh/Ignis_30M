"""Tokenizador BPE do Ignis, usando a biblioteca Hugging Face `tokenizers`.

Tokens especiais e IDs (fixos, reservados nas 4 primeiras posições do
vocabulário por construção do treinamento — ver verificação em
`train_tokenizer`):

    0 -> <pad>
    1 -> <bos>
    2 -> <eos>
    3 -> <unk>

O vocabulário-alvo oficial é 16.384 (IgnisModelConfig.vocab_size), mas o
tokenizador só deve ser treinado com esse alvo quando houver corpus real
suficiente. Com corpus pequeno (testes, por exemplo), o vocabulário final
fica menor que o alvo — isso é esperado e **nunca é completado
artificialmente** (ver aviso em `IgnisTokenizer.train_tokenizer`).
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

TOKENS_ESPECIAIS = ["<pad>", "<bos>", "<eos>", "<unk>"]
ID_PAD, ID_BOS, ID_EOS, ID_UNK = range(4)


class IgnisTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    @classmethod
    def train_tokenizer(
        cls,
        arquivos_treino: list[str | Path],
        vocab_size: int = 16384,
        min_frequency: int = 2,
    ) -> "IgnisTokenizer":
        if not arquivos_treino:
            raise ValueError("arquivos_treino não pode ser vazio")

        modelo = BPE(unk_token="<unk>")
        tokenizer = Tokenizer(modelo)
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=TOKENS_ESPECIAIS,
            # Inclui o alfabeto de bytes completo (256 símbolos) desde o
            # início: com ByteLevel, isso garante que QUALQUER sequência de
            # bytes seja representável sem cair em <unk>, mesmo com corpus
            # de treino pequeno (caso contrário, caracteres/combinações não
            # vistas no treino quebram o round-trip da decodificação).
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
            show_progress=False,
        )
        tokenizer.train([str(p) for p in arquivos_treino], trainer)

        # Verificação de sanidade: os tokens especiais devem ocupar
        # exatamente os IDs 0-3, na ordem declarada em TOKENS_ESPECIAIS.
        for id_esperado, token in enumerate(TOKENS_ESPECIAIS):
            id_obtido = tokenizer.token_to_id(token)
            if id_obtido != id_esperado:
                raise RuntimeError(
                    f"token especial '{token}' recebeu id {id_obtido}, "
                    f"esperado {id_esperado} — treinamento inconsistente"
                )

        return cls(tokenizer)

    def encode(self, texto: str, adicionar_bos_eos: bool = True) -> list[int]:
        ids = self._tok.encode(texto).ids
        if adicionar_bos_eos:
            ids = [ID_BOS, *ids, ID_EOS]
        return ids

    def decode(self, ids: list[int], remover_especiais: bool = True) -> str:
        if remover_especiais:
            ids = [i for i in ids if i not in (ID_PAD, ID_BOS, ID_EOS)]
        return self._tok.decode(ids, skip_special_tokens=False)

    def save(self, caminho: str | Path) -> None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(caminho))

    @classmethod
    def load(cls, caminho: str | Path) -> "IgnisTokenizer":
        return cls(Tokenizer.from_file(str(caminho)))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()
