"""DataLoader de sequências tokenizadas para predição do próximo token.

Carrega os arquivos `.sequencias.npy` / `.mascaras.npy` / `.estatisticas.json`
produzidos por `ignis.data.pretrain_prep.salvar_dataset_tokenizado` (Fase 2) e
os expõe como um `torch.utils.data.Dataset` pronto para treinamento causal.

Deslocamento para next-token prediction: cada bloco salvo tem `seq_len`
tokens; o par (entrada, alvo) é obtido deslocando o próprio bloco em uma
posição (`entrada = bloco[:-1]`, `alvo = bloco[1:]`), produzindo um contexto
efetivo de treinamento de `seq_len - 1` tokens por amostra — isso evita
reprocessar os dados da Fase 2 (que já empacota os tokens em blocos de
`seq_len`) só para acrescentar um token extra por bloco. O contexto efetivo
nunca excede `max_seq_len` do Ignis-30M (1024): quem gera os blocos na Fase 2
já define `seq_len <= max_seq_len`, e `criar_dataloader` valida isso
explicitamente quando `max_seq_len` é informado.

Máscara de padding: como a máscara só marca padding no FINAL de um bloco
(ver `pretrain_prep.construir_sequencias`) e a atenção é estritamente causal
(cada posição só enxerga posições anteriores ou iguais a ela), nenhum token
real jamais atende a um token de padding — o padding só aparece depois de
todos os tokens reais do bloco. Por isso não é necessário mascarar a atenção
em si; basta impedir que posições de padding contribuam para a função de
perda, o que é feito atribuindo `IGNORE_INDEX` ao alvo nessas posições (lido
por `torch.nn.functional.cross_entropy(..., ignore_index=IGNORE_INDEX)`).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ..data.pretrain_prep import carregar_dataset_tokenizado

IGNORE_INDEX = -100


class SequenciasTokenizadasDataset(Dataset):
    """Dataset de blocos de tokens já preparados pela Fase 2 (um split por
    instância — nunca misture treino/validação/teste na mesma instância,
    isso é responsabilidade de quem chama, não deste dataset)."""

    def __init__(self, caminho_base: str | Path, max_seq_len: int | None = None):
        sequencias, mascaras, estatisticas = carregar_dataset_tokenizado(caminho_base)
        if sequencias.shape[0] == 0:
            raise ValueError(
                f"dataset tokenizado em '{caminho_base}' não possui nenhuma sequência "
                "(split vazio) — não é possível treinar/avaliar sobre um split vazio."
            )
        if sequencias.shape[1] < 2:
            raise ValueError(
                f"seq_len do dataset tokenizado ({sequencias.shape[1]}) precisa ser >= 2 "
                "para permitir o deslocamento entrada/alvo de next-token prediction."
            )
        if max_seq_len is not None and sequencias.shape[1] - 1 > max_seq_len:
            raise ValueError(
                f"contexto efetivo deste dataset ({sequencias.shape[1] - 1} tokens, "
                f"derivado de seq_len={sequencias.shape[1]}) excede max_seq_len "
                f"({max_seq_len}) da configuração do modelo — gere o dataset novamente "
                f"com um --seq-len compatível (seq_len <= max_seq_len + 1)."
            )

        self.sequencias = torch.from_numpy(sequencias).long()
        self.mascaras = torch.from_numpy(mascaras).long()
        self.estatisticas = estatisticas

    def __len__(self) -> int:
        return int(self.sequencias.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        bloco = self.sequencias[idx]
        mascara = self.mascaras[idx]

        input_ids = bloco[:-1].clone()
        target_ids = bloco[1:].clone()
        mascara_alvo = mascara[1:]
        target_ids[mascara_alvo == 0] = IGNORE_INDEX
        return input_ids, target_ids


def criar_dataloader(
    caminho_base: str | Path,
    batch_size: int,
    embaralhar: bool,
    seed: int = 0,
    epoch: int = 0,
    max_seq_len: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Cria um DataLoader para um split já tokenizado.

    `embaralhar` deve ser True apenas para o split de TREINO — validação e
    teste são sempre avaliados na ordem salva, sem embaralhamento, para que
    a avaliação seja determinística e comparável entre execuções.

    O embaralhamento usa um `torch.Generator` semeado com `seed + epoch` (e
    não um gerador global que avança continuamente) — isso permite retomar
    o treinamento a partir de um checkpoint e recriar exatamente a mesma
    permutação de uma época já iniciada, sem precisar serializar o estado
    interno do iterador do DataLoader.
    """
    dataset = SequenciasTokenizadasDataset(caminho_base, max_seq_len=max_seq_len)

    gerador = None
    if embaralhar:
        gerador = torch.Generator()
        gerador.manual_seed(seed + epoch)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=embaralhar,
        generator=gerador,
        drop_last=False,
        num_workers=num_workers,
    )
