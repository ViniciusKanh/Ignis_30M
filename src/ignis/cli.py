"""CLI do pipeline de dados, tokenizador e treinamento do Ignis (Fases 2-4).

Subcomandos:
    validate-corpus   Ingere, limpa, triagem e gera o manifesto de um corpus local
    prepare-splits    Gera splits reproduzíveis (treino/validação/teste) a partir de um manifesto
    train-tokenizer   Treina o tokenizador BPE exclusivamente no split de treino
    eval-tokenizer    Avalia round-trip e métricas de eficiência do tokenizador
    prepare-dataset   Converte os splits tokenizados em sequências para predição do próximo token
    report            Gera um relatório consolidado (manifesto + splits + tokenizador, quando disponíveis)
    audit-tokenizer   Audita se corpus+tokenizador atendem aos critérios objetivos (seção 8 de
                      docs/plano-corpus-piloto.md) para serem declarados "oficiais do piloto"
    train             Dry-run OBRIGATÓRIO (arquivos/shapes/memória/forward de teste) do piloto;
                      só inicia treinamento real com --iniciar-treinamento-real E dry-run aprovado

Exemplos:
    python -m ignis.cli validate-corpus --corpus-dir corpus/ --manifesto-saida data/manifesto.jsonl
    python -m ignis.cli prepare-splits --manifesto data/manifesto.jsonl --splits-saida data/splits.json
    python -m ignis.cli train-tokenizer --corpus-dir corpus/ --manifesto data/manifesto.jsonl --splits data/splits.json --saida data/tokenizer.json
    python -m ignis.cli eval-tokenizer --tokenizer data/tokenizer.json --corpus-dir corpus/ --manifesto data/manifesto.jsonl --splits data/splits.json
    python -m ignis.cli prepare-dataset --tokenizer data/tokenizer.json --corpus-dir corpus/ --manifesto data/manifesto.jsonl --splits data/splits.json --saida-base data/tokenizado
    python -m ignis.cli report --manifesto data/manifesto.jsonl --splits data/splits.json --saida data/relatorio.json
    python -m ignis.cli audit-tokenizer --tokenizer data/tokenizer.json --corpus-dir corpus/ --manifesto data/manifesto.jsonl --splits data/splits.json --textos-externos data/textos_externos.txt --dataset-base data/tokenizado
    python -m ignis.cli train --config configs/train-pilot.yaml
    python -m ignis.cli train --config configs/train-pilot.yaml --iniciar-treinamento-real
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .data.ingest import ingerir_diretorio
from .data.manifest import EntradaManifesto, StatusDocumento
from .data.normalize import normalizar_texto
from .data.pipeline import carregar_manifesto, processar_diretorio, salvar_manifesto
from .data.splits import (
    AtribuicaoSplits,
    GruposInsuficientesError,
    ProporcoesSplit,
    carregar_splits,
    contar_unidades_por_fonte,
    gerar_splits,
    salvar_splits,
    verificar_contaminacao,
)


def _carregar_textos_do_split(
    manifesto: list[EntradaManifesto], corpus_dir: Path, ids_do_split: list[str]
) -> dict[str, str]:
    ids_desejados = set(ids_do_split)
    textos: dict[str, str] = {}
    for doc in ingerir_diretorio(corpus_dir):
        if doc.documento_id in ids_desejados:
            textos[doc.documento_id] = normalizar_texto(doc.texto)
    faltando = ids_desejados - textos.keys()
    if faltando:
        print(
            f"AVISO: {len(faltando)} documento(s) do split não encontrados ao reler "
            f"o corpus (ex.: {sorted(faltando)[:5]})",
            file=sys.stderr,
        )
    return textos


def cmd_validate_corpus(args: argparse.Namespace) -> int:
    destino = Path(args.manifesto_saida)
    incremental = getattr(args, "incremental", False)
    limite_novos_documentos = getattr(args, "limite_novos_documentos", None)
    manifesto_existente = None
    if incremental:
        if destino.is_file():
            manifesto_existente = carregar_manifesto(destino)
            print(
                f"execucao incremental: {len(manifesto_existente)} documento(s) ja presentes em "
                f"{destino} (apenas documentos novos ou com conteudo alterado serao reprocessados)",
                file=sys.stderr,
            )
        else:
            print(
                f"execucao incremental: {destino} ainda nao existe; tratando como primeira execucao",
                file=sys.stderr,
            )

    resultado = processar_diretorio(
        Path(args.corpus_dir),
        fonte=args.fonte,
        manifesto_existente=manifesto_existente,
        limite_novos_documentos=limite_novos_documentos,
    )
    salvar_manifesto(resultado, destino)
    print(json.dumps(resultado.relatorio(), ensure_ascii=False, indent=2))
    print(f"\nmanifesto salvo em: {args.manifesto_saida}")
    if limite_novos_documentos is not None:
        print(
            f"limite de documentos novos/atualizados desta execucao: {limite_novos_documentos} "
            f"(processados de fato: {resultado.n_novos_processados}) -- rode novamente com "
            "--incremental para continuar de onde parou",
            file=sys.stderr,
        )
    return 0


def cmd_prepare_splits(args: argparse.Namespace) -> int:
    manifesto = carregar_manifesto(Path(args.manifesto))
    proporcoes = ProporcoesSplit(treino=args.treino, validacao=args.validacao, teste=args.teste)
    try:
        atribuicao = gerar_splits(
            manifesto, seed=args.seed, proporcoes=proporcoes, agrupar_por_fonte=not args.ignorar_fonte
        )
    except GruposInsuficientesError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    contaminacao = verificar_contaminacao(atribuicao)
    salvar_splits(atribuicao, Path(args.splits_saida))

    relatorio = atribuicao.relatorio()
    print(
        f"treino={len(atribuicao.treino)} validacao={len(atribuicao.validacao)} "
        f"teste={len(atribuicao.teste)}  (unidades independentes: {atribuicao.n_unidades})"
    )
    print(f"proporções solicitadas: {relatorio['proporcoes_solicitadas']}")
    print(f"proporções realizadas:  {relatorio['proporcoes_realizadas']}")
    if contaminacao:
        print(f"ERRO: {len(contaminacao)} documento(s) contaminando splits: {contaminacao}", file=sys.stderr)
        return 1
    print("nenhuma contaminação detectada entre splits")
    print(f"splits salvos em: {args.splits_saida}")
    return 0


def cmd_train_tokenizer(args: argparse.Namespace) -> int:
    from .tokenizer.bpe import IgnisTokenizer

    manifesto = carregar_manifesto(Path(args.manifesto))
    splits = carregar_splits(Path(args.splits))
    textos_treino = _carregar_textos_do_split(manifesto, Path(args.corpus_dir), splits.treino)

    if not textos_treino:
        print("ERRO: nenhum documento de treino disponível para treinar o tokenizador", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        arquivos = []
        for doc_id, texto in textos_treino.items():
            caminho = Path(tmp) / f"{doc_id}.txt"
            caminho.write_text(texto, encoding="utf-8")
            arquivos.append(caminho)
        tokenizador = IgnisTokenizer.train_tokenizer(
            arquivos, vocab_size=args.vocab_size, min_frequency=args.min_frequency
        )

    destino = Path(args.saida)
    if destino.exists() and not args.sobrescrever:
        print(f"ERRO: {destino} já existe; use --sobrescrever para substituir", file=sys.stderr)
        return 1
    tokenizador.save(destino)

    print(f"tokenizador treinado em {len(textos_treino)} documento(s) de treino")
    print(f"vocab_size alvo={args.vocab_size}  vocab_size real={tokenizador.vocab_size}")
    if tokenizador.vocab_size < args.vocab_size:
        print(
            "AVISO: corpus insuficiente para atingir o vocab_size alvo; o vocabulário "
            "final é menor e NÃO foi completado artificialmente. Este tokenizador NÃO "
            "deve ser declarado 'oficial' do Ignis-30M até que um corpus adequado "
            "(ver docs/plano-corpus-piloto.md) permita atingir o vocab_size alvo.",
            file=sys.stderr,
        )
    print(f"tokenizador salvo em: {destino}")
    return 0


def cmd_eval_tokenizer(args: argparse.Namespace) -> int:
    from .tokenizer.bpe import IgnisTokenizer
    from .tokenizer.metrics import metricas_eficiencia

    tokenizador = IgnisTokenizer.load(Path(args.tokenizer))
    manifesto = carregar_manifesto(Path(args.manifesto))
    splits = carregar_splits(Path(args.splits))

    ids_avaliacao = splits.validacao or splits.treino
    textos = _carregar_textos_do_split(manifesto, Path(args.corpus_dir), ids_avaliacao)

    falhas_round_trip = []
    for doc_id, texto in textos.items():
        ids = tokenizador.encode(texto)
        decodificado = tokenizador.decode(ids)
        if decodificado.strip() != texto.strip():
            falhas_round_trip.append(doc_id)

    metricas = metricas_eficiencia(tokenizador, list(textos.values()))
    resultado = {
        "vocab_size": tokenizador.vocab_size,
        "documentos_avaliados": len(textos),
        "falhas_round_trip": falhas_round_trip,
        "metricas_eficiencia": metricas,
    }

    if args.textos_externos:
        caminho_externo = Path(args.textos_externos)
        linhas_externas = [
            linha.strip() for linha in caminho_externo.read_text(encoding="utf-8").splitlines() if linha.strip()
        ]
        falhas_externas = []
        for i, texto_externo in enumerate(linhas_externas):
            ids = tokenizador.encode(texto_externo)
            decodificado = tokenizador.decode(ids)
            if decodificado.strip() != texto_externo.strip():
                falhas_externas.append(i)
        resultado["avaliacao_externa"] = {
            "descricao": (
                "Textos NÃO usados em nenhum split, fornecidos via --textos-externos (um por "
                "linha), para detectar overfitting do vocabulário do tokenizador aos dados de "
                "treino (ver docs/plano-corpus-piloto.md, seção 7)."
            ),
            "arquivo": str(caminho_externo),
            "n_linhas": len(linhas_externas),
            "falhas_round_trip": falhas_externas,
            "metricas_eficiencia": metricas_eficiencia(tokenizador, linhas_externas),
        }

    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    if args.saida:
        Path(args.saida).write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if falhas_round_trip else 0


def cmd_audit_tokenizer(args: argparse.Namespace) -> int:
    """Audita, de forma mecanica e objetiva, se o corpus + tokenizador
    atendem simultaneamente aos 4 criterios da secao 8 de
    docs/plano-corpus-piloto.md para serem declarados "oficiais do piloto
    Ignis-30M" (bloqueia, nunca aprova por omissao):

    1. Licenca e permissao de treinamento documentadas -- zero documentos
       com status 'licenca_desconhecida' no manifesto.
    2. Zero contaminacao entre os splits treino/validacao/teste.
    3. Round-trip do tokenizador sem NENHUMA falha em treino, validacao e
       um conjunto externo nao usado em nenhum split (--textos-externos e
       obrigatorio aqui, ao contrario de `eval-tokenizer`).
    4. Volume de tokens de treino dentro da faixa alvo, medido de fato
       pelas estatisticas reais de `prepare-dataset` (nunca estimado).

    `oficial_do_piloto` so e True quando TODOS os criterios acima sao
    True -- esta funcao nunca promove parcialmente."""
    from .tokenizer.bpe import IgnisTokenizer

    criterios: list[dict] = []

    def _criterio(nome: str, ok: bool, detalhe: str) -> None:
        criterios.append({"nome": nome, "ok": bool(ok), "detalhe": detalhe})

    manifesto = carregar_manifesto(Path(args.manifesto))
    splits = carregar_splits(Path(args.splits))
    tokenizador = IgnisTokenizer.load(Path(args.tokenizer))

    _criterio("vocabulario_compativel", tokenizador.vocab_size == 16384,
              f"vocabulário efetivo={tokenizador.vocab_size}; exigido=16384")
    ids = splits.treino + splits.validacao + splits.teste
    aceitos = {e.documento_id for e in manifesto if e.status == StatusDocumento.ACEITO}
    _criterio("cobertura_dos_splits", set(ids) == aceitos and len(ids) == len(set(ids)),
              "splits devem conter todos e somente os aceitos, uma vez cada")
    por_split = {id_: nome for nome in ("treino", "validacao", "teste")
                 for id_ in getattr(splits, nome)}
    vazamentos = []
    for atributo in ("sha256", "unidade_agrupamento", "grupo_quase_duplicata"):
        grupos = {}
        for e in manifesto:
            chave = getattr(e, atributo)
            if chave and e.documento_id in por_split:
                grupos.setdefault(chave, set()).add(por_split[e.documento_id])
        if any(len(g) > 1 for g in grupos.values()):
            vazamentos.append(atributo)
    _criterio("grupos_sem_vazamento", not vazamentos, f"atributos com vazamento: {vazamentos}")

    n_licenca_desconhecida = sum(1 for e in manifesto if e.status == StatusDocumento.LICENCA_DESCONHECIDA)
    _criterio(
        "licencas_documentadas",
        n_licenca_desconhecida == 0,
        f"{n_licenca_desconhecida} documento(s) com status 'licenca_desconhecida' no manifesto "
        "(secao 2 de docs/plano-corpus-piloto.md exige licenca e permissao de treinamento "
        "documentadas para toda fonte usada)",
    )

    contaminacao = verificar_contaminacao(splits)
    _criterio(
        "splits_sem_contaminacao",
        len(contaminacao) == 0,
        f"{len(contaminacao)} documento(s) presentes em mais de um split: {contaminacao[:5]}"
        if contaminacao
        else "0 documentos contaminados entre treino/validacao/teste",
    )

    def _avaliar_round_trip(ids_split: list[str], nome_split: str) -> None:
        if not ids_split:
            _criterio(f"round_trip_sem_falhas:{nome_split}", False, f"split '{nome_split}' vazio -- nao foi possivel avaliar")
            return
        textos = _carregar_textos_do_split(manifesto, Path(args.corpus_dir), ids_split)
        falhas = [
            doc_id
            for doc_id, texto in textos.items()
            if tokenizador.decode(tokenizador.encode(texto)).strip() != texto.strip()
        ]
        _criterio(
            f"round_trip_sem_falhas:{nome_split}",
            len(falhas) == 0 and len(textos) == len(set(ids_split)),
            f"{len(falhas)} falha(s) de round-trip em {len(textos)} documento(s) do split '{nome_split}'"
            + (f": {falhas[:5]}" if falhas else ""),
        )

    _avaliar_round_trip(splits.treino, "treino")
    _avaliar_round_trip(splits.validacao, "validacao")

    if not args.textos_externos:
        _criterio(
            "round_trip_sem_falhas:externo",
            False,
            "--textos-externos nao foi fornecido -- obrigatorio para a auditoria oficial "
            "(diferente de eval-tokenizer, onde e opcional); ver secao 7 de "
            "docs/plano-corpus-piloto.md",
        )
    else:
        caminho_externo = Path(args.textos_externos)
        linhas_externas = [
            linha.strip() for linha in caminho_externo.read_text(encoding="utf-8").splitlines() if linha.strip()
        ]
        falhas_externas = [
            i
            for i, texto_externo in enumerate(linhas_externas)
            if tokenizador.decode(tokenizador.encode(texto_externo)).strip() != texto_externo.strip()
        ]
        _criterio(
            "round_trip_sem_falhas:externo",
            len(linhas_externas) > 0 and len(falhas_externas) == 0,
            f"{len(falhas_externas)} falha(s) de round-trip em {len(linhas_externas)} linha(s) "
            f"externas do arquivo {caminho_externo}",
        )

    base_treino = Path(args.dataset_base).with_name(f"{Path(args.dataset_base).name}_treino")
    caminho_stats = Path(f"{base_treino}.estatisticas.json")
    if caminho_stats.is_file():
        stats = json.loads(caminho_stats.read_text(encoding="utf-8"))
        n_tokens = stats.get("n_tokens_total", 0)
        dentro_da_faixa = args.volume_min_tokens <= n_tokens <= args.volume_max_tokens
        _criterio(
            "volume_de_tokens_na_faixa_alvo",
            dentro_da_faixa,
            f"n_tokens_total={n_tokens} (faixa alvo: {args.volume_min_tokens}-{args.volume_max_tokens}, "
            f"medido de fato em {caminho_stats}, nunca estimado)",
        )
    else:
        _criterio(
            "volume_de_tokens_na_faixa_alvo",
            False,
            f"arquivo de estatisticas nao encontrado: {caminho_stats} (rode 'prepare-dataset' "
            "com o mesmo --saida-base antes de auditar)",
        )

    oficial = all(c["ok"] for c in criterios)
    relatorio = {
        "comando": "audit-tokenizer",
        "criterios": criterios,
        "oficial_do_piloto": oficial,
        "nota": (
            "Criterios objetivos da secao 8 de docs/plano-corpus-piloto.md. oficial_do_piloto=True "
            "exige TODOS os criterios com ok=True simultaneamente -- nenhuma promocao parcial e "
            "permitida. Enquanto oficial_do_piloto=False, o corpus/tokenizador NAO devem alimentar "
            "configs/train-pilot.yaml para um piloto real."
        ),
    }
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    if args.saida:
        Path(args.saida).write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if oficial else 1


def cmd_prepare_dataset(args: argparse.Namespace) -> int:
    from .data.pretrain_prep import construir_sequencias, salvar_dataset_tokenizado
    from .tokenizer.bpe import IgnisTokenizer

    tokenizador = IgnisTokenizer.load(Path(args.tokenizer))
    manifesto = carregar_manifesto(Path(args.manifesto))
    splits = carregar_splits(Path(args.splits))

    resumo = {}
    for nome_split, ids_split in (
        ("treino", splits.treino),
        ("validacao", splits.validacao),
        ("teste", splits.teste),
    ):
        if not ids_split:
            continue
        textos = _carregar_textos_do_split(manifesto, Path(args.corpus_dir), ids_split)
        sequencias, mascaras, estatisticas = construir_sequencias(
            textos, tokenizador, seq_len=args.seq_len
        )
        base = Path(args.saida_base).with_name(f"{Path(args.saida_base).name}_{nome_split}")
        salvar_dataset_tokenizado(base, sequencias, mascaras, estatisticas)
        resumo[nome_split] = estatisticas.__dict__
        print(f"{nome_split}: {estatisticas}")

    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0


def _distribuicao_por_fonte(manifesto: list[EntradaManifesto]) -> dict:
    """Distribuição quantitativa por fonte (seção 3 do Sprint 4): quantos
    documentos de cada fonte caíram em cada status de triagem, e quanto cada
    fonte representa do total ACEITO — para que uma fonte excessivamente
    concentrada (ex.: 90% do corpus aceito vindo de um único site) seja
    visível no relatório, em vez de descoberta só depois do treinamento."""
    por_fonte: dict[str, dict] = {}
    total_aceitos = 0
    for entrada in manifesto:
        registro = por_fonte.setdefault(
            entrada.fonte, {"total": 0, "por_status": {s.value: 0 for s in StatusDocumento}, "caracteres_aceitos": 0}
        )
        registro["total"] += 1
        registro["por_status"][entrada.status.value] += 1
        if entrada.status == StatusDocumento.ACEITO:
            registro["caracteres_aceitos"] += entrada.tamanho_caracteres
            total_aceitos += 1

    # Auditoria de granularidade (Sprint 5): quantas unidades de agrupamento
    # INDEPENDENTES (ver ignis.data.splits) cada fonte realmente contribui.
    # Uma fonte com muitos documentos aceitos mas apenas 1 unidade está, na
    # prática, colapsada — todos os seus documentos sempre cairão no mesmo
    # split, por maior que seja o volume (ver docstring de
    # ignis.data.splits e ignis.data.ingest para a distinção fonte vs.
    # unidade_agrupamento).
    n_unidades_por_fonte = contar_unidades_por_fonte(manifesto)
    for fonte in por_fonte:
        por_fonte[fonte]["n_unidades_agrupamento"] = n_unidades_por_fonte.get(fonte, 0)

    avisos = []
    for fonte, registro in por_fonte.items():
        n_aceitos_fonte = registro["por_status"][StatusDocumento.ACEITO.value]
        proporcao = (n_aceitos_fonte / total_aceitos) if total_aceitos else 0.0
        registro["proporcao_do_total_aceito"] = round(proporcao, 4)
        if total_aceitos > 0 and proporcao > 0.5 and len(por_fonte) > 1:
            avisos.append(
                f"fonte '{fonte}' representa {proporcao:.0%} dos documentos aceitos — "
                "corpus pode estar cientificamente fragil/pouco diverso (ver "
                "docs/plano-corpus-piloto.md, secao 4)."
            )
        elif total_aceitos > 0 and len(por_fonte) == 1:
            avisos.append(
                f"apenas uma fonte ('{fonte}') contribui documentos aceitos — "
                "nenhuma diversidade de fonte no corpus aceito."
            )
        n_unidades_fonte = registro.get("n_unidades_agrupamento", 0)
        if n_aceitos_fonte > 1 and n_unidades_fonte == 1:
            avisos.append(
                f"fonte '{fonte}' tem {n_aceitos_fonte} documento(s) aceito(s) mas forma UMA UNICA "
                "unidade de agrupamento independente — todos os seus documentos sempre cairao no "
                "mesmo split (treino, validacao OU teste), nunca divididos entre eles. Se esta fonte "
                "e um provedor grande (ex.: um dump de Wikipedia), ingira-a em varias subpastas/shards "
                "com 'unidade_agrupamento' distinta por shard em vez de uma pasta unica — ver "
                "docs/plano-corpus-piloto.md, secao sobre granularidade de splits."
            )

    return {
        "n_fontes": len(por_fonte),
        "total_documentos_aceitos": total_aceitos,
        "por_fonte": por_fonte,
        "avisos_concentracao": avisos,
    }


def cmd_report(args: argparse.Namespace) -> int:
    relatorio: dict = {}

    if args.manifesto:
        manifesto = carregar_manifesto(Path(args.manifesto))
        contagens = {status.value: 0 for status in StatusDocumento}
        for entrada in manifesto:
            contagens[entrada.status.value] += 1
        relatorio["manifesto"] = {"total_documentos": len(manifesto), "por_status": contagens}
        relatorio["distribuicao_por_fonte"] = _distribuicao_por_fonte(manifesto)

    if args.splits:
        splits = carregar_splits(Path(args.splits))
        relatorio["splits"] = splits.relatorio()
        relatorio["splits"]["contaminacao"] = verificar_contaminacao(splits)

    if args.tokenizer:
        from .tokenizer.bpe import IgnisTokenizer

        tokenizador = IgnisTokenizer.load(Path(args.tokenizer))
        relatorio["tokenizador"] = {"vocab_size": tokenizador.vocab_size}

    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    if args.saida:
        Path(args.saida).write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nrelatório salvo em: {args.saida}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Ponto de entrada de treinamento do piloto (Fase 3/4).

    Este comando SEMPRE executa primeiro uma verificação "dry-run" completa
    (arquivos existem, formatos/shapes batem, contagem de parâmetros do
    modelo oficial confere, estimativa de memória, um forward de teste SEM
    backward e SEM optimizer.step()). Essa verificação não pode ser pulada,
    nem mesmo quando --iniciar-treinamento-real é passado: o treinamento
    real só começa se TODAS as checagens do dry-run passarem E a flag
    --iniciar-treinamento-real for explicitamente fornecida. Sem essa flag,
    o comando sempre para após o relatório de dry-run, não importa o que
    encontrar. Isto implementa em código a regra da skill
    `ignis-llm-engineer`: nenhum treinamento real sem autorização explícita.
    """
    from .config import IgnisConfig, PilotoConfig

    relatorio: dict = {"comando": "train", "config": args.config, "checagens": []}
    erros: list[str] = []

    def _checar(nome: str, ok: bool, detalhe: str) -> None:
        relatorio["checagens"].append({"nome": nome, "ok": bool(ok), "detalhe": detalhe})
        if not ok:
            erros.append(f"{nome}: {detalhe}")

    try:
        piloto = PilotoConfig.from_yaml(args.config)
        _checar("carregar_config_piloto", True, args.config)
    except Exception as exc:
        _checar("carregar_config_piloto", False, str(exc))
        print(json.dumps(relatorio, ensure_ascii=False, indent=2))
        return 2

    if getattr(args, "device", None) is not None:
        device_original = piloto.execucao.device
        piloto.execucao.device = args.device
        _checar(
            "device_override_cli",
            True,
            f"device sobrescrito via --device: '{device_original}' -> '{piloto.execucao.device}' "
            "(config YAML original preservada em disco, so o objeto em memoria foi alterado)",
        )

    try:
        modelo_cfg = IgnisConfig.from_yaml(piloto.modelo.config)
        _checar("carregar_config_modelo_oficial", True, piloto.modelo.config)
    except Exception as exc:
        _checar("carregar_config_modelo_oficial", False, str(exc))
        print(json.dumps(relatorio, ensure_ascii=False, indent=2))
        return 2

    try:
        piloto.validar_contexto_contra_modelo(modelo_cfg.modelo)
        _checar(
            "contexto_piloto_compativel_com_modelo",
            True,
            f"seq_len={piloto.dados.seq_len} (contexto efetivo {piloto.dados.seq_len - 1}) <= "
            f"max_seq_len={modelo_cfg.modelo.max_seq_len}",
        )
    except ValueError as exc:
        _checar("contexto_piloto_compativel_com_modelo", False, str(exc))

    caminhos_obrigatorios = {
        "manifesto": piloto.dados.manifesto,
        "splits": piloto.dados.splits,
        "tokenizer": piloto.dados.tokenizer,
    }
    for nome, caminho in caminhos_obrigatorios.items():
        existe = Path(caminho).is_file()
        _checar(f"arquivo_existe:{nome}", existe, caminho if existe else f"{caminho} (AUSENTE)")

    base = Path(piloto.dados.dataset_base)
    bases_splits: dict[str, Path] = {}
    for nome_split in ("treino", "validacao", "teste"):
        base_split = base.with_name(f"{base.name}_{nome_split}")
        sufixos = (".sequencias.npy", ".mascaras.npy", ".estatisticas.json")
        todos_existem = all(Path(f"{base_split}{sufixo}").is_file() for sufixo in sufixos)
        obrigatorio = nome_split in ("treino", "validacao")
        if todos_existem:
            bases_splits[nome_split] = base_split
        _checar(
            f"dataset_tokenizado_existe:{nome_split}",
            todos_existem or not obrigatorio,
            f"base={base_split}  presentes={todos_existem}  obrigatorio={obrigatorio}",
        )

    for nome_split, base_split in bases_splits.items():
        try:
            stats = json.loads(Path(f"{base_split}.estatisticas.json").read_text(encoding="utf-8"))
            _checar(
                f"dataset_nao_vazio:{nome_split}",
                stats.get("n_sequencias", 0) > 0,
                f"n_sequencias={stats.get('n_sequencias')}  n_tokens_total={stats.get('n_tokens_total')}",
            )
        except Exception as exc:
            _checar(f"dataset_nao_vazio:{nome_split}", False, str(exc))

    tokenizador = None
    if Path(piloto.dados.tokenizer).is_file():
        try:
            from .tokenizer.bpe import IgnisTokenizer

            tokenizador = IgnisTokenizer.load(Path(piloto.dados.tokenizer))
            vocab_ok = tokenizador.vocab_size == modelo_cfg.modelo.vocab_size
            _checar(
                "tokenizer_vocab_size_bate_com_modelo",
                vocab_ok,
                f"tokenizer.vocab_size={tokenizador.vocab_size}  modelo.vocab_size={modelo_cfg.modelo.vocab_size}",
            )
        except Exception as exc:
            _checar("tokenizer_vocab_size_bate_com_modelo", False, str(exc))

    n_params = None
    modelo = None
    try:
        import torch  # noqa: F401 -- só para detectar ausência de torch explicitamente
        from .model import IgnisModel

        modelo = IgnisModel(modelo_cfg.modelo)
        n_params = modelo.contar_parametros()
        if modelo_cfg.modelo.nome == "ignis-30m":
            esperado = 30_416_384
            _checar("contagem_parametros_confere_com_ignis_30m", n_params == esperado, f"esperado={esperado}  real={n_params}")
        else:
            _checar("contagem_parametros_calculada", True, f"n_params={n_params} (config não-canônica)")

        dispositivo_resolvido = piloto.execucao.device
        if dispositivo_resolvido == "cuda":
            cuda_disponivel = torch.cuda.is_available()
            _checar(
                "device_cuda_disponivel",
                cuda_disponivel,
                (
                    f"torch.cuda.is_available()=True -- device 'cuda' solicitado e disponivel"
                    if cuda_disponivel
                    else (
                        "device 'cuda' solicitado mas torch.cuda.is_available()=False neste "
                        "interpretador (build CPU-only do PyTorch, ou nenhuma GPU CUDA visivel) -- "
                        "o treinamento real falharia ao mover o modelo para a GPU; use --device cpu "
                        "ou prepare o ambiente CUDA antes (ver docs/plano-migracao-cuda.md)"
                    )
                ),
            )
            if cuda_disponivel:
                try:
                    modelo.to(dispositivo_resolvido)
                    _checar("device_modelo_movido", True, f"modelo movido para '{dispositivo_resolvido}' com sucesso")
                except Exception as exc:
                    _checar("device_modelo_movido", False, str(exc))
        else:
            _checar("device_resolvido", True, f"device='{dispositivo_resolvido}' (cpu -- sem checagem adicional necessaria)")
    except ImportError as exc:
        _checar("torch_disponivel", False, f"torch não encontrado neste interpretador: {exc}")
    except Exception as exc:
        _checar("instanciar_modelo_oficial", False, str(exc))

    if n_params is not None:
        bytes_por_param_fp32 = 4 * 4  # pesos + gradiente + 2 momentos do AdamW, fp32
        memoria_estimada_mb = (n_params * bytes_por_param_fp32) / (1024 * 1024)
        relatorio["memoria_estimada"] = {
            "n_parametros": n_params,
            "estimativa_mb_pesos_grad_adamw_fp32": round(memoria_estimada_mb, 1),
            "nota": (
                "Estimativa heurística (pesos + gradiente + 2 momentos do AdamW, fp32); NÃO "
                "inclui ativações/grafo de autograd nem overhead do processo/SO. Ver "
                "scripts/benchmark_training.py para RAM efetivamente medida em execução real."
            ),
        }

    if modelo is not None and "treino" in bases_splits:
        try:
            import torch
            from .training.dataloader import criar_dataloader

            loader = criar_dataloader(
                bases_splits["treino"],
                batch_size=piloto.lote.microbatch_size,
                embaralhar=False,
                seed=piloto.execucao.seed,
                max_seq_len=modelo_cfg.modelo.max_seq_len,
            )
            input_ids, _target_ids = next(iter(loader))
            # Usa o device configurado apenas se o modelo foi de fato movido
            # para la no bloco de checagem acima (cuda disponivel); caso
            # contrario, permanece em cpu para nao mascarar o motivo real
            # da falha de device com um erro de "tensors on different devices".
            dispositivo_forward = (
                piloto.execucao.device
                if (piloto.execucao.device == "cpu" or torch.cuda.is_available())
                else "cpu"
            )
            input_ids = input_ids.to(dispositivo_forward)
            modelo.eval()
            with torch.no_grad():
                logits = modelo(input_ids)
            shape_ok = tuple(logits.shape[:2]) == tuple(input_ids.shape) and logits.shape[2] == modelo_cfg.modelo.vocab_size
            finito = bool(torch.isfinite(logits).all())
            _checar(
                "forward_de_teste_sem_atualizar_pesos",
                shape_ok and finito,
                f"device={dispositivo_forward}  input_ids.shape={tuple(input_ids.shape)}  "
                f"logits.shape={tuple(logits.shape)}  shape_ok={shape_ok}  finito={finito}  "
                "(nenhum backward/optimizer.step() executado)",
            )
        except StopIteration:
            _checar("forward_de_teste_sem_atualizar_pesos", False, "dataloader de treino não produziu nenhum lote")
        except Exception as exc:
            _checar("forward_de_teste_sem_atualizar_pesos", False, str(exc))
    else:
        _checar(
            "forward_de_teste_sem_atualizar_pesos",
            False,
            "pulado: modelo não instanciado ou dataset de treino indisponível",
        )

    relatorio["pronto_para_treinamento_real"] = len(erros) == 0
    relatorio["erros"] = erros
    relatorio["lote_efetivo"] = piloto.lote.lote_efetivo
    relatorio["max_steps"] = piloto.orcamento.max_steps

    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    if args.saida:
        Path(args.saida).write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.iniciar_treinamento_real:
        print(
            "\nDry-run concluído. Nenhum peso foi atualizado e NENHUM treinamento real foi "
            "iniciado. Para iniciar o treinamento real (somente após autorização explícita "
            "separada do responsável pelo projeto), rode novamente com --iniciar-treinamento-real.",
            file=sys.stderr,
        )
        return 0 if relatorio["pronto_para_treinamento_real"] else 1

    if erros:
        print(
            f"\nERRO: {len(erros)} checagem(ns) de dry-run falharam; treinamento real NÃO "
            "será iniciado:\n  - " + "\n  - ".join(erros),
            file=sys.stderr,
        )
        return 1

    print(
        "\nTodas as checagens de dry-run passaram e --iniciar-treinamento-real foi "
        "passado: iniciando treinamento real agora.",
        file=sys.stderr,
    )
    return _executar_treinamento_real(piloto, modelo_cfg, bases_splits, piloto.dados.tokenizer)


def _executar_treinamento_real(piloto, modelo_cfg, bases_splits: dict, tokenizer_caminho: str) -> int:
    """Constrói o Treinador real a partir de `PilotoConfig.para_treinador_config()`
    e chama `Treinador.treinar`. Só é alcançada por `cmd_train` depois que
    TODAS as checagens de dry-run passaram e --iniciar-treinamento-real foi
    passado explicitamente — nunca é chamada por conta própria."""
    from dataclasses import asdict

    import torch  # noqa: F401 -- garante erro claro se torch sumir entre o dry-run e aqui

    from .model import IgnisModel
    from .training.checkpoint import CheckpointMetadata
    from .training.dataloader import criar_dataloader
    from .training.trainer import Treinador

    modelo = IgnisModel(modelo_cfg.modelo)
    treinador_config = piloto.para_treinador_config()

    metadata = CheckpointMetadata.construir(
        config_modelo=modelo_cfg.modelo.model_dump(),
        tokenizer_caminho=tokenizer_caminho,
        dataset_caminho=piloto.dados.splits,
        dataset_identificador=str(bases_splits.get("treino")),
    )
    treinador = Treinador(modelo, treinador_config, metadata_checkpoint=metadata)

    def _criar_loader_treino(epoca: int):
        return criar_dataloader(
            bases_splits["treino"],
            batch_size=piloto.lote.microbatch_size,
            embaralhar=True,
            seed=piloto.execucao.seed,
            epoch=epoca,
            max_seq_len=modelo_cfg.modelo.max_seq_len,
        )

    val_loader = None
    if "validacao" in bases_splits:
        val_loader = criar_dataloader(
            bases_splits["validacao"],
            batch_size=piloto.lote.microbatch_size,
            embaralhar=False,
            max_seq_len=modelo_cfg.modelo.max_seq_len,
        )

    resultado = treinador.treinar(_criar_loader_treino, val_loader)
    print(json.dumps(asdict(resultado), ensure_ascii=False, indent=2))
    return 0


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ignis.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="comando", required=True)

    p = subparsers.add_parser("validate-corpus", help="Ingere, limpa e gera o manifesto de um corpus local")
    p.add_argument("--corpus-dir", required=True, help="Diretório com arquivos .txt/.md/.jsonl")
    p.add_argument("--fonte", default=None, help="Nome da fonte (padrão: nome da subpasta de cada arquivo)")
    p.add_argument("--manifesto-saida", required=True)
    p.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Se --manifesto-saida ja existir, carrega-o e so reprocessa documentos novos ou com "
            "conteudo alterado (por hash SHA-256), em vez de reprocessar o diretorio inteiro do "
            "zero -- permite rodar este comando repetidamente sobre um diretorio que cresce aos "
            "poucos. Sem esta flag, o comportamento e o de sempre: processa tudo do zero."
        ),
    )
    p.add_argument(
        "--limite-novos-documentos",
        type=int,
        default=None,
        help=(
            "Processa no maximo N documentos novos/atualizados nesta chamada; o restante fica para "
            "uma execucao futura (use junto com --incremental para ingerir um diretorio grande em "
            "lotes controlados, sem processar/baixar tudo de uma vez)."
        ),
    )
    p.set_defaults(func=cmd_validate_corpus)

    p = subparsers.add_parser("prepare-splits", help="Gera splits reproduzíveis a partir de um manifesto")
    p.add_argument("--manifesto", required=True)
    p.add_argument("--splits-saida", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--treino", type=float, default=0.9)
    p.add_argument("--validacao", type=float, default=0.05)
    p.add_argument("--teste", type=float, default=0.05)
    p.add_argument(
        "--ignorar-fonte", action="store_true", help="Não agrupar por fonte ao atribuir splits"
    )
    p.set_defaults(func=cmd_prepare_splits)

    p = subparsers.add_parser("train-tokenizer", help="Treina o tokenizador BPE no split de treino")
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--manifesto", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--saida", required=True)
    p.add_argument("--sobrescrever", action="store_true")
    p.set_defaults(func=cmd_train_tokenizer)

    p = subparsers.add_parser("eval-tokenizer", help="Avalia round-trip e eficiência do tokenizador")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--manifesto", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument(
        "--textos-externos",
        default=None,
        help="Arquivo .txt com um texto por linha, NÃO usado em nenhum split, para avaliar overfitting do vocabulário",
    )
    p.add_argument("--saida", default=None)
    p.set_defaults(func=cmd_eval_tokenizer)

    p = subparsers.add_parser(
        "prepare-dataset", help="Converte os splits tokenizados em sequências para predição do próximo token"
    )
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--manifesto", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--saida-base", required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.set_defaults(func=cmd_prepare_dataset)

    p = subparsers.add_parser("report", help="Gera um relatório consolidado")
    p.add_argument("--manifesto", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--saida", default=None)
    p.set_defaults(func=cmd_report)

    p = subparsers.add_parser(
        "audit-tokenizer",
        help="Audita se corpus+tokenizador atendem aos criterios objetivos para serem 'oficiais do piloto'",
    )
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--manifesto", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument(
        "--textos-externos",
        required=True,
        help="Obrigatorio aqui (diferente de eval-tokenizer): arquivo .txt com um texto por linha, nao usado em nenhum split",
    )
    p.add_argument(
        "--dataset-base",
        required=True,
        help="Mesma --saida-base usada em 'prepare-dataset', para ler as estatisticas reais de volume de tokens",
    )
    p.add_argument("--volume-min-tokens", type=int, default=1_000_000)
    p.add_argument("--volume-max-tokens", type=int, default=5_000_000)
    p.add_argument("--saida", default=None)
    p.set_defaults(func=cmd_audit_tokenizer)

    p = subparsers.add_parser(
        "train",
        help="Dry-run obrigatorio do piloto; so treina de verdade com --iniciar-treinamento-real",
    )
    p.add_argument("--config", default="configs/train-pilot.yaml")
    p.add_argument("--saida", default=None, help="Salva o relatório de dry-run em um arquivo JSON")
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help=(
            "Sobrescreve o device configurado em execucao.device do YAML do piloto, sem editar o "
            "arquivo (ex.: testar em --device cpu sem mudar a config pensada para cuda). Se omitido, "
            "usa o device do YAML. Com --device cuda, o dry-run verifica torch.cuda.is_available() "
            "e reporta falha explicita (sem iniciar treinamento real) se a GPU nao estiver disponivel."
        ),
    )
    p.add_argument(
        "--iniciar-treinamento-real",
        action="store_true",
        help=(
            "Sem esta flag, o comando SEMPRE para após o relatório de dry-run (nenhum peso é "
            "atualizado). Com esta flag, inicia o treinamento real APENAS se todas as checagens "
            "de dry-run passarem — nunca use sem autorização explícita separada do responsável "
            "pelo projeto (ver skill ignis-llm-engineer e docs/plano-corpus-piloto.md)."
        ),
    )
    p.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # mensagens de erro claras em vez de traceback cru
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
