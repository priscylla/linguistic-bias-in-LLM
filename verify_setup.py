"""
verify_setup.py
Verificacao do setup ANTES de rodar os experimentos completos.

Carrega apenas tokenizers (~30s) para verificar:
    0. Token HuggingFace — acesso a modelos restritos
    1. Completion prompts — ultimo token de cada prompt (reais e neutros)
    3. Tokenizacao de nomes e indicadores apos cada prompt
       (split, compartilhado entre candidatos, palavra comum)

A configuracao (config/*.yaml) e validada antes de tudo.
Saidas: verification_report.json (resumo) e
        reports/tokenization/{modelo}.csv (tabela completa).
Executar sempre antes de rodar modelos.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from config.experiment_config import ExperimentConfig
from models.model_loader import check_repo_access, find_hf_token, load_tokenizer
from config.loader import require_valid
from data.tokenization_check import (
    check_tokenizer, dropped_words, elision_check, indicator_cells, prefix_forced, write_summary,
)
from models.chat_template import ChatFormatter

TOKENIZATION_DIR = Path("reports") / "tokenization"


def check_token_access(config: ExperimentConfig):
    """
    Verifica disponibilidade do token HuggingFace e
    acesso a cada modelo do experimento.
    """
    print("=" * 60)
    print("VERIFICACAO 0: TOKEN HUGGING FACE")
    print("=" * 60)

    token, source = find_hf_token()
    gated = [k for k, m in config.registry.models.items() if m.gated]

    if token is None:
        print("  Token nao encontrado")
        print(f"  Modelos restritos (precisam de token): {', '.join(gated)}")
        print("\n  Para configurar:")
        print("  1. hf auth login            (= huggingface-cli login)")
        print("  2. export HF_TOKEN='hf_seu_token'")
        return False

    print(f"  Token encontrado: {token[:8]}...  ({source})")
    all_ok = True
    # HEAD de config.json no commit fixo: acesso (licença aceita) e commit, sem download
    for model_key, spec in config.registry.models.items():
        if not spec.gated:
            print(f"  OK  {model_key}: publico (sem token necessario)")
            continue
        ok, detail = check_repo_access(spec.id, spec.revision, token)
        print(f"  {'OK ' if ok else 'ERR'} {model_key}: {detail}")
        all_ok &= ok

    return all_ok


def check_completion_prompts(tokenizers: dict, config: ExperimentConfig):
    """
    Ultimo token de cada prompt completion (reais e neutros da linha
    de base): confirma que a lacuna termina na palavra esperada e nao
    num subtoken inesperado.
    """
    print("\n" + "=" * 60)
    print("VERIFICACAO 1: COMPLETION PROMPTS")
    print("=" * 60)

    prompts = [p for p in list(config.prompt_specs) + list(config.baseline_spec.prompts)
               if p.format == "completion"]
    report = {}
    for model_key, tokenizer in tokenizers.items():
        print(f"\n[{model_key.upper()}]")
        report[model_key] = {}
        for p in prompts:
            ids  = tokenizer.encode(p.text, add_special_tokens=False)
            last = tokenizer.decode([ids[-1]])
            key  = f"{p.fact or 'baseline'}/{p.lang}/{p.type}.{p.paraphrase}"
            report[model_key][key] = {"prompt": p.text, "n_tokens": len(ids),
                                      "last_token": last}
            print(f"  [{key}]  n={len(ids):2d}  last='{last:12s}'  \"{p.text}\"")
    return report


def check_name_tokenization(tokenizers: dict, config: ExperimentConfig):
    """
    Verificacao 3: como cada nome e indicador tokeniza apos cada prompt.
    QA so para tokenizers com chat template (modelos instruct).
    """
    print("\n" + "=" * 60)
    print("VERIFICACAO 3: TOKENIZACAO DE NOMES E INDICADORES")
    print("=" * 60)

    TOKENIZATION_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for model_key, tokenizer in tokenizers.items():
        chat = None
        # QA só em modelos instruct (kind do registro)
        if config.model(model_key).kind == "instruct":
            chat = ChatFormatter.for_tokenizer(tokenizer, model_key)

        res = check_tokenizer(
            model_key, tokenizer, config.fact_specs, config.prompt_specs,
            zipf_threshold=config.common_word_zipf, chat_formatter=chat)

        csv_path = TOKENIZATION_DIR / f"{model_key}.csv"
        pd.DataFrame(res["rows"]).to_csv(csv_path, index=False)
        # Palavras que não contam e células de indicador (regra dos dois candidatos)
        dropped = dropped_words(res["rows"])
        cells = indicator_cells(res["rows"])
        pd.DataFrame(dropped).to_csv(TOKENIZATION_DIR / f"{model_key}_dropped.csv", index=False)
        forced = prefix_forced(res["rows"])
        pd.DataFrame(forced).to_csv(TOKENIZATION_DIR / f"{model_key}_prefix_forced.csv", index=False)
        res["summary"]["n_prefix_forced"] = len(forced)
        elis = elision_check(model_key, tokenizer, config.fact_specs, config.prompt_specs,
                             config.forcing_spec)
        pd.DataFrame(elis).to_csv(TOKENIZATION_DIR / f"{model_key}_elision.csv", index=False)
        res["summary"]["elision"] = elis
        pd.DataFrame(cells).to_csv(TOKENIZATION_DIR / f"{model_key}_indicator_cells.csv", index=False)
        res["summary"]["n_dropped"] = len(dropped)
        res["summary"]["skipped_indicator_cells"] = [
            f"{c['fact']}/{c['lang']}/{c['prompt_type']}" for c in cells if c["status"] == "skipped"]
        summaries[model_key] = {**res["summary"], "csv": str(csv_path),
                                "qa_checked": chat is not None}

        c = res["summary"]["counts"]
        print(f"\n[{model_key.upper()}] {res['summary']['n_rows']} linhas -> {csv_path}")
        print(f"  split={c.get('split', 0)}  shared={c.get('shared', 0)}  "
              f"common={c.get('common', 0)}  excluded={c.get('excluded', 0)}"
              + ("" if chat else "  (QA nao verificado: modelo base)"))
        if res["summary"]["skipped_indicator_cells"]:
            print("  células de indicador puladas: " + ", ".join(res["summary"]["skipped_indicator_cells"]))
        for issue in res["summary"]["issues"]:
            if issue["flag"] in ("shared", "excluded"):
                print(f"  {issue['flag']:8s} [{issue['fact']}][{issue['lang']}] "
                      f"{issue['candidate']}: {issue['word']!r} -> {issue['token']!r}"
                      + (f" (com {issue['shared_with']})" if issue["shared_with"] else ""))

    write_summary(TOKENIZATION_DIR, Path("reports") / "tokenization_summary.md", list(config.models),
                  config.fact_specs, config.prompt_specs)
    print("\n  Resumo: reports/tokenization_summary.md")
    return summaries


def main():
    # Configuracao consistente antes de carregar qualquer tokenizer
    require_valid()
    config = ExperimentConfig()

    # ── Verificacao 0: Token ──
    token_ok = check_token_access(config)

    # ── Carregar tokenizers ──
    print("\n\nCarregando tokenizers...")
    tokenizers = {}
    for model_key, model_name in config.models.items():
        print(f"  {model_key}...")
        try:
            tokenizers[model_key] = load_tokenizer(model_key, config.registry)
        except Exception as e:
            print(f"  ERRO ao carregar tokenizer de {model_key}: {e}")
    print(f"  {len(tokenizers)} tokenizers carregados")

    # ── Verificacao 1: Completion Prompts ──
    prompt_report = check_completion_prompts(tokenizers, config)

    # ── Verificacao 3: Tokenizacao de nomes e indicadores ──
    tokenization = check_name_tokenization(tokenizers, config)

    # ── Relatorio Final ──
    print("\n\n" + "=" * 60)
    print("RELATORIO FINAL")
    print("=" * 60)

    excluded = {mk: t["counts"].get("excluded", 0) for mk, t in tokenization.items()}
    problems = [mk for mk, n in excluded.items() if n]
    if not problems:
        print("  Nenhuma palavra excluida por boundary merge.")
    else:
        for mk in problems:
            print(f"  [{mk}] {excluded[mk]} palavra(s) excluida(s): ver {tokenization[mk]['csv']}")
    print("\n  Proximo passo (com o config confirmado):")
    print("  python run_lens.py --model mistral --sanity-only")

    # Salvar relatorio
    report = {
        "token_ok": token_ok,
        "prompts":  prompt_report,
        "excluded": excluded,
        "tokenization": tokenization,
    }
    with open("verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n  Relatorio salvo: verification_report.json")

    return len(problems) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)