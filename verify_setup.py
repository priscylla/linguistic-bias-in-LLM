"""
verify_setup.py
Verificacao do setup ANTES de rodar os experimentos completos.

Carrega apenas tokenizers (~30s) para verificar:
    0. Token HuggingFace — acesso a modelos restritos
    1. Completion prompts — ultimo token correto por formulacao
    2. Keyword finding — localizacao de keywords nos tokens

Nao verifica mais chat templates — removidos do pipeline.
Executar sempre antes de run_single_model.py.
"""

import json
import sys
from transformers import AutoTokenizer
from huggingface_hub import HfApi

from config.experiment_config import ExperimentConfig
from models.model_loader import get_hf_token, requires_token
from data.keyword_finder import find_keyword_position_robust
from data.prompts import COMPLETION_PROMPTS


def check_token_access(config: ExperimentConfig):
    """
    Verifica disponibilidade do token HuggingFace e
    acesso a cada modelo do experimento.
    """
    print("=" * 60)
    print("VERIFICACAO 0: TOKEN HUGGING FACE")
    print("=" * 60)

    token = get_hf_token()

    if token is None:
        print("  Token nao encontrado")
        print("  Modelos publicos (ok sem token): mistral, qwen")
        print("  Modelos restritos (precisam de token): llama, gemma")
        print("\n  Para configurar:")
        print("  1. export HF_TOKEN='hf_seu_token'")
        print("  2. huggingface-cli login")
        print("  3. Criar .env com HF_TOKEN=hf_seu_token")
        return False

    print(f"  Token encontrado: {token[:8]}...")

    api = HfApi(token=token)
    all_ok = True

    for model_key, model_name in config.models.items():
        if requires_token(model_name):
            try:
                api.model_info(model_name)
                print(f"  OK  {model_key}: acesso confirmado")
            except Exception:
                print(f"  ERR {model_key}: sem acesso")
                print(f"      Aceitar licenca: https://huggingface.co/{model_name}")
                all_ok = False
        else:
            print(f"  OK  {model_key}: publico (sem token necessario)")

    return all_ok


def check_completion_prompts(tokenizers: dict):
    """
    Verifica os completion prompts — confirma que o ultimo token
    de cada prompt e a preposicao/verbo esperado, nao um subtoken
    inesperado.

    Ultimo token esperado por formulacao:
        F1: "por" / "by" / "von" / "da"
        F2: "considerado" / "be" / "als" / "considerato"
        F3: "e" / "was" / "war" / "stato"
    """
    print("\n" + "=" * 60)
    print("VERIFICACAO 1: COMPLETION PROMPTS")
    print("=" * 60)

    facts = ["aviao", "telefone"]
    langs = ["pt", "en", "de", "it"]
    forms = ["F1", "F2", "F3"]

    report = {}

    for model_key, tokenizer in tokenizers.items():
        print(f"\n[{model_key.upper()}]")
        report[model_key] = {}

        for fact in facts:
            for lang in langs:
                for form in forms:
                    prompt = COMPLETION_PROMPTS[fact][lang][form]
                    ids    = tokenizer.encode(prompt, add_special_tokens=False)
                    last_token = tokenizer.decode([ids[-1]])
                    n_tokens   = len(ids)

                    report[model_key][f"{fact}/{lang}/{form}"] = {
                        "prompt":     prompt,
                        "n_tokens":   n_tokens,
                        "last_token": last_token
                    }

                    print(
                        f"  [{fact}][{lang}][{form}]  "
                        f"n={n_tokens:2d}  "
                        f"last='{last_token:12s}'  "
                        f"\"{prompt}\""
                    )

    return report


def check_keyword_finding(tokenizers: dict, config: ExperimentConfig):
    """
    Verifica a localizacao de keywords nos completion prompts.
    Usa F1 como prompt de teste — suficiente para verificar
    que a keyword e encontrada corretamente.
    """
    print("\n" + "=" * 60)
    print("VERIFICACAO 2: KEYWORD FINDING")
    print("=" * 60)

    problems = []

    for model_key, tokenizer in tokenizers.items():
        print(f"\n[{model_key.upper()}]")

        for fact in ["aviao", "telefone"]:
            for lang in ["pt", "en", "de", "it"]:

                keyword = config.keyword_tokens[fact][lang]

                # Usar completion prompt F1 diretamente
                # sem chat template — igual ao run_single_model.py
                prompt  = COMPLETION_PROMPTS[fact][lang]["F1"]
                input_ids = tokenizer.encode(
                    prompt, add_special_tokens=False
                )

                pos, strategy = find_keyword_position_robust(
                    tokenizer, input_ids, keyword, model_key
                )

                if pos is not None:
                    found_token = tokenizer.decode([input_ids[pos]])
                    status = "OK " if strategy == "exact" else "OK*"
                else:
                    found_token = "NOT FOUND"
                    status = "ERR"
                    problems.append({
                        "model": model_key, "fact": fact,
                        "lang": lang, "keyword": keyword
                    })

                print(
                    f"  {status} [{fact}][{lang}]  "
                    f"kw='{keyword:12s}'  "
                    f"pos={str(pos):4s}  "
                    f"token='{found_token:12s}'  "
                    f"{strategy}"
                )

    return problems


def main():
    config = ExperimentConfig()

    # ── Verificacao 0: Token ──
    token_ok = check_token_access(config)

    # ── Carregar tokenizers ──
    print("\n\nCarregando tokenizers...")
    tokenizers = {}
    for model_key, model_name in config.models.items():
        print(f"  {model_key}...")
        try:
            token = get_hf_token()
            tokenizers[model_key] = AutoTokenizer.from_pretrained(
                model_name, token=token, trust_remote_code=True
            )
        except Exception as e:
            print(f"  ERRO ao carregar tokenizer de {model_key}: {e}")
    print(f"  {len(tokenizers)} tokenizers carregados")

    # ── Verificacao 1: Completion Prompts ──
    prompt_report = check_completion_prompts(tokenizers)

    # ── Verificacao 2: Keyword Finding ──
    problems = check_keyword_finding(tokenizers, config)

    # ── Relatorio Final ──
    print("\n\n" + "=" * 60)
    print("RELATORIO FINAL")
    print("=" * 60)

    if not problems:
        print("  Nenhum problema encontrado!")
        print("\n  Proximo passo:")
        print("  python run_single_model.py mistral")
    else:
        print(f"  {len(problems)} problema(s) encontrado(s):\n")
        for p in problems:
            print(
                f"  [{p['model']}][{p['fact']}][{p['lang']}] "
                f"keyword='{p['keyword']}'"
            )
        print("\n  Correcao: editar KEYWORD_VARIANTS em data/keyword_finder.py")
        print("  Depois:   python verify_setup.py")

    # Salvar relatorio
    report = {
        "token_ok": token_ok,
        "prompts":  prompt_report,
        "problems": problems
    }
    with open("verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n  Relatorio salvo: verification_report.json")

    return len(problems) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)