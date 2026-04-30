"""
verify_setup.py
Verificação do setup ANTES de rodar os experimentos completos.

Carrega apenas tokenizers (muito mais rápido que carregar modelos)
para verificar:
    1. Token HuggingFace — acesso a modelos restritos
    2. Chat templates    — formatação correta por modelo
    3. Keyword finding   — localização de keywords nos tokens

Executar sempre antes de run_single_model.py.
Tempo estimado: ~30-60 segundos.
"""

import json
import sys
from transformers import AutoTokenizer
from huggingface_hub import HfApi

from config.experiment_config import ExperimentConfig
from models.model_loader import get_hf_token, requires_token
from models.chat_template import apply_template
from data.keyword_finder import find_keyword_position_robust
from data.prompts import PROMPTS


def check_token_access(config: ExperimentConfig):
    """
    Verifica disponibilidade do token HuggingFace e
    acesso a cada modelo do experimento.
    """
    print("=" * 60)
    print("VERIFICAÇÃO 0: TOKEN HUGGING FACE")
    print("=" * 60)

    token = get_hf_token()

    if token is None:
        print("⚠️  Token não encontrado")
        print("   Modelos públicos (ok sem token): mistral, qwen")
        print("   Modelos restritos (precisam de token): llama, gemma")
        print("\n   Para configurar:")
        print("   1. export HF_TOKEN='hf_seu_token'")
        print("   2. huggingface-cli login")
        print("   3. Criar .env com HF_TOKEN=hf_seu_token")
        return False

    print(f"✅ Token encontrado: {token[:8]}...")

    api = HfApi(token=token)
    all_ok = True

    for model_key, model_name in config.models.items():
        if requires_token(model_name):
            try:
                api.model_info(model_name)
                print(f"  ✅ {model_key}: acesso confirmado")
            except Exception as e:
                print(f"  ❌ {model_key}: sem acesso")
                print(f"     → Aceitar licença: https://huggingface.co/{model_name}")
                all_ok = False
        else:
            print(f"  ✅ {model_key}: público (sem necessidade de token)")

    return all_ok


def check_chat_templates(tokenizers: dict):
    """
    Verifica o chat template de cada modelo com um prompt de teste.
    """
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO 1: CHAT TEMPLATES")
    print("=" * 60)

    test_prompt = "Quem inventou o avião?"
    report = {}

    for model_key, tokenizer in tokenizers.items():

        formatted, last_pos = apply_template(
            tokenizer, model_key, test_prompt
        )
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in tokens]
        last_token = tokenizer.decode([tokens[last_pos]])

        print(f"\n[{model_key.upper()}]")
        print(f"  Tokens totais: {len(tokens)}")

        # Mostrar os últimos 6 tokens
        start = max(0, len(tokens) - 6)
        for i in range(start, len(tokens)):
            marker = " ← LAST" if i == last_pos else ""
            print(f"    [{i:3d}] '{decoded[i]}'{marker}")

        print(f"  ✅ Último token: pos={last_pos} → '{last_token}'")

        report[model_key] = {
            "n_tokens":  len(tokens),
            "last_pos":  last_pos,
            "last_token": last_token
        }

    return report


def check_keyword_finding(tokenizers: dict, config: ExperimentConfig):
    """
    Verifica a localização de keywords em todos os modelos,
    fatos e idiomas.
    """
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO 2: KEYWORD FINDING")
    print("=" * 60)

    problems = []

    for model_key, tokenizer in tokenizers.items():
        print(f"\n[{model_key.upper()}]")

        for fact in ["aviao", "telefone"]:
            for lang in ["pt", "en", "de", "it"]:

                keyword   = config.keyword_tokens[fact][lang]
                prompt    = PROMPTS[fact][lang]["F1"]

                formatted, _ = apply_template(tokenizer, model_key, prompt)
                input_ids = tokenizer.encode(
                    formatted, add_special_tokens=False
                )

                pos, strategy = find_keyword_position_robust(
                    tokenizer, input_ids, keyword, model_key
                )

                if pos is not None:
                    found_token = tokenizer.decode([input_ids[pos]])
                    status = "✅" if strategy == "exact" else "⚠️ "
                else:
                    found_token = "NOT FOUND"
                    status = "❌"
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

    # ── Verificação 0: Token ──
    token_ok = check_token_access(config)

    # ── Carregar tokenizers ──
    print("\n\nCarregando tokenizers...")
    tokenizers = {}
    for model_key, model_name in config.models.items():
        print(f"  {model_key}...")
        try:
            from models.model_loader import get_hf_token
            token = get_hf_token()
            tokenizers[model_key] = AutoTokenizer.from_pretrained(
                model_name, token=token, trust_remote_code=True
            )
        except Exception as e:
            print(f"  ❌ Erro ao carregar tokenizer de {model_key}: {e}")
    print(f"✅ {len(tokenizers)} tokenizers carregados")

    # ── Verificação 1: Chat Templates ──
    template_report = check_chat_templates(tokenizers)

    # ── Verificação 2: Keyword Finding ──
    problems = check_keyword_finding(tokenizers, config)

    # ── Relatório Final ──
    print("\n\n" + "=" * 60)
    print("RELATÓRIO FINAL")
    print("=" * 60)

    if not problems:
        print("✅ Nenhum problema encontrado!")
        print("\n   Próximo passo:")
        print("   python run_single_model.py mistral")
    else:
        print(f"❌ {len(problems)} problema(s) encontrado(s):\n")
        for p in problems:
            print(
                f"  → [{p['model']}][{p['fact']}][{p['lang']}] "
                f"keyword='{p['keyword']}'"
            )
        print("\n   Correção: editar KEYWORD_VARIANTS em data/keyword_finder.py")
        print("   Depois:   python verify_setup.py  (repetir até ✅)")

    # Salvar relatório
    report = {
        "token_ok":  token_ok,
        "templates": template_report,
        "problems":  problems
    }
    with open("verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n✅ Relatório salvo: verification_report.json")

    return len(problems) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)