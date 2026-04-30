from transformers import AutoTokenizer
from typing import Tuple, Dict


def apply_template(
    tokenizer: AutoTokenizer,
    model_key: str,
    user_message: str
) -> Tuple[str, int]:
    """
    Aplica o chat template sem system prompt.

    Apenas a mensagem do usuario e incluida, garantindo que
    o unico contexto linguistico seja o idioma da pergunta.

    Args:
        tokenizer:    tokenizer do modelo
        model_key:    chave do modelo (llama, mistral, qwen, gemma)
        user_message: texto da pergunta do usuario

    Returns:
        Tupla (prompt_formatado, last_relevant_pos)
    """
    messages = [
        {"role": "user", "content": user_message}
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback para modelos sem chat template definido
        formatted = f"User: {user_message}\nAssistant:"

    tokens = tokenizer.encode(formatted, add_special_tokens=False)
    last_relevant_pos = len(tokens) - 1

    return formatted, last_relevant_pos


def verify_templates(
    models_dict: Dict[str, str],
    tokenizers_dict: Dict
):
    """
    Verifica o chat template de cada modelo apos a remocao
    do system prompt.
    """
    test_prompts = {
        "pt": "Quem inventou o aviao?",
        "en": "Who invented the airplane?",
        "de": "Wer hat das Flugzeug erfunden?",
        "it": "Chi ha inventato l'aereo?"
    }

    print("\n" + "=" * 60)
    print("VERIFICACAO: CHAT TEMPLATES (sem system prompt)")
    print("=" * 60)

    for model_key, tokenizer in tokenizers_dict.items():
        print(f"\n[{model_key.upper()}]")

        for lang, prompt in test_prompts.items():
            formatted, last_pos = apply_template(
                tokenizer, model_key, prompt
            )
            tokens        = tokenizer.encode(formatted, add_special_tokens=False)
            last_token    = tokenizer.decode([tokens[last_pos]])
            n_tokens      = len(tokens)

            print(f"  [{lang}] {n_tokens} tokens | "
                  f"last='{last_token}' | "
                  f"prompt: {repr(formatted[-80:])}")

        print("-" * 40)