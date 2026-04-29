"""
models/chat_template.py
Aplicação padronizada de chat templates para cada modelo.

Cada modelo instruct usa um formato diferente de template.
Este módulo garante que todos os prompts sejam formatados
corretamente e que a posição do último token relevante
para o Logit Lens seja identificada de forma consistente.
"""

from transformers import AutoTokenizer
from typing import Tuple, Dict


# System prompt neutro — sem instrução cultural que possa
# introduzir viés adicional nos experimentos
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions "
    "accurately and concisely."
)


def apply_template(
    tokenizer: AutoTokenizer,
    model_key: str,
    user_message: str
) -> Tuple[str, int]:
    """
    Aplica o chat template correto para cada modelo e retorna
    o prompt formatado e a posição do último token relevante
    para o Logit Lens.

    "Último token relevante" = último token antes do início
    da geração do assistente. É desse ponto que o modelo
    "decide" qual token gerar, logo é o ponto correto para
    aplicar o Logit Lens.

    Usa add_generation_prompt=True para garantir que o template
    inclua o marcador de início do turno do assistente, que é
    exatamente onde a geração começa.

    Args:
        tokenizer:    tokenizer do modelo
        model_key:    chave do modelo (llama, mistral, qwen, gemma)
        user_message: texto da pergunta do usuário (sem template)

    Returns:
        Tupla (prompt_formatado, last_relevant_pos) onde:
        - prompt_formatado: string com template completo aplicado
        - last_relevant_pos: índice do último token (0-based)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ]

    try:
        # add_generation_prompt=True adiciona o início do turno
        # do assistente — ponto exato onde a geração começa
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback para modelos sem chat template definido
        formatted = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User: {user_message}\n"
            f"Assistant:"
        )

    # Tokenizar para identificar a posição do último token
    tokens = tokenizer.encode(formatted, add_special_tokens=False)
    last_relevant_pos = len(tokens) - 1

    return formatted, last_relevant_pos


def verify_templates(
    models_dict: Dict[str, str],
    tokenizers_dict: Dict
):
    """
    Utilitário de verificação — imprime o template formatado
    de cada modelo para inspeção visual antes de rodar
    o experimento completo.

    Deve ser executado via verify_setup.py antes dos experimentos.

    Args:
        models_dict:    dict {model_key: model_name}
        tokenizers_dict: dict {model_key: tokenizer}
    """
    test_prompt = "Quem inventou o avião?"

    print("\n" + "=" * 60)
    print("VERIFICAÇÃO: CHAT TEMPLATES")
    print("=" * 60)

    for model_key, tokenizer in tokenizers_dict.items():

        formatted, last_pos = apply_template(
            tokenizer, model_key, test_prompt
        )
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        last_token_decoded = tokenizer.decode([tokens[last_pos]])

        print(f"\n[{model_key.upper()}]")
        print(f"  Template formatado:")
        print(f"  {repr(formatted[:200])}...")
        print(f"\n  Tokens ({len(tokens)} total):")

        # Mostrar apenas os últimos 8 tokens — mais relevantes
        start = max(0, len(tokens) - 8)
        for i in range(start, len(tokens)):
            marker = " ← LAST (análise aqui)" if i == last_pos else ""
            print(
                f"    [{i:3d}] id={tokens[i]:6d}  "
                f"'{decoded_tokens[i]}'{marker}"
            )

        print(
            f"\n  ✅ Último token relevante: "
            f"pos={last_pos} → '{last_token_decoded}'"
        )
        print("-" * 40)
