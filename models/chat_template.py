from transformers import AutoTokenizer
from typing import Tuple
import torch


# System prompt neutro — sem instrução cultural
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
    Aplica o chat template correto para cada modelo e
    retorna o prompt formatado e a posição do último
    token relevante para o Logit Lens.
    
    A posição retornada é sempre o índice do último token
    ANTES do início da geração — independente do template.
    
    Args:
        tokenizer:    tokenizer do modelo
        model_key:    chave do modelo (llama, mistral, etc.)
        user_message: pergunta do usuário
    
    Returns:
        Tupla (prompt_formatado, last_relevant_pos)
        onde last_relevant_pos é o índice do último token
        a ser analisado no Logit Lens
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ]
    
    # Aplicar template via HuggingFace
    # add_generation_prompt=True adiciona o início do turno
    # do assistente, que é o ponto exato onde a geração começa
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback para modelos sem chat template definido
        formatted = f"{SYSTEM_PROMPT}\n\nUser: {user_message}\nAssistant:"
    
    # Tokenizar para identificar posições
    tokens = tokenizer.encode(formatted, add_special_tokens=False)
    
    # O último token relevante é sempre o último da sequência
    # (seja ele um token de abertura do assistente ou outro)
    # pois é desse ponto que a geração parte
    last_relevant_pos = len(tokens) - 1
    
    return formatted, last_relevant_pos


def verify_templates(models_dict: dict, tokenizers_dict: dict):
    """
    Utilitário de verificação — imprime o template formatado
    de cada modelo para inspeção visual antes de rodar
    o experimento completo.
    """
    test_prompt = "Quem inventou o avião?"
    
    print("\n" + "="*60)
    print("VERIFICAÇÃO DE CHAT TEMPLATES")
    print("="*60)
    
    for model_key, tokenizer in tokenizers_dict.items():
        formatted, last_pos = apply_template(
            tokenizer, model_key, test_prompt
        )
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        
        print(f"\n[{model_key.upper()}]")
        print(f"Template formatado:")
        print(formatted)
        print(f"\nNúmero de tokens: {len(tokens)}")
        print(f"Último token relevante: pos={last_pos} → '{tokenizer.decode([tokens[last_pos]])}'")
        print("-"*40)