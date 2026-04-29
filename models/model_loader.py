import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple


def get_hf_token() -> str | None:
    """
    Recupera o token do Hugging Face de forma segura.
    Tenta múltiplas fontes em ordem de prioridade.
    """
    # Opção 1: variável de ambiente
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    
    # Opção 2: arquivo .env
    env_path = os.path.join(
        os.path.dirname(__file__), "..", ".env"
    )
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    return token
    
    # Opção 3: cache do huggingface-cli
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/token"
    )
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            token = f.read().strip()
            return token if token else None
    
    return None


# Modelos que requerem token
RESTRICTED_MODELS = [
    "meta-llama",
    "google/gemma"
]


def requires_token(model_name: str) -> bool:
    """Verifica se o modelo requer autenticação."""
    return any(
        prefix in model_name
        for prefix in RESTRICTED_MODELS
    )


def load_model(
    model_name: str,
    model_key: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    print(f"\nCarregando {model_key}: {model_name}")
    
    # Verificar token se necessário
    token = get_hf_token()
    
    if requires_token(model_name):
        if token is None:
            raise EnvironmentError(
                f"\nToken do Hugging Face necessário para '{model_name}'\n"
                f"   Opções:\n"
                f"   1. export HF_TOKEN='hf_seu_token'\n"
                f"   2. huggingface-cli login\n"
                f"   3. Criar arquivo .env com HF_TOKEN=hf_seu_token\n"
                f"\n   Obter token em: https://huggingface.co/settings/tokens\n"
                f"   Aceitar licença: https://huggingface.co/{model_name}"
            )
        print(f"  🔑 Usando token HF: {token[:8]}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True
    )
    
    model.eval()
    
    print(f"  ✅ Carregado: "
          f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params  "
          f"{model.config.num_hidden_layers} camadas  "
          f"hidden_size={model.config.hidden_size}")
    
    return model, tokenizer