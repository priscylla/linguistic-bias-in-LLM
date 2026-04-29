"""
models/model_loader.py
Carregamento de modelos com suporte a token HuggingFace
para modelos restritos (LLaMA, Gemma).
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple
from dotenv import load_dotenv


# Prefixos de modelos que exigem autenticação HuggingFace
RESTRICTED_MODELS = [
    "meta-llama",
    "google/gemma"
]


def get_hf_token() -> str | None:
    """
    Recupera o token do Hugging Face de forma segura.

    Prioridade:
        1. Variável de ambiente HF_TOKEN (export HF_TOKEN=hf_...)
        2. Arquivo .env na raiz do projeto (HF_TOKEN=hf_...)
        3. Cache do huggingface-cli login (~/.cache/huggingface/token)

    Returns:
        Token como string, ou None se não encontrado.
    """
    # Tenta carregar .env se existir — popula os.environ automaticamente
    load_dotenv()

    # Após load_dotenv, os.environ já inclui variáveis do .env
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Fallback: cache do huggingface-cli login
    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            token = f.read().strip()
            return token if token else None

    return None


def requires_token(model_name: str) -> bool:
    """Verifica se o modelo requer autenticação HuggingFace."""
    return any(prefix in model_name for prefix in RESTRICTED_MODELS)


def load_model(
    model_name: str,
    model_key: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Carrega modelo e tokenizer com configurações otimizadas
    para extração de ativações via Logit Lens.

    Configurações aplicadas:
        - torch_dtype=float16: reduz uso de VRAM pela metade
        - device_map="auto":   distribui entre GPUs disponíveis
        - output_hidden_states=True: habilita saída dos hidden states
                                     (obrigatório para o Logit Lens)

    Args:
        model_name: path/nome no HuggingFace Hub
        model_key:  chave curta (llama, mistral, qwen, gemma)

    Returns:
        Tupla (model, tokenizer)

    Raises:
        EnvironmentError: se o modelo requer token e nenhum foi encontrado
    """
    print(f"\nCarregando {model_key}: {model_name}")

    token = get_hf_token()

    if requires_token(model_name):
        if token is None:
            raise EnvironmentError(
                f"\n❌ Token do Hugging Face necessário para '{model_name}'\n"
                f"   Opções:\n"
                f"   1. export HF_TOKEN='hf_seu_token'\n"
                f"   2. huggingface-cli login\n"
                f"   3. Criar arquivo .env com HF_TOKEN=hf_seu_token\n"
                f"\n   Obter token:     https://huggingface.co/settings/tokens\n"
                f"   Aceitar licença: https://huggingface.co/{model_name}"
            )
        print(f"  🔑 Token HF: {token[:8]}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True
    )

    # Garantir que o tokenizer tem padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True    # CRÍTICO para o Logit Lens
    )

    model.eval()  # desativa dropout — importante para reprodutibilidade

    n_params  = sum(p.numel() for p in model.parameters()) / 1e9
    n_layers  = model.config.num_hidden_layers
    hid_size  = model.config.hidden_size

    print(f"  ✅ {n_params:.1f}B params | {n_layers} camadas | hidden_size={hid_size}")

    return model, tokenizer


def get_num_layers(model: AutoModelForCausalLM) -> int:
    """Retorna número de camadas transformer do modelo."""
    return model.config.num_hidden_layers


def get_hidden_size(model: AutoModelForCausalLM) -> int:
    """Retorna dimensão do hidden state (ex: 5120 para LLaMA 13B)."""
    return model.config.hidden_size
