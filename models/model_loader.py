"""
models/model_loader.py
Carrega modelo e tokenizer a partir do registro (config/models.yaml):
ID, commit fixo, precisão, quantização e token só para modelos gated.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Tuple
from dotenv import load_dotenv

from config.loader import ModelSpec, Registry, load_models


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


# Precisões aceitas no config (string -> dtype do torch)
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def resolve_dtype(dtype: str) -> torch.dtype:
    """
    Converte o nome da precisão do config em torch.dtype.

    bfloat16 é o padrão: o Gemma 2 se comporta mal em float16
    (overflow de ativações), e bf16 tem o mesmo custo de VRAM.
    """
    if dtype not in DTYPES:
        raise ValueError(
            f"dtype '{dtype}' inválido. Use um de: {list(DTYPES)}"
        )
    return DTYPES[dtype]


def quantization_config(spec: ModelSpec, registry: Registry) -> Optional[BitsAndBytesConfig]:
    """BitsAndBytesConfig do registro (ex.: nf4), ou None."""
    if spec.quantization is None:
        return None
    q = dict(registry.quantization[spec.quantization])
    if "bnb_4bit_compute_dtype" in q:
        q["bnb_4bit_compute_dtype"] = resolve_dtype(q["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**q)


class ArchitectureMismatch(RuntimeError):
    """O modelo carregado não é da arquitetura declarada no registro."""


def _token_for(spec: ModelSpec):
    token = get_hf_token()
    if spec.gated and token is None:
        raise EnvironmentError(
            f"\n❌ Token do Hugging Face necessário para '{spec.id}' (gated)\n"
            f"   Opções:\n"
            f"   1. export HF_TOKEN='hf_seu_token'\n"
            f"   2. huggingface-cli login\n"
            f"   3. Criar arquivo .env com HF_TOKEN=hf_seu_token\n"
            f"\n   Obter token:     https://huggingface.co/settings/tokens\n"
            f"   Aceitar licença: https://huggingface.co/{spec.id}"
        )
    return token


def load_tokenizer(model_key: str, registry: Optional[Registry] = None):
    """Só o tokenizer, no commit do registro (verify_setup)."""
    spec = (registry or load_models())[model_key]
    tok = AutoTokenizer.from_pretrained(spec.id, revision=spec.revision,
                                        token=_token_for(spec) if spec.gated else get_hf_token())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_key: str,
    registry: Optional[Registry] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Carrega modelo e tokenizer de uma entrada do registro.

    Configurações aplicadas:
        - revision fixa (commit do registro), modelo e tokenizer
        - dtype da entrada (bfloat16 em todos: o Gemma 2 falha em fp16)
        - quantização da entrada (gemma27: NF4, lm_head sem quantizar)
        - device_map="auto": distribui entre GPUs disponíveis
        - output_hidden_states=True (obrigatório para o Logit Lens)
        - sem trust_remote_code: todas as arquiteturas são nativas do
          transformers

    Raises:
        EnvironmentError:     modelo gated sem token
        ArchitectureMismatch: model_type carregado != architecture do registro
    """
    registry = registry or load_models()
    spec     = registry[model_key]
    quant    = quantization_config(spec, registry)
    print(f"\nCarregando {model_key}: {spec.id}@{spec.revision[:10]} "
          f"({spec.precision}{', ' + spec.quantization if spec.quantization else ''})")

    token = _token_for(spec) if spec.gated else get_hf_token()
    if spec.gated:
        print(f"  🔑 Token HF: {token[:8]}...")

    tokenizer = AutoTokenizer.from_pretrained(spec.id, revision=spec.revision, token=token)
    # Garantir que o tokenizer tem padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(revision=spec.revision, token=token, dtype=resolve_dtype(spec.precision),
                  device_map="auto", output_hidden_states=True)
    if quant is not None:
        kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(spec.id, **kwargs)
    model.eval()  # desativa dropout — importante para reprodutibilidade

    if model.config.model_type != spec.architecture:
        raise ArchitectureMismatch(
            f"{model_key}: model_type '{model.config.model_type}' carregado, "
            f"registro diz '{spec.architecture}'")

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✅ {n_params:.1f}B params | {model.config.num_hidden_layers} camadas | "
          f"hidden_size={model.config.hidden_size}")
    return model, tokenizer


def get_num_layers(model: AutoModelForCausalLM) -> int:
    """Retorna número de camadas transformer do modelo."""
    return model.config.num_hidden_layers


def get_hidden_size(model: AutoModelForCausalLM) -> int:
    """Retorna dimensão do hidden state."""
    return model.config.hidden_size
