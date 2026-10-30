"""
models/model_loader.py
Carrega modelo e tokenizer a partir do registro (config/models.yaml):
ID, commit fixo, precisão, quantização e token só para modelos gated.
"""

import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Tuple
from dotenv import dotenv_values

from config.loader import ModelSpec, Registry, load_models


# Token real: "hf_" + letras e dígitos (hoje 34). Placeholders como
# "hf_seu_token", "hf_tok…" ou "hf_xxxx…" não passam
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{30,}")
_PLACEHOLDER_WORDS = ("yourtoken", "tokenhere", "placeholder", "example")
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
_warned = set()


def looks_like_token(token) -> bool:
    """True se `token` tem a forma de um token real do Hugging Face."""
    if not token:
        return False
    t = token.strip()
    if not _TOKEN_RE.fullmatch(t):
        return False
    rest = t[3:].lower()
    return len(set(rest)) >= 8 and not any(w in rest for w in _PLACEHOLDER_WORDS)


def _warn_once(msg: str):
    if msg not in _warned:
        _warned.add(msg)
        print(f"  ⚠️  {msg}", file=sys.stderr)


def _token_files():
    """Onde `hf auth login` (= huggingface-cli login) grava o token."""
    paths = []
    try:
        from huggingface_hub import constants
        paths.append(Path(constants.HF_TOKEN_PATH))          # $HF_TOKEN_PATH ou $HF_HOME/token
    except Exception:
        pass
    if os.environ.get("HF_HOME"):
        paths.append(Path(os.environ["HF_HOME"]) / "token")
    paths.append(Path("~/.cache/huggingface/token").expanduser())
    return list(dict.fromkeys(paths))


def find_hf_token(env_file=None):
    """
    Token do Hugging Face e de onde veio: (token, origem), ou (None, None).

    Ordem:
        1. variável de ambiente HF_TOKEN (export HF_TOKEN=hf_...)
        2. HF_TOKEN no arquivo .env da raiz do projeto
        3. token do `hf auth login` / `huggingface-cli login`
           ($HF_TOKEN_PATH, $HF_HOME/token, ~/.cache/huggingface/token)

    Um HF_TOKEN de 1 ou 2 que não tem a forma de um token real
    (placeholder) é ignorado com um aviso, e a busca segue. O .env nunca
    é modificado; um placeholder na variável de ambiente é retirado só
    deste processo, para o huggingface_hub não enviá-lo nas chamadas
    sem token explícito.
    """
    env_file = Path(env_file) if env_file is not None else ENV_FILE
    from_env = os.environ.get("HF_TOKEN")
    if from_env:
        if looks_like_token(from_env):
            return from_env.strip(), "HF_TOKEN (ambiente)"
        _warn_once("HF_TOKEN do ambiente não parece um token real (placeholder?): ignorado")
        os.environ.pop("HF_TOKEN", None)

    from_file = dotenv_values(env_file).get("HF_TOKEN") if env_file.exists() else None
    if from_file:
        if looks_like_token(from_file):
            os.environ["HF_TOKEN"] = from_file.strip()      # como o load_dotenv fazia
            return from_file.strip(), f"HF_TOKEN ({env_file.name})"
        _warn_once(f"HF_TOKEN em {env_file} não parece um token real (placeholder?): "
                   f"ignorado, usando o token do `hf auth login`")

    for path in _token_files():
        if path.is_file():
            token = path.read_text().strip()
            if token:
                return token, f"hf auth login ({path})"
    return None, None


def get_hf_token() -> str | None:
    """Token do Hugging Face (ver find_hf_token), ou None."""
    return find_hf_token()[0]


def check_repo_access(repo_id: str, revision: str, token=None, repo_type: str = "model",
                      filename: str = None) -> Tuple[bool, str]:
    """
    Confere, sem baixar nada (só HEAD de um arquivo), que o repositório
    existe, o commit fixo existe e o token tem acesso (licença aceita
    nos modelos gated). Returns: (ok, detalhe).
    """
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import (
        EntryNotFoundError, GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError)
    filename = filename or ("config.json" if repo_type == "model" else "README.md")
    url = hf_hub_url(repo_id, filename, revision=revision, repo_type=repo_type)
    try:
        get_hf_file_metadata(url, token=token)
    except GatedRepoError:
        page = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
        return False, f"no access (gated): accept the license at {page} with the token's account"
    except RevisionNotFoundError:
        return False, f"pinned commit {revision} not found"
    except RepositoryNotFoundError:
        return False, "repository not found (or private without access)"
    except EntryNotFoundError:
        return False, f"{filename} not found at commit {revision[:10]} (wrong commit?)"
    except Exception as e:                                  # rede, 5xx
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    return True, f"access ok, commit {revision[:10]} found"


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
            f"   1. hf auth login   (= huggingface-cli login)\n"
            f"   2. export HF_TOKEN='hf_seu_token'\n"
            f"   3. Criar arquivo .env com HF_TOKEN=hf_seu_token\n"
            f"   (um HF_TOKEN placeholder no .env é ignorado)\n"
            f"\n   Obter token:     https://huggingface.co/settings/tokens\n"
            f"   Aceitar licença: https://huggingface.co/{spec.id}"
        )
    return token


def _cache_first(load, repo_id: str, **kwargs):
    """
    from_pretrained do cache local primeiro (local_files_only: nenhuma
    chamada ao Hub, que não tem tempo limite e já travou um job); só vai
    ao Hub se o cache não tiver os arquivos desse commit. Com
    HF_HUB_OFFLINE=1 a segunda tentativa também falha, com o erro do
    transformers. Correção do coautor no servidor (2026-10-30).
    """
    try:
        return load(repo_id, local_files_only=True, **kwargs)
    except OSError:
        return load(repo_id, **kwargs)


def load_tokenizer(model_key: str, registry: Optional[Registry] = None):
    """Só o tokenizer, no commit do registro (verify_setup)."""
    spec = (registry or load_models())[model_key]
    tok = _cache_first(AutoTokenizer.from_pretrained, spec.id, revision=spec.revision,
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
        - revision fixa (commit do registro), modelo e tokenizer, do
          cache local primeiro (_cache_first)
        - dtype da entrada (bfloat16 em todos: o Gemma 2 falha em fp16)
        - quantização da entrada, se houver (nenhum modelo por padrão;
          --dtype nf4: NF4 com lm_head sem quantizar)
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

    tokenizer = _cache_first(AutoTokenizer.from_pretrained, spec.id, revision=spec.revision, token=token)
    # Garantir que o tokenizer tem padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(revision=spec.revision, token=token, dtype=resolve_dtype(spec.precision),
                  device_map="auto", output_hidden_states=True)
    if quant is not None:
        kwargs["quantization_config"] = quant
    model = _cache_first(AutoModelForCausalLM.from_pretrained, spec.id, **kwargs)
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
