"""
models/runtime.py
Opções comuns dos pontos de entrada que carregam modelos (run_lens.py,
run_behavioral.py, train_tuned_lens.py):

    --model KEY        chave do registro (config/models.yaml)
    --gpus "1 2"       GPUs físicas do job (define CUDA_VISIBLE_DEVICES antes
                       de qualquer uso de CUDA); com mais de uma, o modelo é
                       dividido entre elas (device_map="auto"). Sem --gpus,
                       vale o CUDA_VISIBLE_DEVICES do ambiente (run_queue.sh)
    --dtype D          sobrescreve a precisão do registro: bfloat16 | float16 |
                       float32 (sem quantização) ou nf4 (4 bits; Gemma 2 27B
                       usava nf4 até 2026-10-29, hoje bf16). Muda o método:
                       registrado no manifest e verificado na retomada.

Cada modelo grava nos próprios arquivos (results/lens/{lens}/{modelo},
results/behavioral/{modelo}, lenses/{modelo}), então jobs paralelos
nunca escrevem no mesmo arquivo.
"""

import dataclasses
import os

DTYPE_ALIASES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
                 "bfloat16": "bfloat16", "float16": "float16", "float32": "float32", "nf4": "nf4"}


def add_model_args(parser, models):
    parser.add_argument("--model", required=True, choices=list(models))
    parser.add_argument("--gpus", default=None,
                        help='GPUs do job, ex.: "0" ou "1 2" (padrão: CUDA_VISIBLE_DEVICES)')
    parser.add_argument("--dtype", default=None, choices=sorted(DTYPE_ALIASES),
                        help="sobrescreve a precisão do registro (nf4 = 4 bits)")
    return parser


def apply_gpus(gpus) -> str:
    """Restringe o processo às GPUs pedidas. Chamar antes de usar CUDA."""
    if gpus is None:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [g for g in str(gpus).replace(",", " ").split() if g]
    if not ids or not all(g.isdigit() for g in ids):
        raise ValueError(f"--gpus inválido: {gpus!r} (ex.: \"0\" ou \"1 2\")")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA já inicializado: --gpus precisa ser aplicado antes")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # mesma numeração do nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    return os.environ["CUDA_VISIBLE_DEVICES"]


def registry_with_dtype(registry, model_key: str, dtype=None):
    """Cópia do registro com a precisão de `model_key` trocada por --dtype."""
    if dtype is None:
        return registry
    d = DTYPE_ALIASES[dtype]
    spec = registry[model_key]
    if d == "nf4":
        if "nf4" not in registry.quantization:
            raise ValueError("quantização nf4 ausente de config/models.yaml")
        new = dataclasses.replace(spec, precision="bfloat16", quantization="nf4")
    else:
        new = dataclasses.replace(spec, precision=d, quantization=None)
    return dataclasses.replace(registry, models={**registry.models, model_key: new})


def precision_tag(spec) -> str:
    return spec.quantization or spec.precision


def check_resume(manifest: dict, spec, out_dir):
    """Recusa retomar resultados gravados com outra precisão ou outro commit."""
    if not manifest:
        return
    old = manifest.get("precision_tag")
    new = precision_tag(spec)
    if old is not None and old != new:
        raise RuntimeError(
            f"{out_dir} foi gravado com precisão '{old}', este run usa '{new}'. "
            f"Use outro --results-dir ou apague a pasta; resultados não são misturados.")
    rev = (manifest.get("registry") or {}).get("revision") or manifest.get("registry_revision")
    if rev and rev != spec.revision:
        raise RuntimeError(f"{out_dir} foi gravado no commit {rev}, o registro diz {spec.revision}.")
