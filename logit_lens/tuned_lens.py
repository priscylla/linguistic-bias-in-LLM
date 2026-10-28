"""
logit_lens/tuned_lens.py
Tuned Lens (Belrose et al., 2023): tradução afim aprendida por camada,
na mesma interface Lens do Logit Lens (logit_lens/lens.py).

    translate(k, h) = h + A_k·h + b_k        camadas k = 1..L−1

A_k e b_k começam em zero, então um Tuned Lens recém-criado é
exatamente o Logit Lens. A última camada não tem tradutor (já é a saída
do modelo) e a cabeça (norm final, unembedding, softcap) é a do próprio
modelo, como em todo Lens.

Os tradutores ficam em float32 (padrão do config); o hidden state é
convertido para float32, traduzido e volta ao dtype do modelo antes da
cabeça.

Arquivos (lenses/{modelo}/):
    translators.pt   state_dict dos tradutores
    config.json      modelo, commit, camadas, d_model, metadados do treino
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from logit_lens.lens import Lens

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class LensMismatch(RuntimeError):
    """Os tradutores salvos não são deste modelo (id, commit ou formas)."""


class TunedLens(Lens):

    name = "tuned_lens"

    def __init__(self, model, architecture=None, translator_dtype: str = "float32"):
        super().__init__(model, architecture)
        d, n = model.config.hidden_size, model.config.num_hidden_layers
        dev  = self.unembedding.weight.device
        self.translator_dtype = DTYPES[translator_dtype]
        # Um tradutor por camada 1..L−1 (índice k−1)
        self.translators = nn.ModuleList()
        for _ in range(n - 1):
            lin = nn.Linear(d, d, bias=True, device=dev, dtype=self.translator_dtype)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.translators.append(lin)
        self.meta: Dict = {}

    def translate(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        t = self.translators[layer_idx - 1]
        h = hidden.to(device=t.weight.device, dtype=self.translator_dtype)
        return (h + t(h)).to(hidden.dtype)

    def layer_logits_grad(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        """
        Logits de uma camada com gradiente nos tradutores (treino).
        `hidden` [..., d] da camada `layer_idx` (1..L−1).
        """
        h = self.translate(layer_idx, hidden)
        h = h.to(self.unembedding.weight.device)
        logits = self.unembedding(self.ln_final(h))
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits.float()

    def parameters(self):
        return self.translators.parameters()

    # ── Arquivos ──

    def identity_info(self) -> Dict:
        cfg = self.model.config
        return {"model_type": cfg.model_type, "n_layers": cfg.num_hidden_layers,
                "d_model": cfg.hidden_size, "vocab_size": cfg.vocab_size,
                "revision": getattr(cfg, "_commit_hash", None),
                "name_or_path": getattr(cfg, "_name_or_path", None)}

    def save(self, out_dir, meta: Optional[Dict] = None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "translators.pt.tmp"
        torch.save(self.translators.state_dict(), tmp)
        tmp.replace(out / "translators.pt")
        info = {**self.identity_info(), "translator_dtype": str(self.translator_dtype),
                "meta": {**self.meta, **(meta or {})}}
        (out / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    @classmethod
    def load(cls, model, lens_dir, architecture=None, expected_revision: Optional[str] = None):
        """
        Carrega tradutores treinados. Recusa (LensMismatch) se as formas
        ou o commit do modelo não batem com os do treino.
        """
        lens_dir = Path(lens_dir)
        info = json.loads((lens_dir / "config.json").read_text(encoding="utf-8"))
        dtype = next(k for k, v in DTYPES.items() if str(v) == info["translator_dtype"])
        lens = cls(model, architecture, translator_dtype=dtype)
        mine = lens.identity_info()
        for k in ("model_type", "n_layers", "d_model", "vocab_size"):
            if info[k] != mine[k]:
                raise LensMismatch(f"{lens_dir}: {k} do treino = {info[k]}, modelo = {mine[k]}")
        trained = info.get("revision")
        for rev in (mine["revision"], expected_revision):
            if trained and rev and trained != rev:
                raise LensMismatch(f"{lens_dir}: treinado no commit {trained}, modelo/registro {rev}")
        state = torch.load(lens_dir / "translators.pt",
                           map_location=lens.unembedding.weight.device, weights_only=True)
        lens.translators.load_state_dict(state)
        lens.meta = info.get("meta", {})
        return lens
