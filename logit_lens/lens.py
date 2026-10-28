"""
logit_lens/lens.py
Interface de lens plugável (Fase 2).

Um lens transforma o hidden state de cada camada em logits do
vocabulário. Todos os lenses compartilham a cabeça de saída do próprio
modelo (norm final → lm_head → softcapping) e diferem só na tradução
aplicada ao hidden state antes dela:

    LogitLens:  tradução identidade (nostalgebraist, 2020)
    TunedLens:  tradução afim aprendida por camada (Fase 4)

Onde ficam a norm final, o unembedding e o softcap vem da arquitetura
(config/models.yaml, `architectures`, pela model.config.model_type).

Na última camada nenhum lens traduz: hidden_states[-1] já é a saída
normalizada do modelo (transformers 5.7, tie_last_hidden_states=True),
então a cabeça pula a norm e o resultado tem que reproduzir os logits
do modelo. Essa igualdade é checada em todo forward (LensSanityError).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from config.loader import Architecture, load_models


class LensSanityError(RuntimeError):
    """O lens na última camada não reproduz os logits do modelo."""


# Prompts fixos do teste de sanidade (conteúdo irrelevante: só importa
# que o lens na última camada bata com os logits do modelo)
SANITY_PROMPTS = [
    "The airplane was invented by",
    "O telefone foi inventado por",
    "Das Telefon wurde erfunden von",
]

def lens_tolerance(dtype: torch.dtype) -> Dict[str, float]:
    """
    Tolerância do teste de sanidade conforme a precisão do modelo.

    Em float32 a diferença é só ruído numérico do matmul.
    Em bf16/fp16 (~3 dígitos significativos) o lm_head aplicado a um
    vetor isolado pode diferir do aplicado à sequência inteira na
    última casa, por isso a folga maior.
    """
    if dtype == torch.float32 or dtype == torch.float64:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 2e-2, "rtol": 2e-2}


def resolve_attr(obj, path: str):
    """Segue um caminho de atributos ("model.norm") a partir de `obj`."""
    for part in path.split("."):
        if not hasattr(obj, part):
            raise ValueError(f"{type(obj).__name__} não tem o atributo '{part}' (caminho '{path}')")
        obj = getattr(obj, part)
    return obj


def output_head(model, architecture: Optional[Architecture] = None) -> Dict:
    """
    Norm final, unembedding e softcap do modelo, pela arquitetura
    declarada em config/models.yaml (chave = model.config.model_type).

    Confere as formas: unembedding [vocab, hidden] e norm com pesos de
    tamanho hidden. Um caminho errado falha aqui, não em silêncio.
    """
    arch  = architecture or load_models().architecture(model.config.model_type.lower())
    norm  = resolve_attr(model, arch.final_norm)
    unemb = resolve_attr(model, arch.unembedding)
    hidden, vocab = model.config.hidden_size, model.config.vocab_size

    w = getattr(unemb, "weight", None)
    if w is None or tuple(w.shape) != (vocab, hidden):
        raise ValueError(f"unembedding '{arch.unembedding}' com forma "
                         f"{None if w is None else tuple(w.shape)}, esperado {(vocab, hidden)}")
    nw = getattr(norm, "weight", None)
    if nw is None or tuple(nw.shape) != (hidden,):
        raise ValueError(f"final_norm '{arch.final_norm}' com forma "
                         f"{None if nw is None else tuple(nw.shape)}, esperado {(hidden,)}")
    softcap = getattr(model.config, arch.softcap, None) if arch.softcap else None
    return {"architecture": arch, "final_norm": norm, "unembedding": unemb, "softcap": softcap}


def assert_matches_model(lens_logits: torch.Tensor, model_logits: torch.Tensor,
                         dtype: torch.dtype, where: str = "") -> Dict:
    """
    Confere que os logits do lens na última camada são os do modelo
    (allclose na tolerância do dtype + mesmo argmax). Levanta
    LensSanityError se não forem.
    """
    model_logits = model_logits.float().to(lens_logits.device)
    tol       = lens_tolerance(dtype)
    max_diff  = (lens_logits - model_logits).abs().max().item()
    argmax_ok = lens_logits.argmax().item() == model_logits.argmax().item()
    close_ok  = torch.allclose(lens_logits, model_logits, **tol)
    if not (close_ok and argmax_ok):
        raise LensSanityError(
            f"❌ Lens na última camada não bate com os logits do modelo "
            f"({where}dtype={dtype}): max |diff| = {max_diff:.3e}, "
            f"tolerância = {tol}, argmax igual = {argmax_ok}. "
            f"Verifique norm final, softcapping e lm_head."
        )
    return {"max_abs_diff": max_diff, "argmax_match": argmax_ok}


class Lens(ABC):
    """
    Lens genérico: tradução por camada + cabeça de saída do modelo.

    Subclasses implementam só `translate`. Camadas são 1-based
    (1..n_layers), como em hidden_states[1:].
    """

    name: str = "lens"

    def __init__(self, model, architecture: Optional[Architecture] = None):
        head = output_head(model, architecture)
        self.model        = model
        self.architecture = head["architecture"]
        self.ln_final     = head["final_norm"]
        self.unembedding  = head["unembedding"]
        self.softcap      = head["softcap"]

    @abstractmethod
    def translate(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        """Hidden state da camada `layer_idx` → espaço da última camada."""

    def head(self, hidden: torch.Tensor, already_normed: bool = False) -> torch.Tensor:
        """
        Cabeça de saída do modelo:

            normed = LayerNorm_final(h)        (pulado se already_normed)
            logits = Unembedding(normed)
            logits = c * tanh(logits / c)      (só se o config definir
                                                final_logit_softcapping)

        O softcapping faz parte da cabeça de saída do Gemma 2
        (Gemma2ForCausalLM.forward), então vale em todas as camadas.

        Returns:
            logits em float32, shape [..., vocab_size]
        """
        with torch.no_grad():
            # Com device_map em várias GPUs, norm/lm_head podem estar
            # em um device diferente do hidden state
            h = hidden.to(self.unembedding.weight.device)
            normed = h if already_normed else self.ln_final(h)
            logits = self.unembedding(normed)
            if self.softcap is not None:
                logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits.float()

    def layer_logits(self, hidden_states, pos: int) -> torch.Tensor:
        """
        Logits do lens em todas as camadas, na posição `pos`.

        Args:
            hidden_states: outputs.hidden_states (n_layers + 1 tensors
                           [1, seq, hidden]; índice 0 = embeddings)
            pos:           posição analisada

        Returns:
            tensor float32 [n_layers, vocab_size] (linha k-1 = camada k)
        """
        n_layers = len(hidden_states) - 1
        with torch.no_grad():
            mid = [self.translate(k, hidden_states[k][0, pos])
                   for k in range(1, n_layers)]
            rows = []
            if mid:
                rows.append(self.head(torch.stack(mid), already_normed=False))
            # Última camada: já normalizada, sem tradução
            rows.append(self.head(hidden_states[n_layers][0, pos],
                                  already_normed=True).unsqueeze(0))
        return torch.cat(rows, dim=0)

    def check_consistency(self, outputs, pos: int, last_logits: Optional[torch.Tensor] = None) -> Dict:
        """
        Lens na última camada == logits do modelo na posição `pos`.
        `last_logits` evita recalcular quando já se tem layer_logits.
        """
        if last_logits is None:
            last_logits = self.head(outputs.hidden_states[-1][0, pos], already_normed=True)
        return assert_matches_model(
            last_logits, outputs.logits[0, pos], outputs.logits.dtype,
            where=f"{self.name}, {self.model.config.model_type}, posição {pos}, ")

    def sanity_check(self, tokenizer, prompts: Optional[List[str]] = None) -> Dict:
        """
        Roda prompts fixos e confere, na última posição de cada um, que a
        cabeça do lens reproduz os logits do modelo. Levanta
        LensSanityError se falhar.
        """
        device = next(self.model.parameters()).device
        report = {"lens": self.name,
                  "model_type": self.model.config.model_type,
                  "dtype": str(next(self.model.parameters()).dtype),
                  "prompts": []}
        for prompt in prompts or SANITY_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            pos = inputs["input_ids"].shape[1] - 1
            report["prompts"].append({"prompt": prompt, **self.check_consistency(outputs, pos)})
        report["max_abs_diff"] = max(p["max_abs_diff"] for p in report["prompts"])
        print(f"  ✅ Sanidade do lens OK ({self.name}, {report['model_type']}, "
              f"{report['dtype']}): max |diff| = {report['max_abs_diff']:.2e}")
        return report


class LogitLens(Lens):
    """Logit Lens: a cabeça do modelo aplicada direto a cada camada."""

    name = "logit_lens"

    def __init__(self, model, architecture: Optional[Architecture] = None):
        super().__init__(model, architecture)

    def translate(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        return hidden
