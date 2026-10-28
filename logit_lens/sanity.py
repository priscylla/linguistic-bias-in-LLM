"""
logit_lens/sanity.py
Teste de sanidade do lens num checkpoint real (Fase 3), usado por
`run_lens.py --model ... --sanity-only`.

Carregado o modelo (commit, precisão e quantização do registro), a
cabeça de saída é resolvida pela arquitetura e, em toda posição
analisada, o lens na última camada tem que reproduzir os logits do
modelo:
    - os prompts fixos de logit_lens/lens.py (SANITY_PROMPTS)
    - a paráfrase 1 de F1 do primeiro fato, em cada idioma (completion)
    - modelos instruct: os mesmos fatos no formato QA (chat template)

Falhas não levantam erro: ficam em report["failures"] e
report["passed"] = False.
"""

import torch

from config.loader import prompts_for
from logit_lens.entity_tokens import encode_ids
from logit_lens.lens import SANITY_PROMPTS, LensSanityError, LogitLens


def _check_ids(lens, model, ids):
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True)
    return lens.check_consistency(out, len(ids) - 1)


def sanity_report(model_key, model, tokenizer, spec, config, chat_formatter=None) -> dict:
    """
    Relatório de sanidade de um modelo já carregado. Não levanta erro:
    falhas ficam em report["failures"] e report["passed"] = False.
    """
    report = {"model_key": model_key, "model_id": spec.id, "revision": spec.revision,
              "loaded_revision": getattr(model.config, "_commit_hash", None),
              "kind": spec.kind, "precision": spec.precision,
              "quantization": spec.quantization,
              "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
              "model_type": model.config.model_type,
              "n_layers": model.config.num_hidden_layers,
              "checks": [], "failures": []}
    try:
        lens = LogitLens(model)
    except ValueError as e:
        report.update(passed=False, failures=[f"cabeça de saída: {e}"])
        return report
    report["head"] = {"final_norm": lens.architecture.final_norm,
                      "unembedding": lens.architecture.unembedding,
                      "softcap": lens.softcap,
                      "unembedding_dtype": str(lens.unembedding.weight.dtype).replace("torch.", "")}

    cases = [("fixed", None, t, True) for t in SANITY_PROMPTS]
    first = config.facts[0]
    for lang in config.languages:
        ps = prompts_for(config.prompt_specs, first, lang, "F1")
        if ps:
            cases.append(("completion", lang, ps[0].text, True))
        if spec.kind == "instruct" and chat_formatter is not None:
            qa = prompts_for(config.prompt_specs, first, lang, "QA")
            if qa:
                cases.append(("qa", lang, chat_formatter(tokenizer, qa[0].text), False))

    for kind, lang, text, add_special in cases:
        try:
            res = _check_ids(lens, model, encode_ids(tokenizer, text, add_special))
            report["checks"].append({"kind": kind, "lang": lang, **res})
        except LensSanityError as e:
            report["failures"].append(f"{kind}/{lang}: {e}")
    report["max_abs_diff"] = max((c["max_abs_diff"] for c in report["checks"]), default=None)
    report["passed"] = not report["failures"] and len(report["checks"]) == len(cases)
    return report
