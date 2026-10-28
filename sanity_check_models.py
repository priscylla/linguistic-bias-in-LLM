"""
sanity_check_models.py
Teste de sanidade do lens em cada checkpoint real (Fase 3). Roda na
GPU da autora; aqui só é testado com modelos minúsculos.

    python sanity_check_models.py              # todos os modelos do registro
    python sanity_check_models.py llama gemma27
    python sanity_check_models.py --force      # refaz relatórios existentes

Para cada modelo: carrega do registro (commit fixo, precisão,
quantização), resolve norm final / unembedding / softcap pela
arquitetura e confere, em toda posição analisada, que o lens na última
camada reproduz os logits do modelo:
    - os prompts fixos de logit_lens/lens.py (SANITY_PROMPTS)
    - a paráfrase 1 de F1 do primeiro fato, em cada idioma (completion)
    - modelos instruct: os mesmos fatos no formato QA (chat template)

Um modelo por vez; a memória da GPU é liberada entre eles. Grava
reports/sanity/{modelo}.json e pula modelos com relatório "passed".
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from config.experiment_config import ExperimentConfig
from config.loader import prompts_for, require_valid
from logit_lens.entity_tokens import encode_ids
from logit_lens.lens import SANITY_PROMPTS, LensSanityError, LogitLens

REPORT_DIR = Path("reports") / "sanity"


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


def main(argv=None):
    config = ExperimentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help=f"padrão: todos ({', '.join(config.models)})")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    unknown = [m for m in args.models if m not in config.models]
    if unknown:
        parser.error(f"fora do registro: {unknown}")
    require_valid()

    from models.chat_template import apply_template
    from models.model_loader import load_model

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    keys, failed = args.models or list(config.models), []
    for key in keys:
        path = REPORT_DIR / f"{key}.json"
        if path.exists() and not args.force and json.loads(path.read_text()).get("passed"):
            print(f"  ⏭  {key}: já aprovado ({path})")
            continue
        spec = config.model(key)
        model, tok = load_model(key, config.registry)
        chat = (lambda t, x, mk=key: apply_template(t, mk, x)[0]) if spec.kind == "instruct" else None
        rep = sanity_report(key, model, tok, spec, config, chat)
        path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {'✅' if rep['passed'] else '❌'} {key}: max |diff| = {rep['max_abs_diff']}"
              + "".join(f"\n      {f}" for f in rep["failures"]))
        if not rep["passed"]:
            failed.append(key)
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
