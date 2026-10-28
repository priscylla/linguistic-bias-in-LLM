# Verified values

Every fact, translation, model ID or license used by the pipeline is logged here once confirmed by the author. Values not listed here are not yet confirmed and must not be used in runs.

| What | Where | Final value | Confirmed by | Date |
| --- | --- | --- | --- | --- |
| Prompt aviao/pt/F1 | `config/prompts.yaml` aviao.pt.F1[1] | O avião foi inventado por | Priscylla Silva | 2026-10-27 |
| Prompt aviao/pt/F2 | `config/prompts.yaml` aviao.pt.F2[1] | O inventor do avião é considerado | Priscylla Silva | 2026-10-27 |
| Prompt aviao/pt/F3 | `config/prompts.yaml` aviao.pt.F3[1] | O verdadeiro inventor do avião é | Priscylla Silva | 2026-10-27 |
| Prompt aviao/it/F1 | `config/prompts.yaml` aviao.it.F1[1] | L'aereo è stato inventato da | Priscylla Silva | 2026-10-27 |
| Prompt aviao/it/F2 | `config/prompts.yaml` aviao.it.F2[1] | L'inventore dell'aereo è considerato | Priscylla Silva | 2026-10-27 |
| Prompt aviao/it/F3 | `config/prompts.yaml` aviao.it.F3[1] | Il vero inventore dell'aereo è stato | Priscylla Silva | 2026-10-27 |
| Prompt telefone/pt/F1 | `config/prompts.yaml` telefone.pt.F1[1] | O telefone foi inventado por | Priscylla Silva | 2026-10-27 |
| Prompt telefone/pt/F2 | `config/prompts.yaml` telefone.pt.F2[1] | O inventor do telefone é considerado | Priscylla Silva | 2026-10-27 |
| Prompt telefone/pt/F3 | `config/prompts.yaml` telefone.pt.F3[1] | O verdadeiro inventor do telefone é | Priscylla Silva | 2026-10-27 |
| Prompt telefone/it/F1 | `config/prompts.yaml` telefone.it.F1[1] | Il telefono è stato inventato da | Priscylla Silva | 2026-10-27 |
| Prompt telefone/it/F2 | `config/prompts.yaml` telefone.it.F2[1] | L'inventore del telefono è considerato | Priscylla Silva | 2026-10-27 |
| Prompt telefone/it/F3 | `config/prompts.yaml` telefone.it.F3[1] | Il vero inventore del telefono è stato | Priscylla Silva | 2026-10-27 |
| Keyword aviao/pt | `config/facts.yaml` aviao.keyword.pt | avião | Priscylla Silva | 2026-10-27 |
| Model `llama_base` | `config/models.yaml` models.llama_base | meta-llama/Llama-3.1-8B @ d04e592bb4f6aa9cfee91e2e20afa771667e1d4b; base; license llama3.1; gated yes; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `llama` | `config/models.yaml` models.llama | meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f249a16976918f6564b8830bc894c89659; instruct; license llama3.1; gated yes; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `mistral_base` | `config/models.yaml` models.mistral_base | mistralai/Mistral-7B-v0.3 @ caa1feb0e54d415e2df31207e5f4e273e33509b1; base; license apache-2.0; gated no; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `mistral` | `config/models.yaml` models.mistral | mistralai/Mistral-7B-Instruct-v0.3 @ c170c708c41dac9275d15a8fff4eca08d52bab71; instruct; license apache-2.0; gated no; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `qwen_base` | `config/models.yaml` models.qwen_base | Qwen/Qwen2.5-7B @ d149729398750b98c0af14eb82c78cfe92750796; base; license apache-2.0; gated no; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `qwen` | `config/models.yaml` models.qwen | Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28; instruct; license apache-2.0; gated no; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `gemma_base` | `config/models.yaml` models.gemma_base | google/gemma-2-9b @ 33c193028431c2fde6c6e51f29e6f17b60cbfac6; base; license gemma; gated yes; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `gemma` | `config/models.yaml` models.gemma | google/gemma-2-9b-it @ 11c9b309abf73637e4b6f9a3fa1e92e615547819; instruct; license gemma; gated yes; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `gemma27` | `config/models.yaml` models.gemma27 | google/gemma-2-27b-it @ aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0; instruct; license gemma; gated yes; bfloat16, nf4 (bnb NF4, bf16 compute, no double quant) | Priscylla Silva | 2026-10-28 |
| Model `sabia_base` | `config/models.yaml` models.sabia_base | huggyllama/llama-7b @ 4782ad278652c7c71b72204d462d6d01eaaf7549; base; license LLaMA-1 non-commercial (other); gated no; bfloat16 | Priscylla Silva | 2026-10-28 |
| Model `sabia` | `config/models.yaml` models.sabia | maritaca-ai/sabia-7b @ a082d7feed07a2659a38cc2ddd1952396820365b; base; license LLaMA-1 research only (model card); gated no; bfloat16 | Priscylla Silva | 2026-10-28 |
| Sabiá base model | `config/models.yaml` models.sabia.base_model | sabia_base = huggyllama/llama-7b (community LLaMA-1-7B conversion; Sabiá card: trained from LLaMA-1-7B weights) | Priscylla Silva | 2026-10-28 |
| Revision pinning | `config/models.yaml` revision | every model pinned to the commit above | Priscylla Silva | 2026-10-28 |
| Tuned Lens implementation | `logit_lens/tuned_lens.py`, `config/tuned_lens.yaml` | own per-layer affine translator for all four instruct models, recipe of tuned-lens 0.2.0 (package only supports LlamaModel, no Gemma 2 softcap) | Priscylla Silva | 2026-10-28 |
| Tuned Lens corpus | `config/tuned_lens.yaml` corpus | HuggingFaceFW/fineweb-2 @ af9c13333eb981300149d5ca60a8e9d659b276b9 (por_Latn, deu_Latn, ita_Latn, fra_Latn, rus_Cyrl; ODC-By) + HuggingFaceFW/fineweb sample-10BT @ 9bb295ddab0e05d785b879661af7260fed5140fc (en; ODC-By); equal tokens per language; 1% held out by sha256(id) | Priscylla Silva | 2026-10-28 |
| Tuned Lens corpus filter | `config/tuned_lens.yaml` filter_fact_terms | drop documents mentioning any candidate name (all forms, all languages) or a fact keyword of the document's language | Priscylla Silva | 2026-10-28 |
| Tuned Lens budget | `config/tuned_lens.yaml` training | 250 steps × 2^16 tokens (16M tokens per model) | Priscylla Silva | 2026-10-28 |
| Pretrained tuned lenses | — | none usable: official AlignmentResearch/tuned-lens has Llama-3-8B but not 3.1, Mistral, Qwen 2.5 or Gemma 2; community lenses are for base models or English-only corpora (kdhole/Llama-3.1-8B-Instruct-tuned-lens: corpus not stated) | Priscylla Silva | 2026-10-28 |

EN and DE prompts were unchanged in Phase 0 (already correct). The prompts above are paraphrase 1 of each formulation since Phase 1 (moved from `data/prompts.py`, now removed); the new paraphrases and the FR/RU prompts are added to this log as they are confirmed.
