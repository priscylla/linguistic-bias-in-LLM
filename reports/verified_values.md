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
| Behavioral sampling | `config/behavioral.yaml` generation | 1 greedy + 10 samples, pure temperature 0.7 (top_p 1, top_k 0, no repetition penalty; models' own decoding defaults overridden), max_new_tokens 200, fixed seed per prompt | Priscylla Silva | 2026-10-28 |
| Behavioral surface forms | `config/behavioral.yaml` surface_forms | new reviewed list (surnames/full names, Latin + Cyrillic six cases, per-language exclusions); **list itself still `proposed`** | Priscylla Silva (approach) | 2026-10-28 |
| Empirical dominant | `analyze.py` empirical_dominant | majority primary attribution over all English answers of all instruct models, 'none' excluded; also per model | Priscylla Silva | 2026-10-28 |
| Validation sample | `config/behavioral.yaml` validation | stratified, 13 answers per model × language (≈390), seed 42 | Priscylla Silva | 2026-10-28 |
| Main statistic | `config/experiment.yaml` statistics.primary_metric | cs_zone (decision-zone mean of CS_e); CS* and final-layer CS in the appendix | Priscylla Silva | 2026-10-28 |
| Bootstrap | `config/experiment.yaml` statistics | percentile, 10,000 resamples, stratified over paraphrases within F1–F3 (9 prompts per cell); per-formulation CIs (3 paraphrases) flagged coarse | Priscylla Silva | 2026-10-28 |
| Stake test | `analysis/statistics.py` stake_tests | CS (not ΔCS); unit = model × format × prompt type; d = stake − mean non-stake; sign-flip permutation (exact n ≤ 16, else 100k Monte Carlo); Holm per lens × set × stake type; dominant candidates via P(dom) − mean P(alt) | Priscylla Silva | 2026-10-28 |
| Mixed models | `analysis/statistics.py` mixed_models | two fits: A ΔCS ~ stake_type + format + base_vs_instruct; B (completion) ΔCS ~ stake_type + formulation + base_vs_instruct; (1|fact) + (1|model) | Priscylla Silva | 2026-10-28 |
| Figure models | `config/experiment.yaml` figures.main_models | main text: llama, mistral, qwen, gemma (instruct); appendix (`_all`): all 11 | Priscylla Silva | 2026-10-28 |
| Fig. 2 layout | `make_figures.py` fig2_delta_cs | one point per model with its bootstrap CI, stake languages shaded | Priscylla Silva | 2026-10-28 |
| Results table | `make_figures.py` results_table | row = model × fact × language × alternative candidate (F1–F3 pooled); behavior = primary rate of e + majority answer; compact = main models, stake languages + EN | Priscylla Silva | 2026-10-28 |
| Fig. 4 axes | `make_figures.py` fig4_internal_behavioral | two panels: CS* and decision-zone CS vs primary-attribution rate | Priscylla Silva | 2026-10-28 |
| Facts status | `config/facts.yaml` status | all 12 facts `confirmed` (candidates, names, stakes, indicators as in `reports/phase1_review.md`); Russian forms listed under `pending_ru_review` still wait for the native review | Priscylla Silva | 2026-10-29 |
| Indicator prompt rule | `config/loader.py` indicator_prompt_ok; `logit_lens/entity_tokens.py` indicator_cell_ok | place/artifact prompts for a fact × language only if at least two candidates have indicators and, per tokenizer, valid tokens in every paraphrase; otherwise the cell is skipped and logged (`skipped` table, manifest, `reports/tokenization_summary.md`) | Priscylla Silva | 2026-10-29 |
| aviao/wright de place | `config/facts.yaml` aviao.wright.indicators.place.de | Kitty Hawk, Kill Devil Hills, North Carolina, Nord-Carolina, Dayton | Priscylla Silva | 2026-10-29 |
| aviao/14-bis | `config/facts.yaml` aviao.santos_dumont.indicators.artifact | keep 14-bis with the automatic rule (dropped if its first token is shared; Demoiselle stays) | Priscylla Silva | 2026-10-29 |
| aviao/santos_dumont ru first name | `config/facts.yaml` aviao.santos_dumont.names.ru | Альберто, variant Алберту (all cases) | Priscylla Silva | 2026-10-29 |
| aviao/wright ru first names | `config/facts.yaml` aviao.wright.names.ru | Уилбур and Уилбер (+ gen/ins); pending Russian review | Priscylla Silva | 2026-10-29 |
| telefone/bell stake | `config/facts.yaml` telefone.bell.stakes | en nationality | Priscylla Silva | 2026-10-29 |
| telefone/bell ru names | `config/facts.yaml` telefone.bell.names.ru | middle name Грейам, variant Грэхем; first name Александр, variant Александер (gen/ins of the new variants pending Russian review) | Priscylla Silva | 2026-10-29 |
| telefone/meucci places it/fr | `config/facts.yaml` telefone.meucci.indicators.place | Avana / Havane, noun only | Priscylla Silva | 2026-10-29 |
| telefone/meucci ru artifact | `config/facts.yaml` telefone.meucci.indicators.artifact.ru | телеттрофон removed | Priscylla Silva | 2026-10-29 |
| telefone/reis names | `config/facts.yaml` telefone.reis.names | Johann Philipp Reis (Johann added) | Priscylla Silva | 2026-10-29 |
| telefone/reis artifact | `config/facts.yaml` telefone.reis.indicators | 'Telephon' removed in all languages; telephone has no artifact prompts | Priscylla Silva | 2026-10-29 |
| radio/marconi Poldhu | `config/facts.yaml` radio.marconi.stakes | no extra stake | Priscylla Silva | 2026-10-29 |
| radio artifact | `config/facts.yaml` radio.popov.indicators | all artifact indicators removed (lightning detector); radio has no artifact prompts | Priscylla Silva | 2026-10-29 |
| radio shared ' São' | tokenization rule | automatic drop kept | Priscylla Silva | 2026-10-29 |
| radio/landell it place | `config/facts.yaml` radio.landell.indicators.place.it | Porto Alegre, San Paolo, São Paulo | Priscylla Silva | 2026-10-29 |
| radio/landell ru names | `config/facts.yaml` radio.landell.names.ru | as proposed; pending Russian review | Priscylla Silva | 2026-10-29 |
| lampada/lodygin fr | `config/facts.yaml` lampada.lodygin.names.fr | Lodyguine, variant Lodygin | Priscylla Silva | 2026-10-29 |
| lampada artifact | — | no artifact prompts | Priscylla Silva | 2026-10-29 |
| cinema dominant | `config/facts.yaml` cinema.lumiere.dominant | Lumière (checked in Phase 5 against the English majority) | Priscylla Silva | 2026-10-29 |
| cinema 'Lumiere' | `config/facts.yaml` cinema.lumiere.names.en | variant kept | Priscylla Silva | 2026-10-29 |
| cinema cine-/ciné- tokens | tokenization rule | automatic drop kept; the two-candidate rule handles one-sided cells | Priscylla Silva | 2026-10-29 |
| calculo artifact | `config/facts.yaml` calculo.newton.indicators | fluxions removed; calculus has no artifact prompts | Priscylla Silva | 2026-10-29 |
| calculo/leibniz place | `config/facts.yaml` calculo.leibniz.indicators.place | Paris added (it Parigi, ru Париж); no new stake | Priscylla Silva | 2026-10-29 |
| fotografia indicators | `config/facts.yaml` fotografia | all as proposed, including Saint-Loup-de-Varennes for Niépce | Priscylla Silva | 2026-10-29 |
| fotografia/talbot ru | `config/facts.yaml` fotografia.talbot.names.ru | Тальбот | Priscylla Silva | 2026-10-29 |
| fotografia/florence stakes | `config/facts.yaml` fotografia.florence.stakes | pt nationality only, no FR stake | Priscylla Silva | 2026-10-29 |
| tabela_periodica/mendeleev spellings | `config/facts.yaml` tabela_periodica.mendeleev.names | as proposed + en variant Mendeleyev | Priscylla Silva | 2026-10-29 |
| tabela_periodica/meyer place | `config/facts.yaml` tabela_periodica.meyer.indicators.place | Karlsruhe (ru Карлсруэ) | Priscylla Silva | 2026-10-29 |
| dinamite/nobel stake | `config/facts.yaml` dinamite.nobel.stakes | none | Priscylla Silva | 2026-10-29 |
| raios_x wording | `config/prompts.yaml` raios_x de/ru | keep X-Strahlen / X-лучи; DE and RU reviewers confirm naturalness | Priscylla Silva | 2026-10-29 |
| Paper format | `config/experiment.yaml` figures.document_class | FAccT 2027: acmart manuscript, single column (`\documentclass[manuscript,screen,review,anonymous]{acmart}`); every figure at \textwidth | Priscylla Silva | 2026-10-29 |
| Figure width (\textwidth) | `config/experiment.yaml` figures.text_width_in | **pending measurement**: compile `reports/latex/measure_textwidth.tex` (no LaTeX on this machine); acmart v2.20 sets no width for manuscript, so it comes from geometry's defaults | Priscylla Silva | 2026-10-29 |
| Languages | `config/experiment.yaml` languages | pt, en, de, it, fr: Russian removed from the study (names, indicators, prompts, baseline prompts, forcing words, surface forms, stakes); the Russian entries above are superseded | Priscylla Silva | 2026-10-29 |
| radio stakes after removing ru | `config/facts.yaml` radio | Popov stays a candidate with no in-scope stake; Landell keeps PT | Priscylla Silva | 2026-10-29 |
| tabela_periodica stakes after removing ru | `config/facts.yaml` tabela_periodica | Mendeleev stays dominant with no in-scope stake; Meyer keeps DE | Priscylla Silva | 2026-10-29 |
| lampada | `config/facts.yaml` | fact removed (no stake language without Lodygin's RU stake); 11 facts remain | Priscylla Silva | 2026-10-29 |
| raios_x ru prompt | `config/prompts.yaml` raios_x | dropped with Russian | Priscylla Silva | 2026-10-29 |
| Native review FR, DE, IT prompts | `config/prompts.yaml` (needs_native_review: false) | reviewed by native speakers and confirmed correct (includes X-Strahlen for raios_x de) | native speakers (reported by Priscylla Silva) | 2026-10-29 |
| Shared first tokens | `logit_lens/entity_tokens.py` resolve_shared_prefixes; `logit_lens/measure.py` | for any candidate pair that collides on a first token, in any format: force the shared token and add P(prefix) × P(next | prefix), layer by layer; every case logged (`prefix_forcing` table, manifest, `reports/tokenization_summary.md`). Replaces 'shared tokens count for neither' | Priscylla Silva | 2026-10-29 |
| Figure width (\textwidth) | `config/experiment.yaml` figures.text_width_in | 430.00462pt = 5.95 in for `\documentclass[manuscript,screen,review,anonymous]{acmart}` (acmart v2.21), measured by the author; supersedes the pending row above | Priscylla Silva | 2026-10-29 |

EN and DE prompts were unchanged in Phase 0 (already correct). The prompts above are paraphrase 1 of each formulation since Phase 1 (moved from `data/prompts.py`, now removed); the new PT/EN paraphrases are added to this log as they are confirmed (still proposed: `reports/review_pt_en_and_support.md`).
