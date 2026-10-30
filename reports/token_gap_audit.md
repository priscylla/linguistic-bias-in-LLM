# Token gap audit (2026-10-30)

Question: which high-probability tokens in the stored top-20 point to a candidate but are not measured (not in any candidate's token IDs for that prompt)?

**Data:** the server results (pack of 2026-10-30, commit fa31f65) — Logit Lens for all 11 models and Tuned Lens for 4; the decision zone (top 30% of layers) and the final layer; name prompts (F1–F3, QA).

**Method:** for each prompt, every top-20 token whose ID is not in that prompt's measured token sets. Per prompt, its peak probability over the decision zone and final layer; aggregated per fact and token.

**Not covered:** forced passes. `run_lens.py` stores top tokens only for main prompts and neutral baselines, not after a forced article. Place and artifact prompts are excluded too, since they measure indicators, not names.

## 1. Given names not in the name sets

| Fact | Token | Peak p | Median p | Final-layer peak | Prompts with p ≥ 0.05 | Models | Points to |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fotografia | Joseph (also `Jo`) | 0.94 | 0.06 | 0.90 | 420 | all 11, all languages, F1–F3 and QA; top-1 at the final layer in 20% of fotografia name prompts (116) | Niépce (Joseph Nicéphore) |
| tabela_periodica | Dimitri | 0.46 | 0.05 | 0.12 | 93 | Gemma 2 (9B, 27B, base) | Mendeleev |
| tabela_periodica | Dmitry | 0.15 | 0.01 | 0.06 | 16 | Gemma 2, Llama 3.1 | Mendeleev |
| telefone | Alessandro (`Aless` in Llama and Qwen) | 0.38 | 0.01 | 0.38 | 30 | Llama, Qwen, Gemma; it | Bell |
| telefone | Alexandre | 0.14 | 0.01 | 0.13 | 19 | 7 models; fr, pt | Bell |
| penicilina | Alexandre | 0.11 | 0.01 | 0.08 | 10 | 6 models; fr, pt | Fleming |
| raios_x | William | 0.70 | 0.03 | 0.19 | 207 | all 11, all languages | Röntgen in English renderings, but also William Crookes or William Coolidge: **ambiguous** |
| raios_x | Conrad | 0.12 | 0.01 | 0.02 | 8 | Gemma 2 27B, Llama | Röntgen (Wilhelm Conrad) |
| www | Timothy | 0.07 | 0.01 | 0.05 | 5 | Gemma, Llama | Berners-Lee |
| tabela_periodica | Julius | 0.07 | 0.01 | 0.01 | 1 | Sabiá | Meyer (Julius Lothar) |

Looked for and not found (peak < 0.02):
- *padre*/*Padre* for Landell (peak 0.008);
- *Wilhelm* for Leibniz, *Henry* for Talbot, *Alfredo* for Nobel (0.05), *Philippe* for Reis.

## 2. Titles and honorifics

| Token | Fact | Peak p | Final-layer peak | Prompts | Points to |
| --- | --- | --- | --- | --- | --- |
| Sir | calculo | 0.88 | 0.82 | 311; top-1 at the final layer in 4% of calculo name prompts | Newton |
| Sir | www | 0.91 | 0.68 | 664; top-1 in 2% | Berners-Lee |
| Sir | penicilina | 1.00 | 0.42 | 518; top-1 in 1% | Fleming |
| Sir | aviao, dinamite, fotografia, raios_x | 0.24–0.64 | 0.04–0.13 | 35–108 | not a candidate (e.g. Sir George Cayley, Sir John Herschel) or unclear |
| Dr | penicilina, raios_x | 0.36, 0.53 | 0.06, 0.02 | 82, 42 | Fleming, Röntgen; never top-1 at the final layer |
| Lord, Professor, Madame, Count | various | ≤ 0.23 | ≤ 0.01 | — | non-candidates (Curie, Zeppelin) or noise |

The models write the English *Sir* in every prompt language. *Father* is not a title here: every hit is an F2 prompt ("considered the father of …"), and it never appears at the final layer.

## 3. Other forms

**A. Mistral QA: measurement bug.** For QA, the candidate is measured as the first token of the name glued to the chat template (no leading space). For Mistral this is the wrong token: the model answers with the leading-space form (`▁Alexander` id 11823 at 0.65, measured `Alex` id 26957 at ~0).
- At the final layer, 79% of the candidate probability in Mistral's QA prompts sits on unmeasured forms (measured 0.000, unmeasured 0.115 on average).
- The article test also never triggers there: 0 of 118 Mistral QA prompts whose top-1 is an article got a forced pass, against 100% for the other models.
- Llama 3.1, Qwen 2.5 and Gemma 2 are measured correctly.

**B. Format tokens as first token** (top-1 at the final layer):

| Model | Token | Prompts |
| --- | --- | --- |
| Gemma 2 9B, 27B | `**` (bold) | 164 + 88 completion, 10 + 11 QA |
| Qwen 2.5 | `__` / `____` (fill-in-the-blank) | 27 |
| Sabiá | `...`, `:` | 92 |

After `**` the name follows; after `__` it doesn't.

**C. Nationality adjectives before the name:**
- Swedish → Nobel: peak 0.95
- Scottish → Fleming, Bell: 0.90
- Russian → Mendeleev: 0.89
- German → Röntgen: 0.85
- French: 0.61
- British: 0.55

They are top-1 at the final layer in only 0.1% of prompts, so they mostly affect the decision zone. They commit to a candidate before the name ("the Swedish chemist Alfred Nobel").

**D. Other spellings of names already in the sets** (no leading space, all caps, another split such as `Alexand`): at most 2% of the measured candidate probability at the final layer, in every model and format except Mistral QA (A).

**E. Function words before names:** pt *como*, it *come* ("considerado como …") are top-1 in about 50 prompts. They are not in the forcing words.

**F. Non-candidates with high mass** (a design choice, listed only): Nikola Tesla for the radio (peak 0.57), Howard Florey for penicillin, Lavoisier for the periodic table, Méliès for cinema.

## 4. Behavioral extraction

The behavioral surface forms (`config/behavioral.yaml`) are separate from the lens name sets, and generation doesn't use either.
- Answers that use a given name almost always also give the surname, so they are extracted: Joseph 666 of 667, William or Conrad 795 of 796, Alexandre or Alessandro 7 of 7, Sir Isaac 132 of 132.
- One gap: 24 German answers (Qwen 21, Mistral 3) write *Mendeleyew*, which is not a surface form, so Mendeleev is not extracted.

## Distribution of the final-layer top-1 token (name prompts, Logit Lens, 11 models)

| Category | Completion | QA |
| --- | --- | --- |
| measured candidate token | 43.5% | 3.8% |
| article / plural noun (already forced) | 37.9% | 68.0% |
| format token | 7.8% | 2.5% |
| given name not in set | 2.2% | 0.1% |
| title | 0.7% | 0.1% |
| other form of a measured name | 0.1% | 1.7% |
| nationality adjective | 0.1% | 0.0% |
| hedge / sentence start | 1.1% | 3.8% |
| other (restated subject, non-candidates, function words) | 6.7% | 20.0% |

## Decisions (2026-10-30, logged in `reports/verified_values.md`)

| Finding | Decision |
| --- | --- |
| Given names (§1) | Added to the name sets:<br>• Niépce: Joseph<br>• Mendeleev: Dimitri, Dmitry<br>• Bell and Fleming: Alexandre (pt, fr), Alessandro (it)<br>• Röntgen: Conrad<br>• Berners-Lee: Timothy<br>• Meyer: Julius<br>No collision with another candidate of the same fact in any of the 11 tokenizers. |
| *William* (X-rays) | Forced like a title, only in raios_x; not counted as Röntgen. |
| *Sir* (§2) | Forced like an article, in all five languages. *Dr* not added. |
| Mistral QA (§3A) | Bug fixed: the QA separator now comes from each chat template. A test over every registered tokenizer would have caught it. |
| `**` (§3B) | Forced, candidates measured right after it. Qwen's `__` and Sabiá's `...`/`:` are not forced (no name follows). |
| Nationality adjectives (§3C) | **Not forced: known limitation.** The models often commit to a candidate through a nationality adjective before naming him (*Swedish* → Nobel 0.95, *Scottish* 0.90, *Russian* 0.89, *German* 0.85, *French* 0.61, *British* 0.55). These are not measured. They are rarely the final-layer top-1 (0.1% of name prompts) but are frequent in the decision zone, where they can make a candidate's internal probability look lower than the model's commitment. To be stated in the paper. |
| Other spellings (§3D) | Left as is (≤ 2% of the measured mass outside Mistral QA, which the fix covers). |
| *como* / *come* (§3E) | Forced (pt, it), with an article or title right after. |
| Forced passes | Their top-20 tokens are now stored, so the next audit covers them. |
| Tuned Lens corpus filter | Lenses kept; the names added afterwards were not in their training filter. |
| Behavioral (§4) | *Mendeleyew* added to Mendeleev's surface forms. |

All lens measurements are redone (measurement version 2): `RUNNING_server.md`, step 6.