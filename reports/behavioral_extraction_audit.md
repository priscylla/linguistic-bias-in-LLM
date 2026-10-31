# Behavioral extraction audit (2026-10-30)

Question: are answers missed by the surface forms (`config/behavioral.yaml`)? Prompted by *Mendeleyew*, which moved Qwen's German tabela_periodica primary rate for Mendeleev from 24% to 97%.

**Data:** all 9,075 answers (5 instruct models × 11 facts × 5 languages × 33), extracted with the current surface forms. **Method:** CPU only, three checks.

## 1. Answers with no candidate

308 answers (3.4%) match no candidate. The table lists every fact × model × language with at least one, with the most frequent capitalized words in those answers (sentence-starter words removed).

| Fact | Model | Language | No candidate | Most frequent capitalized words |
| --- | --- | --- | --- | --- |
| aviao | llama | de | 14/33 | Ingenieur (13), Lilienthal (12), Otto (12), Pioniere (11), Luftfahrt (10), Flugzeugs (9) |
| aviao | llama | en | 1/33 | Aeronautics (1), Lilienthal (1), Otto (1), German (1), Sir (1), Father (1) |
| aviao | llama | fr | 23/33 | Clément (19), Ader (19), L'invention (17), Lilienthal (12), Otto (12), Éole (12) |
| aviao | llama | it | 27/33 | L'invenzione (25), Lilienthal (14), Otto (14), Ader (14), Clément (12), Éole (12) |
| aviao | llama | pt | 17/33 | Clément (15), Ader (15), Lilienthal (14), Otto (14), Éole (12), Chanute (4) |
| aviao | mistral | de | 5/33 | Erfinder (5), Flugzeugs (4), Lilienthal (3), Otto (3), Cayley (3), Sir (3) |
| aviao | mistral | fr | 2/33 | Lilienthal (2), Otto (2), Sir (1), Cayley (1), George (1), L'avion (1) |
| aviao | mistral | it | 3/33 | L'aereo (3), Samuel (3), Langley (3), Pierpont (3), Lilienthal (2), Otto (2) |
| aviao | qwen | de | 11/33 | Flugzeug (9), Cayley (9), Sir (9), George (9), Vater (7), Person (7) |
| calculo | llama | de | 7/33 | Archimedes (7), Bonaventura (7), Chr (7), Cavalieri (7), Infinitesimalrechnung (7), Mathematikern (7) |
| calculo | llama | fr | 19/33 | Cavalieri (19), Archimède (16), Bonaventura (15), Pierre (12), Fermat (12), Bonaventure (4) |
| calculo | llama | it | 12/33 | Cavalieri (11), Bonaventura (11), Archimede (6), XVII (5), Mengoli (4), Pietro (4) |
| calculo | llama | pt | 2/33 | Cavalieri (2), Archimedes (2), Bonaventura (2) |
| calculo | mistral | fr | 1/33 | Pierre (1), Gilbert (1), Cavalieri (1), Bonaventura (1), Leonhard (1), Euler (1) |
| calculo | mistral | pt | 1/33 | Simon (1), John (1), XVI (1), XVIII (1), Pierre (1), Cavalieri (1) |
| calculo | qwen | pt | 1/33 |  |
| cinema | llama | de | 20/33 | Kinos (20), Muybridge (17), Eadweard (17), Fotograf (15), Erfinder (14), Personen (13) |
| cinema | llama | en | 20/33 | Eadweard (20), Muybridge (20), Louis (19), French (19), Prince (19), Marey (16) |
| cinema | llama | fr | 11/33 | Eadweard (7), Muybridge (7), Louis (6), Marey (6), Étienne-Jules (6), Prince (4) |
| cinema | llama | it | 15/33 | Eadweard (14), Muybridge (14), Louis (11), Prince (11), L'invenzione (9), Scene (8) |
| cinema | llama | pt | 16/33 | Eadweard (15), Muybridge (15), Marey (11), Étienne-Jules (11), Louis (5), Prince (5) |
| cinema | mistral | de | 2/33 | Ingenieur (2), Kurzfilmen (1), November (1), Boxerszene (1), Kino (1), Variete (1) |
| cinema | mistral | fr | 3/33 | Louis (3), Lumière (3), Prince (2), L'inventeur (2), Toutefois (2), Lyon (1) |
| cinema | mistral | it | 3/33 | Louis (3), Prince (3), Lune (2), Georges (2), Méliès (2), Voyage (2) |
| cinema | mistral | pt | 1/33 | Charles-Émile (1), Louis-Jules (1), Reynaud (1), Praxinoscopio (1), Duboscq (1), XIX (1) |
| cinema | qwen | de | 4/33 | Kino (4), Louis (4), Entwicklung (4), Allerdings (3), Daguerre (3), Joseph (3) |
| cinema | qwen | en | 4/33 | French (4), Marey (4), Étienne-Jules (4), Louis (3), Prince (3), Joseph (3) |
| cinema | qwen | fr | 8/33 | Louis (8), Marey (8), Étienne-Jules (8), Joseph (8), Plateau (8), Daguerre (5) |
| cinema | qwen | it | 7/33 | Louis (6), Joseph (5), Plateau (4), Daguerre (4), Prince (3), Marey (3) |
| cinema | qwen | pt | 10/33 | Louis (10), Joseph (9), Daguerre (8), Plateau (7), Marey (4), Étienne-Jules (4) |
| dinamite | llama | de | 6/33 | Dynamit (5), Unternehmer (5), Chemiker (5), Gustav (3), Johan (3), Dynamits (2) |
| dinamite | llama | it | 1/33 | Svante (1), August (1), Arrhenius (1), Ammonium (1), L'invenzione (1) |
| dinamite | llama | pt | 2/33 | Einar (2), Anker (1), Emil (1), Palme (1), Gustaf (1) |
| dinamite | qwen | de | 1/33 | Chemiker (1), Arbeitsfähigkeit (1), Stoffen (1), Viskosität (1), Julius (1), Dynamits (1) |
| fotografia | gemma | it | 1/33 | Albrecht (1), Thomas (1), Ecco (1), Dürer (1), Kepler (1), Johannes (1) |
| fotografia | llama | fr | 1/33 | Alexandrie (1), Thomas (1), L'invention (1), Ier (1), Wedgwood (1), Héron (1) |
| fotografia | mistral | en | 1/33 | Thomas (1), BCE (1), Nicéphore (1), Joseph (1), One (1), Renaissance (1) |
| fotografia | qwen | de | 1/33 | Maier (1), Fotografie (1), Bilder (1), Schritte (1), Substanz (1), Thomas (1) |
| radio | llama | de | 2/33 | Hermann (2), Mathematiker (2), Hertz (2), Clerk (2), Maxwell (2), Helmholtz (2) |
| radio | llama | en | 1/33 | Nikola (1), Scottish (1), Tesla (1), Serbian-American (1), Heinrich (1), Hertz (1) |
| radio | llama | it | 1/33 | Tra (1), Heinrich (1), Hertz (1), Hermann (1), Clerk (1), Helmholtz (1) |
| radio | mistral | de | 2/33 | Clerk (2), Maxwell (2), James (2), Radio (2), Entwicklung (2), Radios (2) |
| radio | mistral | en | 1/33 | Scottish (1), Heinrich (1), Hertz (1), German (1), Clerk (1), Gugliel (1) |
| radio | mistral | fr | 1/33 | Canadien (1), Edwin (1), Fessenden (1), Armstrong (1), Américain (1), Reginald (1) |
| radio | qwen | de | 2/33 | Ingenieuren (2), Jahre (2), Nikola (2), Tesla (2), Entwicklung (2), Radios (2) |
| radio | qwen | fr | 2/33 | Heinrich (2), L'invention (2), Hertz (2), Clerk (2), Maxwell (2), James (2) |
| radio | qwen | it | 2/33 | Heinrich (2), Hertz (2), Clerk (2), Maxwell (2), L'invenzione (2), James (2) |
| radio | qwen | pt | 3/33 | Clerk (3), Maxwell (3), James (3), Nikola (2), Tesla (2), Heinrich (2) |
| raios_x | mistral | pt | 1/33 | William (1), Pierre (1), Bragg (1), Becquerel (1), Henry (1), Henri (1) |
| tabela_periodica | llama | de | 1/33 | John (1), Elemente (1), Chemiker (1), Reihenfolge (1), Atomgewichte (1), Periodensystems (1) |
| tabela_periodica | llama | it | 3/33 | Ivanovič (2), Dmitrij (2), John (2), Newlands (2), Congresso (1), Parigi (1) |
| tabela_periodica | mistral | de | 1/33 | Buch (1), Dmitri (1), Elemente (1), System (1), Tabelle (1), Atomgewichte (1) |
| tabela_periodica | qwen | it | 1/33 | John (1), Antoine (1), Lavoisier (1), Newlands (1) |

Almost all of these are genuine.

**Non-candidates named:**
- airplane: Ader, Lilienthal, Cayley, Chanute (Llama credits Clément Ader or Otto Lilienthal in half its it/fr/pt/de answers);
- cinema: Muybridge, Le Prince, Marey, Plateau, Daguerre and Niépce;
- calculus: Cavalieri, Archimedes, Fermat;
- radio: Maxwell, Hertz, Tesla;
- periodic table: Newlands;
- X-rays: the Braggs.

**Invented people:** dynamite, Llama in de and pt: "Gustav Blix", "Einar Anker".

**No single inventor:** answers that say there isn't one.

**Missed:**
- Mendeleev spelled *Mendel'eev* or *Mendel'ev* (it, Llama) and *Mendeleew* (de, Mistral);
- Lumière in French: "le Lumière frères", "c'est le Lumière" (see 3).

## 2. Near-miss spellings in all answers

Every word within 1–2 edits of a single-word surface form, or equal to one plus a suffix of up to 3 letters, in an answer where that candidate was not extracted. Words containing a compound part are skipped (*Röntgenstrahlen*, *Nobelpreis*).

Only Mendeleev variants come up:

| Spelling | Answers | Language | Model |
| --- | --- | --- | --- |
| *Mendeleew* | 3 | de | Mistral |
| *Mendeleews* | 1 | de | Mistral |
| *Mendel'ev* | 2 | it | Llama |
| *Mendel'eev* | 1 | it | Llama |
| *Mendelev* | 1 | it | Qwen |

The other hits are false: *handelt* and *Ländern* near *Landell*. German genitives (*Edisons*, *Meyers*) only appear in answers where the plain name also matched.

## 3. Language-specific exclusions

Some bare forms are excluded in one language because they are also ordinary words there:
- *Reis* in pt ("kings"): never capitalized in a pt telephone answer.
- *Florence* in en and fr (the city): never in a photography answer.
- *Lumière* in fr ("light"): 4 answers name the brothers but are missed.
  - In "les frères **Lumière, Auguste et Louis**", the bold markup splits *frères Lumière*.
  - "le Lumière frères" and "c'est le Lumière" have no multi-word form to match.
  - Lowercase *lumière* appears in only 1 fr answer.

## Proposals (applied 2026-10-30, logged in `reports/verified_values.md`)

| # | Change | Answers changed (simulated on all answers) |
| --- | --- | --- |
| 1 | Mendeleev: add *Mendeleew, Mendeleews, Mendel'eev, Mendel’eev, Mendel'ev, Mendel’ev, Mendelev* | 7:<br>• no candidate → Mendeleev: 3 (Llama it 2, Mistral de 1)<br>• Meyer → Mendeleev: 3 (Mistral de 2, Qwen it 1; Mendeleev is named first)<br>• order only: 1 |
| 2 | Lumière: count *Lumière* only with a capital L, in every language (like *Bell*), instead of excluding it in fr | 5, all fr:<br>• no candidate → Lumière: 3 (Mistral)<br>• Edison → Lumière: 2 (Gemma 2 9B, 27B) |

Not needed:
- stripping markdown before matching: it only mattered for Lumière in fr, which 2 covers;
- changes to the *Reis* and *Florence* exclusions: no leaks found.

After applying both, `analyze.py` changed exactly the 12 simulated answers. No-candidate answers went from 308 to 302.

## Non-candidate people (descriptive table)

`results/metrics/behavioral_noncandidates.csv` (`analyze.py`) lists, per fact × model × language, every person from `config/behavioral.yaml` `non_candidates` named in the answers with no candidate. Each row has:
- `n_answers`, the number of such answers;
- the cell totals (`n_no_candidate`, `n_answers_cell`);
- a status:
  - **related**: a real person who worked on the invention or a direct precursor;
  - **misattributed**: a real person with no connection to this invention;
  - **fabricated**: no such person (invented or garbled names).

  Every non-related entry has a note saying why (reclassification of 2026-10-30, which replaced real / invented / unclear and resolved the unclear names).

The list is descriptive only: these people are not candidates and enter no metric.

All 302 no-candidate answers name at least one listed person, except 1. That is a Qwen calculus answer in Portuguese that switches to Chinese and names Newton and Leibniz in Chinese characters, which the extraction cannot read. The other former exception, "Albert von Schönbein" (Llama, dynamite, de), is now on the list as fabricated. Most named, in answers with no candidate:

| Fact | No-candidate answers | Most named |
| --- | --- | --- |
| aviao | 103 | Otto Lilienthal 63, Clément Ader 57, George Cayley 32, Octave Chanute 27, Leonardo da Vinci 20, Samuel P. Langley 5 |
| cinema | 121 | Eadweard Muybridge 78, Louis Le Prince 69, Étienne-Jules Marey 67, Joseph Plateau 28, Louis Daguerre 20 |
| calculo | 43 | Bonaventura Cavalieri 41, Archimedes 31, Pierre de Fermat 21, Pietro Mengoli 4, Johannes Kepler 4 |
| radio | 17 | James Clerk Maxwell 14, Heinrich Hertz 12, Nikola Tesla 5 |
| dinamite | 10 | mostly fabricated people (Llama, de and pt): "Einar Anker", "Einar Gustaf Emil Palme", "Gustav Blix", "Albert von Schönbein", … |
| fotografia | 4 | Thomas Wedgwood 4 |
| tabela_periodica | 3 | John Newlands 3, Antoine Lavoisier 2 |
| raios_x | 1 | Becquerel, Curie, the Braggs |

Status, counted over answer × person mentions:
- related 672, misattributed 14, fabricated 26.
- **Fabricated:** Llama 22, Qwen 3, Mistral 1. Dynamite de 10 and pt 2; the rest spread over airplane, calculus and cinema.
- **Misattributed:** Llama 9, Qwen 2, Mistral 2, Gemma 1. Examples: Gilbert N. Lewis for calculus, Eilert Sundt for dynamite, Scott de Martinville for radio.

The full review is in `reports/noncandidates_review.md`:
- every person, with an example;
- the full answers for the fabricated and misattributed names;
- language concentration;
- the share of no-candidate answers per fact × model × language;
- script and language switches (48 script switches, 40 of them Qwen into Chinese, and 5 whole-answer language switches in 9,075 answers).