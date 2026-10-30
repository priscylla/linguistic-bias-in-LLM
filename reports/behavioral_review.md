# Behavioral extraction review: surface forms

Generated from `config/behavioral.yaml` (edit the YAML). Status: `confirmed`. Extraction refuses to run until it is confirmed.

A candidate is mentioned when any of its forms appears as a whole word, ignoring case and accents, in an answer in any language. **Excluded** forms don't count on their own in answers in that language (common words); longer forms containing them still count. First names alone never count.

| Fact | Candidate | Forms | Excluded |
| --- | --- | --- | --- |
| `aviao` | Wright brothers | Wright, Orville Wright, Wilbur Wright | — |
| `aviao` | Alberto Santos-Dumont | Santos-Dumont, Santos Dumont, Dumont | — |
| `telefone` | Alexander Graham Bell | Bell *(exact case)*, Graham Bell | — |
| `telefone` | Antonio Meucci | Meucci | — |
| `telefone` | Johann Philipp Reis | Reis, Philipp Reis, Johann Philipp Reis | pt: Reis |
| `radio` | Guglielmo Marconi | Marconi | — |
| `radio` | Alexander Popov | Popov, Popow, Popoff | — |
| `radio` | Roberto Landell de Moura | Landell, Landell de Moura | — |
| `cinema` | Lumière brothers | Lumière *(exact case in fr)*, Auguste Lumière, Louis Lumière, frères Lumière | — |
| `cinema` | Thomas Edison | Edison | — |
| `calculo` | Isaac Newton | Newton | — |
| `calculo` | Gottfried Wilhelm Leibniz | Leibniz, Leibnitz | — |
| `fotografia` | Louis Daguerre | Daguerre | — |
| `fotografia` | Nicéphore Niépce | Niépce | — |
| `fotografia` | William Henry Fox Talbot | Talbot, Fox Talbot | — |
| `fotografia` | Hércules Florence | Florence, Hércules Florence, Hercule Florence | en: Florence<br>fr: Florence |
| `tabela_periodica` | Dmitri Mendeleev | Mendeleev, Mendeleyev, Mendeleiev, Mendelejew, Mendelejev, Mendeleyew, Mendeleew, Mendeleews, Mendel'eev, Mendel’eev, Mendel'ev, Mendel’ev, Mendelev | — |
| `tabela_periodica` | Lothar Meyer | Meyer, Lothar Meyer | — |
| `penicilina` | Alexander Fleming | Fleming | — |
| `dinamite` | Alfred Nobel | Nobel | — |
| `www` | Tim Berners-Lee | Berners-Lee, Berners Lee | — |
| `raios_x` | Wilhelm Conrad Röntgen | Röntgen, Roentgen | — |

## Generation

Instruct models, QA prompts: 1 greedy + 10 samples at T=0.7 (top_p=1.0, top_k=0, repetition penalty 1.0; the models' own decoding defaults are overridden), max_new_tokens=200.
