# Behavioral extraction review: surface forms

Generated from `config/behavioral.yaml` (edit the YAML). Status: `proposed`; Russian declensions native review pending: `True`. Extraction refuses to run until both are cleared.

A candidate is mentioned when any of its forms appears as a whole word, ignoring case and accents, in an answer in any language. Latin and Cyrillic forms are both matched in every answer. **Excluded** forms don't count on their own in answers in that language (common words); longer forms containing them still count. First names alone never count.

| Fact | Candidate | Latin | Cyrillic (6 cases) | Excluded |
| --- | --- | --- | --- | --- |
| `aviao` | Wright brothers | Wright, Orville Wright, Wilbur Wright | Райт, Райта, Райту, Райтом, Райте, Райтов, Райтам, Райтами, Райтах | — |
| `aviao` | Alberto Santos-Dumont | Santos-Dumont, Santos Dumont, Dumont | Сантос-Дюмон, Сантос-Дюмона, Сантос-Дюмону, Сантос-Дюмоном, Сантос-Дюмоне, Дюмон, Дюмона, Дюмону, Дюмоном, Дюмоне | — |
| `telefone` | Alexander Graham Bell | Bell, Graham Bell | Белл, Белла, Беллу, Беллом, Белле | — |
| `telefone` | Antonio Meucci | Meucci | Меуччи | — |
| `telefone` | Johann Philipp Reis | Reis, Philipp Reis, Johann Philipp Reis | Рейс, Рейса, Рейсу, Рейсом, Рейсе, Филипп Рейс, Филиппа Рейса, Филиппу Рейсу, Филиппом Рейсом, Филиппе Рейсе | pt: Reis<br>ru: Рейс, Рейса, Рейсу, Рейсом, Рейсе |
| `radio` | Guglielmo Marconi | Marconi | Маркони | — |
| `radio` | Alexander Popov | Popov, Popow, Popoff | Попов, Попова, Попову, Поповым, Попове | — |
| `radio` | Roberto Landell de Moura | Landell, Landell de Moura | Ланделл, Ланделла, Ланделлу, Ланделлом, Ланделле, Ланделл де Моура | — |
| `lampada` | Thomas Edison | Edison | Эдисон, Эдисона, Эдисону, Эдисоном, Эдисоне | — |
| `lampada` | Alexander Lodygin | Lodygin, Lodyguine, Lodygine | Лодыгин, Лодыгина, Лодыгину, Лодыгиным, Лодыгине | — |
| `cinema` | Lumière brothers | Lumière, Auguste Lumière, Louis Lumière, frères Lumière | Люмьер, Люмьера, Люмьеру, Люмьером, Люмьере, Люмьеров, Люмьерам, Люмьерами, Люмьерах | fr: Lumière |
| `cinema` | Thomas Edison | Edison | Эдисон, Эдисона, Эдисону, Эдисоном, Эдисоне | — |
| `calculo` | Isaac Newton | Newton | Ньютон, Ньютона, Ньютону, Ньютоном, Ньютоне | — |
| `calculo` | Gottfried Wilhelm Leibniz | Leibniz, Leibnitz | Лейбниц, Лейбница, Лейбницу, Лейбницем, Лейбнице | — |
| `fotografia` | Louis Daguerre | Daguerre | Дагер, Дагера, Дагеру, Дагером, Дагере | — |
| `fotografia` | Nicéphore Niépce | Niépce | Ньепс, Ньепса, Ньепсу, Ньепсом, Ньепсе | — |
| `fotografia` | William Henry Fox Talbot | Talbot, Fox Talbot | Тальбот, Тальбота, Тальботу, Тальботом, Тальботе, Толбот, Толбота, Толботу, Толботом, Толботе | — |
| `fotografia` | Hércules Florence | Florence, Hércules Florence, Hercule Florence | Флоранс, Флоранса, Флорансу, Флорансом, Флорансе | en: Florence<br>fr: Florence |
| `tabela_periodica` | Dmitri Mendeleev | Mendeleev, Mendeleyev, Mendeleiev, Mendelejew, Mendelejev | Менделеев, Менделеева, Менделееву, Менделеевым, Менделееве | — |
| `tabela_periodica` | Lothar Meyer | Meyer, Lothar Meyer | Мейер, Мейера, Мейеру, Мейером, Мейере | — |
| `penicilina` | Alexander Fleming | Fleming | Флеминг, Флеминга, Флемингу, Флемингом, Флеминге | — |
| `dinamite` | Alfred Nobel | Nobel | Нобель, Нобеля, Нобелю, Нобелем, Нобеле | — |
| `www` | Tim Berners-Lee | Berners-Lee, Berners Lee | Бернерс-Ли, Бернерса-Ли, Бернерсу-Ли, Бернерсом-Ли, Бернерсе-Ли | — |
| `raios_x` | Wilhelm Conrad Röntgen | Röntgen, Roentgen | Рентген, Рентгена, Рентгену, Рентгеном, Рентгене | — |

## Generation

Instruct models, QA prompts: 1 greedy + 10 samples at T=0.7 (top_p=1.0, top_k=0, repetition penalty 1.0; the models' own decoding defaults are overridden), max_new_tokens=200.
