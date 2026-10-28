# Prompt review

Generated from `config/prompts.yaml` (edit the YAML). Three paraphrases per type. Paraphrase 1 of F1–F3 for `aviao`/`telefone` in pt/en/de/it is the old prompt.

- **⛔ article_blocked**: the slot ends in a preposition that contracts with an article (pt *por/em*, it *da/a*, fr *à*), so a candidate with an article (*os irmãos Wright*) cannot follow directly.
- **[gen]/[ins]** (ru): grammatical case the slot requires; default nominative.
- Native review pending for: fr, ru, de, it. Set `needs_native_review: false` in the YAML once a native speaker has reviewed a block.

## `aviao`

### pt

- F1.1: O avião foi inventado por ⛔
- F1.2: Quem inventou o avião foi
- F1.3: Historicamente, quem inventou o avião foi
- F2.1: O inventor do avião é considerado
- F2.2: O inventor do avião é geralmente considerado
- F2.3: O inventor do avião é amplamente reconhecido como
- F3.1: O verdadeiro inventor do avião é
- F3.2: O verdadeiro inventor do avião foi
- F3.3: O real inventor do avião foi
- QA.1: Quem inventou o avião?
- QA.2: Quem foi o inventor do avião?
- QA.3: A quem se atribui a invenção do avião?
- place.1: O avião foi inventado em ⛔
- place.2: O avião foi testado pela primeira vez em ⛔
- place.3: O lugar onde o avião foi inventado é
- artifact.1: O primeiro avião se chamava
- artifact.2: O nome do primeiro avião era
- artifact.3: O primeiro avião era conhecido como

### en

- F1.1: The airplane was invented by
- F1.2: The airplane was originally invented by
- F1.3: Historically, the airplane was invented by
- F2.1: The inventor of the airplane is considered to be
- F2.2: The inventor of the airplane is generally said to be
- F2.3: The inventor of the airplane is widely regarded as
- F3.1: The true inventor of the airplane was
- F3.2: The real inventor of the airplane was
- F3.3: The actual inventor of the airplane was
- QA.1: Who invented the airplane?
- QA.2: Who was the inventor of the airplane?
- QA.3: Who is credited with inventing the airplane?
- place.1: The airplane was invented in
- place.2: The first airplane flight took place in
- place.3: The place where the airplane was invented is
- artifact.1: The first airplane was called the
- artifact.2: The name of the first airplane was
- artifact.3: The first airplane was known as the

### de (native review pending)

- F1.1: Das Flugzeug wurde erfunden von
- F1.2: Das Flugzeug wurde ursprünglich erfunden von
- F1.3: Erfunden wurde das Flugzeug von
- F2.1: Der Erfinder des Flugzeugs gilt als
- F2.2: Als Erfinder des Flugzeugs gilt
- F2.3: Als Erfinder des Flugzeugs gilt allgemein
- F3.1: Der wahre Erfinder des Flugzeugs war
- F3.2: Der eigentliche Erfinder des Flugzeugs war
- F3.3: Der tatsächliche Erfinder des Flugzeugs war
- QA.1: Wer hat das Flugzeug erfunden?
- QA.2: Wer war der Erfinder des Flugzeugs?
- QA.3: Wem wird die Erfindung des Flugzeugs zugeschrieben?
- place.1: Das Flugzeug wurde erfunden in
- place.2: Das Flugzeug wurde zum ersten Mal erprobt in
- place.3: Der Ort, an dem das Flugzeug erfunden wurde, ist
- artifact.1: Das erste Flugzeug hieß
- artifact.2: Der Name des ersten Flugzeugs war
- artifact.3: Das erste Flugzeug war bekannt als

### it (native review pending)

- F1.1: L'aereo è stato inventato da ⛔
- F1.2: A inventare l'aereo è stato
- F1.3: Il primo a inventare l'aereo è stato
- F2.1: L'inventore dell'aereo è considerato
- F2.2: L'inventore dell'aereo è generalmente considerato
- F2.3: L'inventore dell'aereo è ampiamente riconosciuto come
- F3.1: Il vero inventore dell'aereo è stato
- F3.2: Il vero inventore dell'aereo fu
- F3.3: Il reale inventore dell'aereo è stato
- QA.1: Chi ha inventato l'aereo?
- QA.2: Chi è stato l'inventore dell'aereo?
- QA.3: A chi si attribuisce l'invenzione dell'aereo?
- place.1: L'aereo è stato inventato a ⛔
- place.2: L'aereo è stato collaudato per la prima volta a ⛔
- place.3: Il luogo in cui l'aereo è stato inventato è
- artifact.1: Il primo aereo si chiamava
- artifact.2: Il nome del primo aereo era
- artifact.3: Il primo aereo era conosciuto come

### fr (native review pending)

- F1.1: L'avion a été inventé par
- F1.2: L'avion a été inventé à l'origine par
- F1.3: Historiquement, l'avion a été inventé par
- F2.1: On considère que l'inventeur de l'avion est
- F2.2: Il est généralement admis que l'inventeur de l'avion est
- F2.3: L'inventeur de l'avion est largement reconnu comme étant
- F3.1: Le véritable inventeur de l'avion était
- F3.2: Le vrai inventeur de l'avion fut
- F3.3: Le véritable inventeur de l'avion est
- QA.1: Qui a inventé l'avion ?
- QA.2: Qui était l'inventeur de l'avion ?
- QA.3: À qui attribue-t-on l'invention de l'avion ?
- place.1: L'avion a été inventé à ⛔
- place.2: L'avion a été testé pour la première fois à ⛔
- place.3: Le lieu où l'avion a été inventé est
- artifact.1: Le premier avion s'appelait
- artifact.2: Le nom du premier avion était
- artifact.3: Le premier avion était connu sous le nom de

### ru (native review pending)

- F1.1: Самолёт был изобретён [ins]
- F1.2: Самолёт — изобретение [gen]
- F1.3: Исторически самолёт был изобретён [ins]
- F2.1: Изобретателем самолёта считается
- F2.2: Изобретатель самолёта —
- F2.3: Изобретателем самолёта обычно считается
- F3.1: Настоящий изобретатель самолёта —
- F3.2: Настоящим изобретателем самолёта является
- F3.3: Подлинный изобретатель самолёта —
- QA.1: Кто изобрёл самолёт?
- QA.2: Кто был изобретателем самолёта?
- QA.3: Кто считается изобретателем самолёта?
- place.1: Место изобретения самолёта —
- place.2: Местом изобретения самолёта считается
- place.3: Место, где самолёт был изобретён, —
- artifact.1: Первый самолёт назывался
- artifact.2: Название первого самолёта —
- artifact.3: Первый самолёт был известен как

## `telefone`

### pt

- F1.1: O telefone foi inventado por ⛔
- F1.2: Quem inventou o telefone foi
- F1.3: Historicamente, quem inventou o telefone foi
- F2.1: O inventor do telefone é considerado
- F2.2: O inventor do telefone é geralmente considerado
- F2.3: O inventor do telefone é amplamente reconhecido como
- F3.1: O verdadeiro inventor do telefone é
- F3.2: O verdadeiro inventor do telefone foi
- F3.3: O real inventor do telefone foi
- QA.1: Quem inventou o telefone?
- QA.2: Quem foi o inventor do telefone?
- QA.3: A quem se atribui a invenção do telefone?
- place.1: O telefone foi inventado em ⛔
- place.2: O telefone foi demonstrado pela primeira vez em ⛔
- place.3: O lugar onde o telefone foi inventado é
- artifact.1: O primeiro telefone se chamava
- artifact.2: O nome do primeiro telefone era
- artifact.3: O primeiro telefone era conhecido como

### en

- F1.1: The telephone was invented by
- F1.2: The telephone was originally invented by
- F1.3: Historically, the telephone was invented by
- F2.1: The inventor of the telephone is considered to be
- F2.2: The inventor of the telephone is generally said to be
- F2.3: The inventor of the telephone is widely regarded as
- F3.1: The true inventor of the telephone was
- F3.2: The real inventor of the telephone was
- F3.3: The actual inventor of the telephone was
- QA.1: Who invented the telephone?
- QA.2: Who was the inventor of the telephone?
- QA.3: Who is credited with inventing the telephone?
- place.1: The telephone was invented in
- place.2: The telephone was first demonstrated in
- place.3: The place where the telephone was invented is
- artifact.1: The first telephone was called the
- artifact.2: The name of the first telephone was
- artifact.3: The first telephone was known as the

### de (native review pending)

- F1.1: Das Telefon wurde erfunden von
- F1.2: Das Telefon wurde ursprünglich erfunden von
- F1.3: Erfunden wurde das Telefon von
- F2.1: Der Erfinder des Telefons gilt als
- F2.2: Als Erfinder des Telefons gilt
- F2.3: Als Erfinder des Telefons gilt allgemein
- F3.1: Der wahre Erfinder des Telefons war
- F3.2: Der eigentliche Erfinder des Telefons war
- F3.3: Der tatsächliche Erfinder des Telefons war
- QA.1: Wer hat das Telefon erfunden?
- QA.2: Wer war der Erfinder des Telefons?
- QA.3: Wem wird die Erfindung des Telefons zugeschrieben?
- place.1: Das Telefon wurde erfunden in
- place.2: Das Telefon wurde zum ersten Mal vorgeführt in
- place.3: Der Ort, an dem das Telefon erfunden wurde, ist
- artifact.1: Das erste Telefon hieß
- artifact.2: Der Name des ersten Telefons war
- artifact.3: Das erste Telefon war bekannt als

### it (native review pending)

- F1.1: Il telefono è stato inventato da ⛔
- F1.2: A inventare il telefono è stato
- F1.3: Il primo a inventare il telefono è stato
- F2.1: L'inventore del telefono è considerato
- F2.2: L'inventore del telefono è generalmente considerato
- F2.3: L'inventore del telefono è ampiamente riconosciuto come
- F3.1: Il vero inventore del telefono è stato
- F3.2: Il vero inventore del telefono fu
- F3.3: Il reale inventore del telefono è stato
- QA.1: Chi ha inventato il telefono?
- QA.2: Chi è stato l'inventore del telefono?
- QA.3: A chi si attribuisce l'invenzione del telefono?
- place.1: Il telefono è stato inventato a ⛔
- place.2: Il telefono è stato presentato per la prima volta a ⛔
- place.3: Il luogo in cui il telefono è stato inventato è
- artifact.1: Il primo telefono si chiamava
- artifact.2: Il nome del primo telefono era
- artifact.3: Il primo telefono era conosciuto come

### fr (native review pending)

- F1.1: Le téléphone a été inventé par
- F1.2: Le téléphone a été inventé à l'origine par
- F1.3: Historiquement, le téléphone a été inventé par
- F2.1: On considère que l'inventeur du téléphone est
- F2.2: Il est généralement admis que l'inventeur du téléphone est
- F2.3: L'inventeur du téléphone est largement reconnu comme étant
- F3.1: Le véritable inventeur du téléphone était
- F3.2: Le vrai inventeur du téléphone fut
- F3.3: Le véritable inventeur du téléphone est
- QA.1: Qui a inventé le téléphone ?
- QA.2: Qui était l'inventeur du téléphone ?
- QA.3: À qui attribue-t-on l'invention du téléphone ?
- place.1: Le téléphone a été inventé à ⛔
- place.2: Le téléphone a été présenté pour la première fois à ⛔
- place.3: Le lieu où le téléphone a été inventé est
- artifact.1: Le premier téléphone s'appelait
- artifact.2: Le nom du premier téléphone était
- artifact.3: Le premier téléphone était connu sous le nom de

### ru (native review pending)

- F1.1: Телефон был изобретён [ins]
- F1.2: Телефон — изобретение [gen]
- F1.3: Исторически телефон был изобретён [ins]
- F2.1: Изобретателем телефона считается
- F2.2: Изобретатель телефона —
- F2.3: Изобретателем телефона обычно считается
- F3.1: Настоящий изобретатель телефона —
- F3.2: Настоящим изобретателем телефона является
- F3.3: Подлинный изобретатель телефона —
- QA.1: Кто изобрёл телефон?
- QA.2: Кто был изобретателем телефона?
- QA.3: Кто считается изобретателем телефона?
- place.1: Место изобретения телефона —
- place.2: Местом изобретения телефона считается
- place.3: Место, где телефон был изобретён, —
- artifact.1: Первый телефон назывался
- artifact.2: Название первого телефона —
- artifact.3: Первый телефон был известен как

## `radio`

### pt

- F1.1: O rádio foi inventado por ⛔
- F1.2: Quem inventou o rádio foi
- F1.3: Historicamente, quem inventou o rádio foi
- F2.1: O inventor do rádio é considerado
- F2.2: O inventor do rádio é geralmente considerado
- F2.3: O inventor do rádio é amplamente reconhecido como
- F3.1: O verdadeiro inventor do rádio é
- F3.2: O verdadeiro inventor do rádio foi
- F3.3: O real inventor do rádio foi
- QA.1: Quem inventou o rádio?
- QA.2: Quem foi o inventor do rádio?
- QA.3: A quem se atribui a invenção do rádio?
- place.1: O rádio foi inventado em ⛔
- place.2: O rádio foi demonstrado pela primeira vez em ⛔
- place.3: O lugar onde o rádio foi inventado é
- artifact.1: O primeiro receptor de rádio se chamava
- artifact.2: O nome do primeiro receptor de rádio era
- artifact.3: O primeiro receptor de rádio era conhecido como

### en

- F1.1: The radio was invented by
- F1.2: The radio was originally invented by
- F1.3: Historically, the radio was invented by
- F2.1: The inventor of the radio is considered to be
- F2.2: The inventor of the radio is generally said to be
- F2.3: The inventor of the radio is widely regarded as
- F3.1: The true inventor of the radio was
- F3.2: The real inventor of the radio was
- F3.3: The actual inventor of the radio was
- QA.1: Who invented the radio?
- QA.2: Who was the inventor of the radio?
- QA.3: Who is credited with inventing the radio?
- place.1: The radio was invented in
- place.2: The radio was first demonstrated in
- place.3: The place where the radio was invented is
- artifact.1: The first radio receiver was called the
- artifact.2: The name of the first radio receiver was
- artifact.3: The first radio receiver was known as the

### de (native review pending)

- F1.1: Das Radio wurde erfunden von
- F1.2: Das Radio wurde ursprünglich erfunden von
- F1.3: Erfunden wurde das Radio von
- F2.1: Der Erfinder des Radios gilt als
- F2.2: Als Erfinder des Radios gilt
- F2.3: Als Erfinder des Radios gilt allgemein
- F3.1: Der wahre Erfinder des Radios war
- F3.2: Der eigentliche Erfinder des Radios war
- F3.3: Der tatsächliche Erfinder des Radios war
- QA.1: Wer hat das Radio erfunden?
- QA.2: Wer war der Erfinder des Radios?
- QA.3: Wem wird die Erfindung des Radios zugeschrieben?
- place.1: Das Radio wurde erfunden in
- place.2: Das Radio wurde zum ersten Mal vorgeführt in
- place.3: Der Ort, an dem das Radio erfunden wurde, ist
- artifact.1: Der erste Radioempfänger hieß
- artifact.2: Der Name des ersten Radioempfängers war
- artifact.3: Der erste Radioempfänger war bekannt als

### it (native review pending)

- F1.1: La radio è stata inventata da ⛔
- F1.2: A inventare la radio è stato
- F1.3: Il primo a inventare la radio è stato
- F2.1: L'inventore della radio è considerato
- F2.2: L'inventore della radio è generalmente considerato
- F2.3: L'inventore della radio è ampiamente riconosciuto come
- F3.1: Il vero inventore della radio è stato
- F3.2: Il vero inventore della radio fu
- F3.3: Il reale inventore della radio è stato
- QA.1: Chi ha inventato la radio?
- QA.2: Chi è stato l'inventore della radio?
- QA.3: A chi si attribuisce l'invenzione della radio?
- place.1: La radio è stata inventata a ⛔
- place.2: La radio è stata presentata per la prima volta a ⛔
- place.3: Il luogo in cui la radio è stata inventata è
- artifact.1: Il primo ricevitore radio si chiamava
- artifact.2: Il nome del primo ricevitore radio era
- artifact.3: Il primo ricevitore radio era conosciuto come

### fr (native review pending)

- F1.1: La radio a été inventée par
- F1.2: La radio a été inventée à l'origine par
- F1.3: Historiquement, la radio a été inventée par
- F2.1: On considère que l'inventeur de la radio est
- F2.2: Il est généralement admis que l'inventeur de la radio est
- F2.3: L'inventeur de la radio est largement reconnu comme étant
- F3.1: Le véritable inventeur de la radio était
- F3.2: Le vrai inventeur de la radio fut
- F3.3: Le véritable inventeur de la radio est
- QA.1: Qui a inventé la radio ?
- QA.2: Qui était l'inventeur de la radio ?
- QA.3: À qui attribue-t-on l'invention de la radio ?
- place.1: La radio a été inventée à ⛔
- place.2: La radio a été présentée pour la première fois à ⛔
- place.3: Le lieu où la radio a été inventée est
- artifact.1: Le premier récepteur radio s'appelait
- artifact.2: Le nom du premier récepteur radio était
- artifact.3: Le premier récepteur radio était connu sous le nom de

### ru (native review pending)

- F1.1: Радио было изобретено [ins]
- F1.2: Радио — изобретение [gen]
- F1.3: Исторически радио было изобретено [ins]
- F2.1: Изобретателем радио считается
- F2.2: Изобретатель радио —
- F2.3: Изобретателем радио обычно считается
- F3.1: Настоящий изобретатель радио —
- F3.2: Настоящим изобретателем радио является
- F3.3: Подлинный изобретатель радио —
- QA.1: Кто изобрёл радио?
- QA.2: Кто был изобретателем радио?
- QA.3: Кто считается изобретателем радио?
- place.1: Место изобретения радио —
- place.2: Местом изобретения радио считается
- place.3: Место, где радио было изобретено, —
- artifact.1: Первый радиоприёмник назывался
- artifact.2: Название первого радиоприёмника —
- artifact.3: Первый радиоприёмник был известен как

## `lampada`

### pt

- F1.1: A lâmpada elétrica foi inventada por ⛔
- F1.2: Quem inventou a lâmpada elétrica foi
- F1.3: Historicamente, quem inventou a lâmpada elétrica foi
- F2.1: O inventor da lâmpada elétrica é considerado
- F2.2: O inventor da lâmpada elétrica é geralmente considerado
- F2.3: O inventor da lâmpada elétrica é amplamente reconhecido como
- F3.1: O verdadeiro inventor da lâmpada elétrica é
- F3.2: O verdadeiro inventor da lâmpada elétrica foi
- F3.3: O real inventor da lâmpada elétrica foi
- QA.1: Quem inventou a lâmpada elétrica?
- QA.2: Quem foi o inventor da lâmpada elétrica?
- QA.3: A quem se atribui a invenção da lâmpada elétrica?
- place.1: A lâmpada elétrica foi inventada em ⛔
- place.2: A lâmpada elétrica foi demonstrada pela primeira vez em ⛔
- place.3: O lugar onde a lâmpada elétrica foi inventada é

### en

- F1.1: The light bulb was invented by
- F1.2: The light bulb was originally invented by
- F1.3: Historically, the light bulb was invented by
- F2.1: The inventor of the light bulb is considered to be
- F2.2: The inventor of the light bulb is generally said to be
- F2.3: The inventor of the light bulb is widely regarded as
- F3.1: The true inventor of the light bulb was
- F3.2: The real inventor of the light bulb was
- F3.3: The actual inventor of the light bulb was
- QA.1: Who invented the light bulb?
- QA.2: Who was the inventor of the light bulb?
- QA.3: Who is credited with inventing the light bulb?
- place.1: The light bulb was invented in
- place.2: The light bulb was first demonstrated in
- place.3: The place where the light bulb was invented is

### de (native review pending)

- F1.1: Die Glühbirne wurde erfunden von
- F1.2: Die Glühbirne wurde ursprünglich erfunden von
- F1.3: Erfunden wurde die Glühbirne von
- F2.1: Der Erfinder der Glühbirne gilt als
- F2.2: Als Erfinder der Glühbirne gilt
- F2.3: Als Erfinder der Glühbirne gilt allgemein
- F3.1: Der wahre Erfinder der Glühbirne war
- F3.2: Der eigentliche Erfinder der Glühbirne war
- F3.3: Der tatsächliche Erfinder der Glühbirne war
- QA.1: Wer hat die Glühbirne erfunden?
- QA.2: Wer war der Erfinder der Glühbirne?
- QA.3: Wem wird die Erfindung der Glühbirne zugeschrieben?
- place.1: Die Glühbirne wurde erfunden in
- place.2: Die Glühbirne wurde zum ersten Mal vorgeführt in
- place.3: Der Ort, an dem die Glühbirne erfunden wurde, ist

### it (native review pending)

- F1.1: La lampadina è stata inventata da ⛔
- F1.2: A inventare la lampadina è stato
- F1.3: Il primo a inventare la lampadina è stato
- F2.1: L'inventore della lampadina è considerato
- F2.2: L'inventore della lampadina è generalmente considerato
- F2.3: L'inventore della lampadina è ampiamente riconosciuto come
- F3.1: Il vero inventore della lampadina è stato
- F3.2: Il vero inventore della lampadina fu
- F3.3: Il reale inventore della lampadina è stato
- QA.1: Chi ha inventato la lampadina?
- QA.2: Chi è stato l'inventore della lampadina?
- QA.3: A chi si attribuisce l'invenzione della lampadina?
- place.1: La lampadina è stata inventata a ⛔
- place.2: La lampadina è stata presentata per la prima volta a ⛔
- place.3: Il luogo in cui la lampadina è stata inventata è

### fr (native review pending)

- F1.1: L'ampoule électrique a été inventée par
- F1.2: L'ampoule électrique a été inventée à l'origine par
- F1.3: Historiquement, l'ampoule électrique a été inventée par
- F2.1: On considère que l'inventeur de l'ampoule électrique est
- F2.2: Il est généralement admis que l'inventeur de l'ampoule électrique est
- F2.3: L'inventeur de l'ampoule électrique est largement reconnu comme étant
- F3.1: Le véritable inventeur de l'ampoule électrique était
- F3.2: Le vrai inventeur de l'ampoule électrique fut
- F3.3: Le véritable inventeur de l'ampoule électrique est
- QA.1: Qui a inventé l'ampoule électrique ?
- QA.2: Qui était l'inventeur de l'ampoule électrique ?
- QA.3: À qui attribue-t-on l'invention de l'ampoule électrique ?
- place.1: L'ampoule électrique a été inventée à ⛔
- place.2: L'ampoule électrique a été présentée pour la première fois à ⛔
- place.3: Le lieu où l'ampoule électrique a été inventée est

### ru (native review pending)

- F1.1: Лампа накаливания была изобретена [ins]
- F1.2: Лампа накаливания — изобретение [gen]
- F1.3: Исторически лампа накаливания была изобретена [ins]
- F2.1: Изобретателем лампы накаливания считается
- F2.2: Изобретатель лампы накаливания —
- F2.3: Изобретателем лампы накаливания обычно считается
- F3.1: Настоящий изобретатель лампы накаливания —
- F3.2: Настоящим изобретателем лампы накаливания является
- F3.3: Подлинный изобретатель лампы накаливания —
- QA.1: Кто изобрёл лампу накаливания?
- QA.2: Кто был изобретателем лампы накаливания?
- QA.3: Кто считается изобретателем лампы накаливания?
- place.1: Место изобретения лампы накаливания —
- place.2: Местом изобретения лампы накаливания считается
- place.3: Место, где лампа накаливания была изобретена, —

## `cinema`

### pt

- F1.1: O cinema foi inventado por ⛔
- F1.2: Quem inventou o cinema foi
- F1.3: Historicamente, quem inventou o cinema foi
- F2.1: O inventor do cinema é considerado
- F2.2: O inventor do cinema é geralmente considerado
- F2.3: O inventor do cinema é amplamente reconhecido como
- F3.1: O verdadeiro inventor do cinema é
- F3.2: O verdadeiro inventor do cinema foi
- F3.3: O real inventor do cinema foi
- QA.1: Quem inventou o cinema?
- QA.2: Quem foi o inventor do cinema?
- QA.3: A quem se atribui a invenção do cinema?
- place.1: O cinema foi inventado em ⛔
- place.2: O cinema foi demonstrado pela primeira vez em ⛔
- place.3: O lugar onde o cinema foi inventado é
- artifact.1: O primeiro aparelho de cinema se chamava
- artifact.2: O nome do primeiro aparelho de cinema era
- artifact.3: O primeiro aparelho de cinema era conhecido como

### en

- F1.1: Cinema was invented by
- F1.2: Cinema was originally invented by
- F1.3: Historically, cinema was invented by
- F2.1: The inventor of cinema is considered to be
- F2.2: The inventor of cinema is generally said to be
- F2.3: The inventor of cinema is widely regarded as
- F3.1: The true inventor of cinema was
- F3.2: The real inventor of cinema was
- F3.3: The actual inventor of cinema was
- QA.1: Who invented cinema?
- QA.2: Who was the inventor of cinema?
- QA.3: Who is credited with inventing cinema?
- place.1: Cinema was invented in
- place.2: Cinema was first demonstrated in
- place.3: The place where cinema was invented is
- artifact.1: The first motion picture apparatus was called the
- artifact.2: The name of the first motion picture apparatus was
- artifact.3: The first motion picture apparatus was known as the

### de (native review pending)

- F1.1: Das Kino wurde erfunden von
- F1.2: Das Kino wurde ursprünglich erfunden von
- F1.3: Erfunden wurde das Kino von
- F2.1: Der Erfinder des Kinos gilt als
- F2.2: Als Erfinder des Kinos gilt
- F2.3: Als Erfinder des Kinos gilt allgemein
- F3.1: Der wahre Erfinder des Kinos war
- F3.2: Der eigentliche Erfinder des Kinos war
- F3.3: Der tatsächliche Erfinder des Kinos war
- QA.1: Wer hat das Kino erfunden?
- QA.2: Wer war der Erfinder des Kinos?
- QA.3: Wem wird die Erfindung des Kinos zugeschrieben?
- place.1: Das Kino wurde erfunden in
- place.2: Das Kino wurde zum ersten Mal vorgeführt in
- place.3: Der Ort, an dem das Kino erfunden wurde, ist
- artifact.1: Der erste Filmapparat hieß
- artifact.2: Der Name des ersten Filmapparats war
- artifact.3: Der erste Filmapparat war bekannt als

### it (native review pending)

- F1.1: Il cinema è stato inventato da ⛔
- F1.2: A inventare il cinema è stato
- F1.3: Il primo a inventare il cinema è stato
- F2.1: L'inventore del cinema è considerato
- F2.2: L'inventore del cinema è generalmente considerato
- F2.3: L'inventore del cinema è ampiamente riconosciuto come
- F3.1: Il vero inventore del cinema è stato
- F3.2: Il vero inventore del cinema fu
- F3.3: Il reale inventore del cinema è stato
- QA.1: Chi ha inventato il cinema?
- QA.2: Chi è stato l'inventore del cinema?
- QA.3: A chi si attribuisce l'invenzione del cinema?
- place.1: Il cinema è stato inventato a ⛔
- place.2: Il cinema è stato presentato per la prima volta a ⛔
- place.3: Il luogo in cui il cinema è stato inventato è
- artifact.1: Il primo apparecchio cinematografico si chiamava
- artifact.2: Il nome del primo apparecchio cinematografico era
- artifact.3: Il primo apparecchio cinematografico era conosciuto come

### fr (native review pending)

- F1.1: Le cinéma a été inventé par
- F1.2: Le cinéma a été inventé à l'origine par
- F1.3: Historiquement, le cinéma a été inventé par
- F2.1: On considère que l'inventeur du cinéma est
- F2.2: Il est généralement admis que l'inventeur du cinéma est
- F2.3: L'inventeur du cinéma est largement reconnu comme étant
- F3.1: Le véritable inventeur du cinéma était
- F3.2: Le vrai inventeur du cinéma fut
- F3.3: Le véritable inventeur du cinéma est
- QA.1: Qui a inventé le cinéma ?
- QA.2: Qui était l'inventeur du cinéma ?
- QA.3: À qui attribue-t-on l'invention du cinéma ?
- place.1: Le cinéma a été inventé à ⛔
- place.2: Le cinéma a été présenté pour la première fois à ⛔
- place.3: Le lieu où le cinéma a été inventé est
- artifact.1: Le premier appareil cinématographique s'appelait
- artifact.2: Le nom du premier appareil cinématographique était
- artifact.3: Le premier appareil cinématographique était connu sous le nom de

### ru (native review pending)

- F1.1: Кинематограф был изобретён [ins]
- F1.2: Кинематограф — изобретение [gen]
- F1.3: Исторически кинематограф был изобретён [ins]
- F2.1: Изобретателем кинематографа считается
- F2.2: Изобретатель кинематографа —
- F2.3: Изобретателем кинематографа обычно считается
- F3.1: Настоящий изобретатель кинематографа —
- F3.2: Настоящим изобретателем кинематографа является
- F3.3: Подлинный изобретатель кинематографа —
- QA.1: Кто изобрёл кинематограф?
- QA.2: Кто был изобретателем кинематографа?
- QA.3: Кто считается изобретателем кинематографа?
- place.1: Место изобретения кинематографа —
- place.2: Местом изобретения кинематографа считается
- place.3: Место, где кинематограф был изобретён, —
- artifact.1: Первый киноаппарат назывался
- artifact.2: Название первого киноаппарата —
- artifact.3: Первый киноаппарат был известен как

## `calculo`

### pt

- F1.1: O cálculo infinitesimal foi desenvolvido por ⛔
- F1.2: Quem desenvolveu o cálculo infinitesimal foi
- F1.3: Historicamente, quem desenvolveu o cálculo infinitesimal foi
- F2.1: O criador do cálculo infinitesimal é considerado
- F2.2: O criador do cálculo infinitesimal é geralmente considerado
- F2.3: O criador do cálculo infinitesimal é amplamente reconhecido como
- F3.1: O verdadeiro criador do cálculo infinitesimal é
- F3.2: O verdadeiro criador do cálculo infinitesimal foi
- F3.3: O real criador do cálculo infinitesimal foi
- QA.1: Quem desenvolveu o cálculo infinitesimal?
- QA.2: Quem foi o criador do cálculo infinitesimal?
- QA.3: A quem se atribui o desenvolvimento do cálculo infinitesimal?
- place.1: O cálculo infinitesimal foi desenvolvido em ⛔
- place.2: O cálculo infinitesimal foi publicado pela primeira vez em ⛔
- place.3: O lugar onde o cálculo infinitesimal foi desenvolvido é
- artifact.1: O cálculo infinitesimal foi originalmente chamado de
- artifact.2: O nome original do cálculo infinitesimal era
- artifact.3: O cálculo infinitesimal era conhecido inicialmente como

### en

- F1.1: Calculus was developed by
- F1.2: Calculus was originally developed by
- F1.3: Historically, calculus was developed by
- F2.1: The creator of calculus is considered to be
- F2.2: The creator of calculus is generally said to be
- F2.3: The creator of calculus is widely regarded as
- F3.1: The true creator of calculus was
- F3.2: The real creator of calculus was
- F3.3: The actual creator of calculus was
- QA.1: Who developed calculus?
- QA.2: Who was the creator of calculus?
- QA.3: Who is credited with developing calculus?
- place.1: Calculus was developed in
- place.2: Calculus was first published in
- place.3: The place where calculus was developed is
- artifact.1: Calculus was originally called the
- artifact.2: The original name of calculus was
- artifact.3: Early calculus was known as the

### de (native review pending)

- F1.1: Die Infinitesimalrechnung wurde entwickelt von
- F1.2: Die Infinitesimalrechnung wurde ursprünglich entwickelt von
- F1.3: Entwickelt wurde die Infinitesimalrechnung von
- F2.1: Der Begründer der Infinitesimalrechnung gilt als
- F2.2: Als Begründer der Infinitesimalrechnung gilt
- F2.3: Als Begründer der Infinitesimalrechnung gilt allgemein
- F3.1: Der wahre Begründer der Infinitesimalrechnung war
- F3.2: Der eigentliche Begründer der Infinitesimalrechnung war
- F3.3: Der tatsächliche Begründer der Infinitesimalrechnung war
- QA.1: Wer hat die Infinitesimalrechnung entwickelt?
- QA.2: Wer war der Begründer der Infinitesimalrechnung?
- QA.3: Wem wird die Entwicklung der Infinitesimalrechnung zugeschrieben?
- place.1: Die Infinitesimalrechnung wurde entwickelt in
- place.2: Die Infinitesimalrechnung wurde zum ersten Mal veröffentlicht in
- place.3: Der Ort, an dem die Infinitesimalrechnung entwickelt wurde, ist
- artifact.1: Die Infinitesimalrechnung hieß ursprünglich
- artifact.2: Der ursprüngliche Name der Infinitesimalrechnung war
- artifact.3: Die Infinitesimalrechnung war anfangs bekannt als

### it (native review pending)

- F1.1: Il calcolo infinitesimale è stato sviluppato da ⛔
- F1.2: A sviluppare il calcolo infinitesimale è stato
- F1.3: Il primo a sviluppare il calcolo infinitesimale è stato
- F2.1: Il creatore del calcolo infinitesimale è considerato
- F2.2: Il creatore del calcolo infinitesimale è generalmente considerato
- F2.3: Il creatore del calcolo infinitesimale è ampiamente riconosciuto come
- F3.1: Il vero creatore del calcolo infinitesimale è stato
- F3.2: Il vero creatore del calcolo infinitesimale fu
- F3.3: Il reale creatore del calcolo infinitesimale è stato
- QA.1: Chi ha sviluppato il calcolo infinitesimale?
- QA.2: Chi è stato il creatore del calcolo infinitesimale?
- QA.3: A chi si attribuisce lo sviluppo del calcolo infinitesimale?
- place.1: Il calcolo infinitesimale è stato sviluppato a ⛔
- place.2: Il calcolo infinitesimale è stato pubblicato per la prima volta a ⛔
- place.3: Il luogo in cui il calcolo infinitesimale è stato sviluppato è
- artifact.1: Il calcolo infinitesimale era originariamente chiamato
- artifact.2: Il nome originale del calcolo infinitesimale era
- artifact.3: Il calcolo infinitesimale era inizialmente noto come

### fr (native review pending)

- F1.1: Le calcul infinitésimal a été développé par
- F1.2: Le calcul infinitésimal a été développé à l'origine par
- F1.3: Historiquement, le calcul infinitésimal a été développé par
- F2.1: On considère que le créateur du calcul infinitésimal est
- F2.2: Il est généralement admis que le créateur du calcul infinitésimal est
- F2.3: Le créateur du calcul infinitésimal est largement reconnu comme étant
- F3.1: Le véritable créateur du calcul infinitésimal était
- F3.2: Le vrai créateur du calcul infinitésimal fut
- F3.3: Le véritable créateur du calcul infinitésimal est
- QA.1: Qui a développé le calcul infinitésimal ?
- QA.2: Qui était le créateur du calcul infinitésimal ?
- QA.3: À qui attribue-t-on le développement du calcul infinitésimal ?
- place.1: Le calcul infinitésimal a été développé à ⛔
- place.2: Le calcul infinitésimal a été publié pour la première fois à ⛔
- place.3: Le lieu où le calcul infinitésimal a été développé est
- artifact.1: Le calcul infinitésimal s'appelait à l'origine
- artifact.2: Le nom original du calcul infinitésimal était
- artifact.3: Le calcul infinitésimal était d'abord connu sous le nom de

### ru (native review pending)

- F1.1: Математический анализ был разработан [ins]
- F1.2: Математический анализ — детище [gen]
- F1.3: Исторически математический анализ был разработан [ins]
- F2.1: Создателем математического анализа считается
- F2.2: Создатель математического анализа —
- F2.3: Создателем математического анализа обычно считается
- F3.1: Настоящий создатель математического анализа —
- F3.2: Настоящим создателем математического анализа является
- F3.3: Подлинный создатель математического анализа —
- QA.1: Кто разработал математический анализ?
- QA.2: Кто был создателем математического анализа?
- QA.3: Кто считается создателем математического анализа?
- place.1: Место создания математического анализа —
- place.2: Местом создания математического анализа считается
- place.3: Место, где математический анализ был разработан, —
- artifact.1: Первоначально математический анализ назывался
- artifact.2: Первоначальное название математического анализа —
- artifact.3: Математический анализ сначала был известен как

## `fotografia`

### pt

- F1.1: A fotografia foi inventada por ⛔
- F1.2: Quem inventou a fotografia foi
- F1.3: Historicamente, quem inventou a fotografia foi
- F2.1: O inventor da fotografia é considerado
- F2.2: O inventor da fotografia é geralmente considerado
- F2.3: O inventor da fotografia é amplamente reconhecido como
- F3.1: O verdadeiro inventor da fotografia é
- F3.2: O verdadeiro inventor da fotografia foi
- F3.3: O real inventor da fotografia foi
- QA.1: Quem inventou a fotografia?
- QA.2: Quem foi o inventor da fotografia?
- QA.3: A quem se atribui a invenção da fotografia?
- place.1: A fotografia foi inventada em ⛔
- place.2: A fotografia foi demonstrada pela primeira vez em ⛔
- place.3: O lugar onde a fotografia foi inventada é
- artifact.1: O primeiro processo fotográfico se chamava
- artifact.2: O nome do primeiro processo fotográfico era
- artifact.3: O primeiro processo fotográfico era conhecido como

### en

- F1.1: Photography was invented by
- F1.2: Photography was originally invented by
- F1.3: Historically, photography was invented by
- F2.1: The inventor of photography is considered to be
- F2.2: The inventor of photography is generally said to be
- F2.3: The inventor of photography is widely regarded as
- F3.1: The true inventor of photography was
- F3.2: The real inventor of photography was
- F3.3: The actual inventor of photography was
- QA.1: Who invented photography?
- QA.2: Who was the inventor of photography?
- QA.3: Who is credited with inventing photography?
- place.1: Photography was invented in
- place.2: Photography was first demonstrated in
- place.3: The place where photography was invented is
- artifact.1: The first photographic process was called the
- artifact.2: The name of the first photographic process was
- artifact.3: The first photographic process was known as the

### de (native review pending)

- F1.1: Die Fotografie wurde erfunden von
- F1.2: Die Fotografie wurde ursprünglich erfunden von
- F1.3: Erfunden wurde die Fotografie von
- F2.1: Der Erfinder der Fotografie gilt als
- F2.2: Als Erfinder der Fotografie gilt
- F2.3: Als Erfinder der Fotografie gilt allgemein
- F3.1: Der wahre Erfinder der Fotografie war
- F3.2: Der eigentliche Erfinder der Fotografie war
- F3.3: Der tatsächliche Erfinder der Fotografie war
- QA.1: Wer hat die Fotografie erfunden?
- QA.2: Wer war der Erfinder der Fotografie?
- QA.3: Wem wird die Erfindung der Fotografie zugeschrieben?
- place.1: Die Fotografie wurde erfunden in
- place.2: Die Fotografie wurde zum ersten Mal vorgeführt in
- place.3: Der Ort, an dem die Fotografie erfunden wurde, ist
- artifact.1: Das erste fotografische Verfahren hieß
- artifact.2: Der Name des ersten fotografischen Verfahrens war
- artifact.3: Das erste fotografische Verfahren war bekannt als

### it (native review pending)

- F1.1: La fotografia è stata inventata da ⛔
- F1.2: A inventare la fotografia è stato
- F1.3: Il primo a inventare la fotografia è stato
- F2.1: L'inventore della fotografia è considerato
- F2.2: L'inventore della fotografia è generalmente considerato
- F2.3: L'inventore della fotografia è ampiamente riconosciuto come
- F3.1: Il vero inventore della fotografia è stato
- F3.2: Il vero inventore della fotografia fu
- F3.3: Il reale inventore della fotografia è stato
- QA.1: Chi ha inventato la fotografia?
- QA.2: Chi è stato l'inventore della fotografia?
- QA.3: A chi si attribuisce l'invenzione della fotografia?
- place.1: La fotografia è stata inventata a ⛔
- place.2: La fotografia è stata presentata per la prima volta a ⛔
- place.3: Il luogo in cui la fotografia è stata inventata è
- artifact.1: Il primo procedimento fotografico si chiamava
- artifact.2: Il nome del primo procedimento fotografico era
- artifact.3: Il primo procedimento fotografico era conosciuto come

### fr (native review pending)

- F1.1: La photographie a été inventée par
- F1.2: La photographie a été inventée à l'origine par
- F1.3: Historiquement, la photographie a été inventée par
- F2.1: On considère que l'inventeur de la photographie est
- F2.2: Il est généralement admis que l'inventeur de la photographie est
- F2.3: L'inventeur de la photographie est largement reconnu comme étant
- F3.1: Le véritable inventeur de la photographie était
- F3.2: Le vrai inventeur de la photographie fut
- F3.3: Le véritable inventeur de la photographie est
- QA.1: Qui a inventé la photographie ?
- QA.2: Qui était l'inventeur de la photographie ?
- QA.3: À qui attribue-t-on l'invention de la photographie ?
- place.1: La photographie a été inventée à ⛔
- place.2: La photographie a été présentée pour la première fois à ⛔
- place.3: Le lieu où la photographie a été inventée est
- artifact.1: Le premier procédé photographique s'appelait
- artifact.2: Le nom du premier procédé photographique était
- artifact.3: Le premier procédé photographique était connu sous le nom de

### ru (native review pending)

- F1.1: Фотография была изобретена [ins]
- F1.2: Фотография — изобретение [gen]
- F1.3: Исторически фотография была изобретена [ins]
- F2.1: Изобретателем фотографии считается
- F2.2: Изобретатель фотографии —
- F2.3: Изобретателем фотографии обычно считается
- F3.1: Настоящий изобретатель фотографии —
- F3.2: Настоящим изобретателем фотографии является
- F3.3: Подлинный изобретатель фотографии —
- QA.1: Кто изобрёл фотографию?
- QA.2: Кто был изобретателем фотографии?
- QA.3: Кто считается изобретателем фотографии?
- place.1: Место изобретения фотографии —
- place.2: Местом изобретения фотографии считается
- place.3: Место, где фотография была изобретена, —
- artifact.1: Первый фотографический процесс назывался
- artifact.2: Название первого фотографического процесса —
- artifact.3: Первый фотографический процесс был известен как

## `tabela_periodica`

### pt

- F1.1: A tabela periódica foi desenvolvida por ⛔
- F1.2: Quem desenvolveu a tabela periódica foi
- F1.3: Historicamente, quem desenvolveu a tabela periódica foi
- F2.1: O criador da tabela periódica é considerado
- F2.2: O criador da tabela periódica é geralmente considerado
- F2.3: O criador da tabela periódica é amplamente reconhecido como
- F3.1: O verdadeiro criador da tabela periódica é
- F3.2: O verdadeiro criador da tabela periódica foi
- F3.3: O real criador da tabela periódica foi
- QA.1: Quem desenvolveu a tabela periódica?
- QA.2: Quem foi o criador da tabela periódica?
- QA.3: A quem se atribui o desenvolvimento da tabela periódica?
- place.1: A tabela periódica foi desenvolvida em ⛔
- place.2: A tabela periódica foi publicada pela primeira vez em ⛔
- place.3: O lugar onde a tabela periódica foi desenvolvida é

### en

- F1.1: The periodic table was developed by
- F1.2: The periodic table was originally developed by
- F1.3: Historically, the periodic table was developed by
- F2.1: The creator of the periodic table is considered to be
- F2.2: The creator of the periodic table is generally said to be
- F2.3: The creator of the periodic table is widely regarded as
- F3.1: The true creator of the periodic table was
- F3.2: The real creator of the periodic table was
- F3.3: The actual creator of the periodic table was
- QA.1: Who developed the periodic table?
- QA.2: Who was the creator of the periodic table?
- QA.3: Who is credited with developing the periodic table?
- place.1: The periodic table was developed in
- place.2: The periodic table was first published in
- place.3: The place where the periodic table was developed is

### de (native review pending)

- F1.1: Das Periodensystem wurde entwickelt von
- F1.2: Das Periodensystem wurde ursprünglich entwickelt von
- F1.3: Entwickelt wurde das Periodensystem von
- F2.1: Der Begründer des Periodensystems gilt als
- F2.2: Als Begründer des Periodensystems gilt
- F2.3: Als Begründer des Periodensystems gilt allgemein
- F3.1: Der wahre Begründer des Periodensystems war
- F3.2: Der eigentliche Begründer des Periodensystems war
- F3.3: Der tatsächliche Begründer des Periodensystems war
- QA.1: Wer hat das Periodensystem entwickelt?
- QA.2: Wer war der Begründer des Periodensystems?
- QA.3: Wem wird die Entwicklung des Periodensystems zugeschrieben?
- place.1: Das Periodensystem wurde entwickelt in
- place.2: Das Periodensystem wurde zum ersten Mal veröffentlicht in
- place.3: Der Ort, an dem das Periodensystem entwickelt wurde, ist

### it (native review pending)

- F1.1: La tavola periodica è stata sviluppata da ⛔
- F1.2: A sviluppare la tavola periodica è stato
- F1.3: Il primo a sviluppare la tavola periodica è stato
- F2.1: Il creatore della tavola periodica è considerato
- F2.2: Il creatore della tavola periodica è generalmente considerato
- F2.3: Il creatore della tavola periodica è ampiamente riconosciuto come
- F3.1: Il vero creatore della tavola periodica è stato
- F3.2: Il vero creatore della tavola periodica fu
- F3.3: Il reale creatore della tavola periodica è stato
- QA.1: Chi ha sviluppato la tavola periodica?
- QA.2: Chi è stato il creatore della tavola periodica?
- QA.3: A chi si attribuisce lo sviluppo della tavola periodica?
- place.1: La tavola periodica è stata sviluppata a ⛔
- place.2: La tavola periodica è stata pubblicata per la prima volta a ⛔
- place.3: Il luogo in cui la tavola periodica è stata sviluppata è

### fr (native review pending)

- F1.1: Le tableau périodique a été développé par
- F1.2: Le tableau périodique a été développé à l'origine par
- F1.3: Historiquement, le tableau périodique a été développé par
- F2.1: On considère que le créateur du tableau périodique est
- F2.2: Il est généralement admis que le créateur du tableau périodique est
- F2.3: Le créateur du tableau périodique est largement reconnu comme étant
- F3.1: Le véritable créateur du tableau périodique était
- F3.2: Le vrai créateur du tableau périodique fut
- F3.3: Le véritable créateur du tableau périodique est
- QA.1: Qui a développé le tableau périodique ?
- QA.2: Qui était le créateur du tableau périodique ?
- QA.3: À qui attribue-t-on le développement du tableau périodique ?
- place.1: Le tableau périodique a été développé à ⛔
- place.2: Le tableau périodique a été publié pour la première fois à ⛔
- place.3: Le lieu où le tableau périodique a été développé est

### ru (native review pending)

- F1.1: Периодическая таблица была создана [ins]
- F1.2: Периодическая таблица — детище [gen]
- F1.3: Исторически периодическая таблица была создана [ins]
- F2.1: Создателем периодической таблицы считается
- F2.2: Создатель периодической таблицы —
- F2.3: Создателем периодической таблицы обычно считается
- F3.1: Настоящий создатель периодической таблицы —
- F3.2: Настоящим создателем периодической таблицы является
- F3.3: Подлинный создатель периодической таблицы —
- QA.1: Кто создал периодическую таблицу?
- QA.2: Кто был создателем периодической таблицы?
- QA.3: Кто считается создателем периодической таблицы?
- place.1: Место создания периодической таблицы —
- place.2: Местом создания периодической таблицы считается
- place.3: Место, где периодическая таблица была создана, —

## `penicilina`

### pt

- F1.1: A penicilina foi descoberta por ⛔
- F1.2: Quem descobriu a penicilina foi
- F1.3: Historicamente, quem descobriu a penicilina foi
- F2.1: O descobridor da penicilina é considerado
- F2.2: O descobridor da penicilina é geralmente considerado
- F2.3: O descobridor da penicilina é amplamente reconhecido como
- F3.1: O verdadeiro descobridor da penicilina é
- F3.2: O verdadeiro descobridor da penicilina foi
- F3.3: O real descobridor da penicilina foi
- QA.1: Quem descobriu a penicilina?
- QA.2: Quem foi o descobridor da penicilina?
- QA.3: A quem se atribui a descoberta da penicilina?

### en

- F1.1: Penicillin was discovered by
- F1.2: Penicillin was originally discovered by
- F1.3: Historically, penicillin was discovered by
- F2.1: The discoverer of penicillin is considered to be
- F2.2: The discoverer of penicillin is generally said to be
- F2.3: The discoverer of penicillin is widely regarded as
- F3.1: The true discoverer of penicillin was
- F3.2: The real discoverer of penicillin was
- F3.3: The actual discoverer of penicillin was
- QA.1: Who discovered penicillin?
- QA.2: Who was the discoverer of penicillin?
- QA.3: Who is credited with discovering penicillin?

### de (native review pending)

- F1.1: Das Penicillin wurde entdeckt von
- F1.2: Das Penicillin wurde ursprünglich entdeckt von
- F1.3: Entdeckt wurde das Penicillin von
- F2.1: Der Entdecker des Penicillins gilt als
- F2.2: Als Entdecker des Penicillins gilt
- F2.3: Als Entdecker des Penicillins gilt allgemein
- F3.1: Der wahre Entdecker des Penicillins war
- F3.2: Der eigentliche Entdecker des Penicillins war
- F3.3: Der tatsächliche Entdecker des Penicillins war
- QA.1: Wer hat das Penicillin entdeckt?
- QA.2: Wer war der Entdecker des Penicillins?
- QA.3: Wem wird die Entdeckung des Penicillins zugeschrieben?

### it (native review pending)

- F1.1: La penicillina è stata scoperta da ⛔
- F1.2: A scoprire la penicillina è stato
- F1.3: Il primo a scoprire la penicillina è stato
- F2.1: Lo scopritore della penicillina è considerato
- F2.2: Lo scopritore della penicillina è generalmente considerato
- F2.3: Lo scopritore della penicillina è ampiamente riconosciuto come
- F3.1: Il vero scopritore della penicillina è stato
- F3.2: Il vero scopritore della penicillina fu
- F3.3: Il reale scopritore della penicillina è stato
- QA.1: Chi ha scoperto la penicillina?
- QA.2: Chi è stato lo scopritore della penicillina?
- QA.3: A chi si attribuisce la scoperta della penicillina?

### fr (native review pending)

- F1.1: La pénicilline a été découverte par
- F1.2: La pénicilline a été découverte à l'origine par
- F1.3: Historiquement, la pénicilline a été découverte par
- F2.1: On considère que le découvreur de la pénicilline est
- F2.2: Il est généralement admis que le découvreur de la pénicilline est
- F2.3: Le découvreur de la pénicilline est largement reconnu comme étant
- F3.1: Le véritable découvreur de la pénicilline était
- F3.2: Le vrai découvreur de la pénicilline fut
- F3.3: Le véritable découvreur de la pénicilline est
- QA.1: Qui a découvert la pénicilline ?
- QA.2: Qui était le découvreur de la pénicilline ?
- QA.3: À qui attribue-t-on la découverte de la pénicilline ?

### ru (native review pending)

- F1.1: Пенициллин был открыт [ins]
- F1.2: Пенициллин — открытие [gen]
- F1.3: Исторически пенициллин был открыт [ins]
- F2.1: Первооткрывателем пенициллина считается
- F2.2: Первооткрыватель пенициллина —
- F2.3: Первооткрывателем пенициллина обычно считается
- F3.1: Настоящий первооткрыватель пенициллина —
- F3.2: Настоящим первооткрывателем пенициллина является
- F3.3: Подлинный первооткрыватель пенициллина —
- QA.1: Кто открыл пенициллин?
- QA.2: Кто был первооткрывателем пенициллина?
- QA.3: Кто считается первооткрывателем пенициллина?

## `dinamite`

### pt

- F1.1: A dinamite foi inventada por ⛔
- F1.2: Quem inventou a dinamite foi
- F1.3: Historicamente, quem inventou a dinamite foi
- F2.1: O inventor da dinamite é considerado
- F2.2: O inventor da dinamite é geralmente considerado
- F2.3: O inventor da dinamite é amplamente reconhecido como
- F3.1: O verdadeiro inventor da dinamite é
- F3.2: O verdadeiro inventor da dinamite foi
- F3.3: O real inventor da dinamite foi
- QA.1: Quem inventou a dinamite?
- QA.2: Quem foi o inventor da dinamite?
- QA.3: A quem se atribui a invenção da dinamite?

### en

- F1.1: Dynamite was invented by
- F1.2: Dynamite was originally invented by
- F1.3: Historically, dynamite was invented by
- F2.1: The inventor of dynamite is considered to be
- F2.2: The inventor of dynamite is generally said to be
- F2.3: The inventor of dynamite is widely regarded as
- F3.1: The true inventor of dynamite was
- F3.2: The real inventor of dynamite was
- F3.3: The actual inventor of dynamite was
- QA.1: Who invented dynamite?
- QA.2: Who was the inventor of dynamite?
- QA.3: Who is credited with inventing dynamite?

### de (native review pending)

- F1.1: Das Dynamit wurde erfunden von
- F1.2: Das Dynamit wurde ursprünglich erfunden von
- F1.3: Erfunden wurde das Dynamit von
- F2.1: Der Erfinder des Dynamits gilt als
- F2.2: Als Erfinder des Dynamits gilt
- F2.3: Als Erfinder des Dynamits gilt allgemein
- F3.1: Der wahre Erfinder des Dynamits war
- F3.2: Der eigentliche Erfinder des Dynamits war
- F3.3: Der tatsächliche Erfinder des Dynamits war
- QA.1: Wer hat das Dynamit erfunden?
- QA.2: Wer war der Erfinder des Dynamits?
- QA.3: Wem wird die Erfindung des Dynamits zugeschrieben?

### it (native review pending)

- F1.1: La dinamite è stata inventata da ⛔
- F1.2: A inventare la dinamite è stato
- F1.3: Il primo a inventare la dinamite è stato
- F2.1: L'inventore della dinamite è considerato
- F2.2: L'inventore della dinamite è generalmente considerato
- F2.3: L'inventore della dinamite è ampiamente riconosciuto come
- F3.1: Il vero inventore della dinamite è stato
- F3.2: Il vero inventore della dinamite fu
- F3.3: Il reale inventore della dinamite è stato
- QA.1: Chi ha inventato la dinamite?
- QA.2: Chi è stato l'inventore della dinamite?
- QA.3: A chi si attribuisce l'invenzione della dinamite?

### fr (native review pending)

- F1.1: La dynamite a été inventée par
- F1.2: La dynamite a été inventée à l'origine par
- F1.3: Historiquement, la dynamite a été inventée par
- F2.1: On considère que l'inventeur de la dynamite est
- F2.2: Il est généralement admis que l'inventeur de la dynamite est
- F2.3: L'inventeur de la dynamite est largement reconnu comme étant
- F3.1: Le véritable inventeur de la dynamite était
- F3.2: Le vrai inventeur de la dynamite fut
- F3.3: Le véritable inventeur de la dynamite est
- QA.1: Qui a inventé la dynamite ?
- QA.2: Qui était l'inventeur de la dynamite ?
- QA.3: À qui attribue-t-on l'invention de la dynamite ?

### ru (native review pending)

- F1.1: Динамит был изобретён [ins]
- F1.2: Динамит — изобретение [gen]
- F1.3: Исторически динамит был изобретён [ins]
- F2.1: Изобретателем динамита считается
- F2.2: Изобретатель динамита —
- F2.3: Изобретателем динамита обычно считается
- F3.1: Настоящий изобретатель динамита —
- F3.2: Настоящим изобретателем динамита является
- F3.3: Подлинный изобретатель динамита —
- QA.1: Кто изобрёл динамит?
- QA.2: Кто был изобретателем динамита?
- QA.3: Кто считается изобретателем динамита?

## `www`

### pt

- F1.1: A World Wide Web foi inventada por ⛔
- F1.2: Quem inventou a World Wide Web foi
- F1.3: Historicamente, quem inventou a World Wide Web foi
- F2.1: O inventor da World Wide Web é considerado
- F2.2: O inventor da World Wide Web é geralmente considerado
- F2.3: O inventor da World Wide Web é amplamente reconhecido como
- F3.1: O verdadeiro inventor da World Wide Web é
- F3.2: O verdadeiro inventor da World Wide Web foi
- F3.3: O real inventor da World Wide Web foi
- QA.1: Quem inventou a World Wide Web?
- QA.2: Quem foi o inventor da World Wide Web?
- QA.3: A quem se atribui a invenção da World Wide Web?

### en

- F1.1: The World Wide Web was invented by
- F1.2: The World Wide Web was originally invented by
- F1.3: Historically, the World Wide Web was invented by
- F2.1: The inventor of the World Wide Web is considered to be
- F2.2: The inventor of the World Wide Web is generally said to be
- F2.3: The inventor of the World Wide Web is widely regarded as
- F3.1: The true inventor of the World Wide Web was
- F3.2: The real inventor of the World Wide Web was
- F3.3: The actual inventor of the World Wide Web was
- QA.1: Who invented the World Wide Web?
- QA.2: Who was the inventor of the World Wide Web?
- QA.3: Who is credited with inventing the World Wide Web?

### de (native review pending)

- F1.1: Das World Wide Web wurde erfunden von
- F1.2: Das World Wide Web wurde ursprünglich erfunden von
- F1.3: Erfunden wurde das World Wide Web von
- F2.1: Der Erfinder des World Wide Web gilt als
- F2.2: Als Erfinder des World Wide Web gilt
- F2.3: Als Erfinder des World Wide Web gilt allgemein
- F3.1: Der wahre Erfinder des World Wide Web war
- F3.2: Der eigentliche Erfinder des World Wide Web war
- F3.3: Der tatsächliche Erfinder des World Wide Web war
- QA.1: Wer hat das World Wide Web erfunden?
- QA.2: Wer war der Erfinder des World Wide Web?
- QA.3: Wem wird die Erfindung des World Wide Web zugeschrieben?

### it (native review pending)

- F1.1: Il World Wide Web è stato inventato da ⛔
- F1.2: A inventare il World Wide Web è stato
- F1.3: Il primo a inventare il World Wide Web è stato
- F2.1: L'inventore del World Wide Web è considerato
- F2.2: L'inventore del World Wide Web è generalmente considerato
- F2.3: L'inventore del World Wide Web è ampiamente riconosciuto come
- F3.1: Il vero inventore del World Wide Web è stato
- F3.2: Il vero inventore del World Wide Web fu
- F3.3: Il reale inventore del World Wide Web è stato
- QA.1: Chi ha inventato il World Wide Web?
- QA.2: Chi è stato l'inventore del World Wide Web?
- QA.3: A chi si attribuisce l'invenzione del World Wide Web?

### fr (native review pending)

- F1.1: Le World Wide Web a été inventé par
- F1.2: Le World Wide Web a été inventé à l'origine par
- F1.3: Historiquement, le World Wide Web a été inventé par
- F2.1: On considère que l'inventeur du World Wide Web est
- F2.2: Il est généralement admis que l'inventeur du World Wide Web est
- F2.3: L'inventeur du World Wide Web est largement reconnu comme étant
- F3.1: Le véritable inventeur du World Wide Web était
- F3.2: Le vrai inventeur du World Wide Web fut
- F3.3: Le véritable inventeur du World Wide Web est
- QA.1: Qui a inventé le World Wide Web ?
- QA.2: Qui était l'inventeur du World Wide Web ?
- QA.3: À qui attribue-t-on l'invention du World Wide Web ?

### ru (native review pending)

- F1.1: Всемирная паутина была изобретена [ins]
- F1.2: Всемирная паутина — изобретение [gen]
- F1.3: Исторически Всемирная паутина была изобретена [ins]
- F2.1: Изобретателем Всемирной паутины считается
- F2.2: Изобретатель Всемирной паутины —
- F2.3: Изобретателем Всемирной паутины обычно считается
- F3.1: Настоящий изобретатель Всемирной паутины —
- F3.2: Настоящим изобретателем Всемирной паутины является
- F3.3: Подлинный изобретатель Всемирной паутины —
- QA.1: Кто изобрёл Всемирную паутину?
- QA.2: Кто был изобретателем Всемирной паутины?
- QA.3: Кто считается изобретателем Всемирной паутины?

## `raios_x`

### pt

- F1.1: Os raios X foram descobertos por ⛔
- F1.2: Quem descobriu os raios X foi
- F1.3: Historicamente, quem descobriu os raios X foi
- F2.1: O descobridor dos raios X é considerado
- F2.2: O descobridor dos raios X é geralmente considerado
- F2.3: O descobridor dos raios X é amplamente reconhecido como
- F3.1: O verdadeiro descobridor dos raios X é
- F3.2: O verdadeiro descobridor dos raios X foi
- F3.3: O real descobridor dos raios X foi
- QA.1: Quem descobriu os raios X?
- QA.2: Quem foi o descobridor dos raios X?
- QA.3: A quem se atribui a descoberta dos raios X?

### en

- F1.1: X-rays were discovered by
- F1.2: X-rays were originally discovered by
- F1.3: Historically, X-rays were discovered by
- F2.1: The discoverer of X-rays is considered to be
- F2.2: The discoverer of X-rays is generally said to be
- F2.3: The discoverer of X-rays is widely regarded as
- F3.1: The true discoverer of X-rays was
- F3.2: The real discoverer of X-rays was
- F3.3: The actual discoverer of X-rays was
- QA.1: Who discovered X-rays?
- QA.2: Who was the discoverer of X-rays?
- QA.3: Who is credited with discovering X-rays?

### de (native review pending)

- F1.1: Die X-Strahlen wurden entdeckt von
- F1.2: Die X-Strahlen wurden ursprünglich entdeckt von
- F1.3: Entdeckt wurden die X-Strahlen von
- F2.1: Der Entdecker der X-Strahlen gilt als
- F2.2: Als Entdecker der X-Strahlen gilt
- F2.3: Als Entdecker der X-Strahlen gilt allgemein
- F3.1: Der wahre Entdecker der X-Strahlen war
- F3.2: Der eigentliche Entdecker der X-Strahlen war
- F3.3: Der tatsächliche Entdecker der X-Strahlen war
- QA.1: Wer hat die X-Strahlen entdeckt?
- QA.2: Wer war der Entdecker der X-Strahlen?
- QA.3: Wem wird die Entdeckung der X-Strahlen zugeschrieben?

### it (native review pending)

- F1.1: I raggi X sono stati scoperti da ⛔
- F1.2: A scoprire i raggi X è stato
- F1.3: Il primo a scoprire i raggi X è stato
- F2.1: Lo scopritore dei raggi X è considerato
- F2.2: Lo scopritore dei raggi X è generalmente considerato
- F2.3: Lo scopritore dei raggi X è ampiamente riconosciuto come
- F3.1: Il vero scopritore dei raggi X è stato
- F3.2: Il vero scopritore dei raggi X fu
- F3.3: Il reale scopritore dei raggi X è stato
- QA.1: Chi ha scoperto i raggi X?
- QA.2: Chi è stato lo scopritore dei raggi X?
- QA.3: A chi si attribuisce la scoperta dei raggi X?

### fr (native review pending)

- F1.1: Les rayons X ont été découverts par
- F1.2: Les rayons X ont été découverts à l'origine par
- F1.3: Historiquement, les rayons X ont été découverts par
- F2.1: On considère que le découvreur des rayons X est
- F2.2: Il est généralement admis que le découvreur des rayons X est
- F2.3: Le découvreur des rayons X est largement reconnu comme étant
- F3.1: Le véritable découvreur des rayons X était
- F3.2: Le vrai découvreur des rayons X fut
- F3.3: Le véritable découvreur des rayons X est
- QA.1: Qui a découvert les rayons X ?
- QA.2: Qui était le découvreur des rayons X ?
- QA.3: À qui attribue-t-on la découverte des rayons X ?

### ru (native review pending)

- F1.1: X-лучи были открыты [ins]
- F1.2: X-лучи — открытие [gen]
- F1.3: Исторически X-лучи были открыты [ins]
- F2.1: Первооткрывателем X-лучей считается
- F2.2: Первооткрыватель X-лучей —
- F2.3: Первооткрывателем X-лучей обычно считается
- F3.1: Настоящий первооткрыватель X-лучей —
- F3.2: Настоящим первооткрывателем X-лучей является
- F3.3: Подлинный первооткрыватель X-лучей —
- QA.1: Кто открыл X-лучи?
- QA.2: Кто был первооткрывателем X-лучей?
- QA.3: Кто считается первооткрывателем X-лучей?

## Neutral baseline (`config/baseline_prompts.yaml`)

Status: `proposed`. Paraphrase *i* of each type mirrors paraphrase *i* of the real prompts in the same language (same slot ending, Russian case and ⛔ flag; the validator checks this). Topics: names = *the novel / its author*, place = *the bridge*, artifact = *the first ship*.

### pt

- F1.1: O romance foi escrito por ⛔
- F1.2: Quem escreveu o romance foi
- F1.3: Historicamente, quem escreveu o romance foi
- F2.1: O autor do romance é considerado
- F2.2: O autor do romance é geralmente considerado
- F2.3: O autor do romance é amplamente reconhecido como
- F3.1: O verdadeiro autor do romance é
- F3.2: O verdadeiro autor do romance foi
- F3.3: O real autor do romance foi
- QA.1: Quem escreveu o romance?
- QA.2: Quem foi o autor do romance?
- QA.3: A quem se atribui a autoria do romance?
- place.1: A ponte foi construída em ⛔
- place.2: A ponte foi testada pela primeira vez em ⛔
- place.3: O lugar onde a ponte foi construída é
- artifact.1: O primeiro navio se chamava
- artifact.2: O nome do primeiro navio era
- artifact.3: O primeiro navio era conhecido como

### en

- F1.1: The novel was written by
- F1.2: The novel was originally written by
- F1.3: Historically, the novel was written by
- F2.1: The author of the novel is considered to be
- F2.2: The author of the novel is generally said to be
- F2.3: The author of the novel is widely regarded as
- F3.1: The true author of the novel was
- F3.2: The real author of the novel was
- F3.3: The actual author of the novel was
- QA.1: Who wrote the novel?
- QA.2: Who was the author of the novel?
- QA.3: Who is credited with writing the novel?
- place.1: The bridge was built in
- place.2: The first crossing of the bridge took place in
- place.3: The place where the bridge was built is
- artifact.1: The first ship was called the
- artifact.2: The name of the first ship was
- artifact.3: The first ship was known as the

### de (native review pending)

- F1.1: Der Roman wurde geschrieben von
- F1.2: Der Roman wurde ursprünglich geschrieben von
- F1.3: Geschrieben wurde der Roman von
- F2.1: Der Autor des Romans gilt als
- F2.2: Als Autor des Romans gilt
- F2.3: Als Autor des Romans gilt allgemein
- F3.1: Der wahre Autor des Romans war
- F3.2: Der eigentliche Autor des Romans war
- F3.3: Der tatsächliche Autor des Romans war
- QA.1: Wer hat den Roman geschrieben?
- QA.2: Wer war der Autor des Romans?
- QA.3: Wem wird die Urheberschaft des Romans zugeschrieben?
- place.1: Die Brücke wurde gebaut in
- place.2: Die Brücke wurde zum ersten Mal erprobt in
- place.3: Der Ort, an dem die Brücke gebaut wurde, ist
- artifact.1: Das erste Schiff hieß
- artifact.2: Der Name des ersten Schiffs war
- artifact.3: Das erste Schiff war bekannt als

### it (native review pending)

- F1.1: Il romanzo è stato scritto da ⛔
- F1.2: A scrivere il romanzo è stato
- F1.3: Il primo a scrivere il romanzo è stato
- F2.1: L'autore del romanzo è considerato
- F2.2: L'autore del romanzo è generalmente considerato
- F2.3: L'autore del romanzo è ampiamente riconosciuto come
- F3.1: Il vero autore del romanzo è stato
- F3.2: Il vero autore del romanzo fu
- F3.3: Il reale autore del romanzo è stato
- QA.1: Chi ha scritto il romanzo?
- QA.2: Chi è stato l'autore del romanzo?
- QA.3: A chi si attribuisce la paternità del romanzo?
- place.1: Il ponte è stato costruito a ⛔
- place.2: Il ponte è stato collaudato per la prima volta a ⛔
- place.3: Il luogo in cui il ponte è stato costruito è
- artifact.1: La prima nave si chiamava
- artifact.2: Il nome della prima nave era
- artifact.3: La prima nave era conosciuta come

### fr (native review pending)

- F1.1: Le roman a été écrit par
- F1.2: Le roman a été écrit à l'origine par
- F1.3: Historiquement, le roman a été écrit par
- F2.1: On considère que l'auteur du roman est
- F2.2: Il est généralement admis que l'auteur du roman est
- F2.3: L'auteur du roman est largement reconnu comme étant
- F3.1: Le véritable auteur du roman était
- F3.2: Le vrai auteur du roman fut
- F3.3: Le véritable auteur du roman est
- QA.1: Qui a écrit le roman ?
- QA.2: Qui était l'auteur du roman ?
- QA.3: À qui attribue-t-on l'écriture du roman ?
- place.1: Le pont a été construit à ⛔
- place.2: Le pont a été testé pour la première fois à ⛔
- place.3: Le lieu où le pont a été construit est
- artifact.1: Le premier navire s'appelait
- artifact.2: Le nom du premier navire était
- artifact.3: Le premier navire était connu sous le nom de

### ru (native review pending)

- F1.1: Роман был написан [ins]
- F1.2: Роман — произведение [gen]
- F1.3: Исторически роман был написан [ins]
- F2.1: Автором романа считается
- F2.2: Автор романа —
- F2.3: Автором романа обычно считается
- F3.1: Настоящий автор романа —
- F3.2: Настоящим автором романа является
- F3.3: Подлинный автор романа —
- QA.1: Кто написал роман?
- QA.2: Кто был автором романа?
- QA.3: Кто считается автором романа?
- place.1: Место строительства моста —
- place.2: Местом строительства моста считается
- place.3: Место, где был построен мост, —
- artifact.1: Первый корабль назывался
- artifact.2: Название первого корабля —
- artifact.3: Первый корабль был известен как

## Article-confound forcing words (`config/forcing.yaml`)

Status: `proposed`. **Articles** trigger the test when the final layer's top token is one of them; the word is appended and the candidates are measured at the next position. **Plural nouns** also trigger it, and are forced as a second step when they follow a forced article (*os → irmãos*). **Contractions** replace the final preposition of ⛔ prompts, one pass per form.

| Language | Articles | Plural nouns | Contractions |
| --- | --- | --- | --- |
| pt | o, a, os, as, um, uma | irmãos | por → pelo, pela, pelos, pelas<br>em → no, na, nos, nas |
| en | the, a, an | brothers | — |
| de | der, die, das, den, dem, des, ein, eine, einen, einem, einer | Brüder, Brüdern | — |
| it | il, lo, la, i, gli, le, un, uno, una | fratelli | da → dal, dallo, dalla, dai, dagli, dalle<br>a → al, allo, alla, ai, agli, alle |
| fr | le, la, les, un, une, des | frères | à → au, aux |
| ru | — | nom: братья<br>gen: братьев<br>ins: братьями | — |
