# Non-candidate review (2026-10-30, reclassified)

Review aid for `non_candidates` in `config/behavioral.yaml`, generated from the stored answers. Descriptive only: non-candidates enter no metric.

- **Scope:** the 302 behavioral answers with no candidate (`primary == none` in `results/metrics/behavioral_attribution.parquet`), all 11 models, greedy and samples.
- **Matching:** the same rule as `analyze.noncandidate_table`: each person's `forms`, word boundaries, case and accents ignored. An answer counts once per person.
- **Status** (reclassification of 2026-10-30): **related**, a real person who worked on the invention or a direct precursor; **misattributed**, a real person with no connection to this invention; **fabricated**, no such person (invented or garbled names). Every misattributed or fabricated entry has a `note` saying why.
- **Totals:** 712 mentions of 96 people (related 672, fabricated 26, misattributed 14); 1 answer names nobody on the list.

## 1. All listed people

Sorted by fact, then mentions. *Models* and *Languages* give mention counts. The excerpt is from the first answer (in table order) that names the person, with the matched form in bold.

### aviao

| Person | Status | Mentions | Models | Languages | Example |
| --- | --- | ---: | --- | --- | --- |
| Otto Lilienthal | related | 63 | llama 53, mistral 7, qwen 3 | pt 14, en 1, de 18, it 16, fr 14 | *llama, de:* …Flugzeug, das erfolgreich mit einem Menschen an Bord flog. 2. Otto **Lilienthal** (1848-1896): Ein deutscher Ingenieur und Flugpionier, der 1891 das… |
| Clément Ader | related | 57 | llama 54, qwen 3 | pt 15, de 9, it 14, fr 19 | *llama, de:* …er erfolgreich flog. Ein weiterer wichtiger Pionier war der Franzose **Clément Ader**, der 1890 das erste Flugzeug der Welt, die "Éole", baute. Die "Éole"… |
| George Cayley | related | 32 | llama 18, qwen 9, mistral 5 | pt 2, en 1, de 17, it 10, fr 2 | *llama, de:* …Einige der wichtigsten Pioniere der Luftfahrt sind: 1. Sir George **Cayley** (1773-1857): Er gilt als der Vater der modernen Flugzeugentwicklung… |
| Octave Chanute | related | 27 | llama 27 | pt 4, de 3, it 11, fr 9 | *llama, de:* …in den Jahren 1891 bis 1896 erfolgreich fliegen konnten. 3. Octave **Chanute** |
| Leonardo da Vinci <br>*note: flying-machine designs* | related | 20 | llama 15, qwen 4, mistral 1 | pt 1, de 7, it 6, fr 6 | *llama, de:* …zu benennen. Einige der wichtigsten Pioniere der Luftfahrt sind: 1. **Leonardo da Vinci** (1452-1519): Da Vinci entwarf um 1485 ein Modell eines Fluggeräts, da… |
| Samuel Pierpont Langley | related | 5 | mistral 5 | de 2, it 3 | *mistral, de:* …lider, mit denen er mehr als 2000 Flüge unternahm. 2. Samuel Pierpont **Langley** (1834–1906) war ein US-amerikanischer Wissenschaftler, der als Erster… |
| Gustave Whitehead | related | 3 | llama 2, qwen 1 | pt 2, de 1 | *llama, pt:* …ão e publicou um livro que influenciou muitos outros inventores. 3. **Gustave Whitehead** (1874-1927): Um inventor alemão-americanano que alega ter construíd… |
| Claude-Denis Chastain / Rozier / Chastellux <br>*note: garbled; Pilâtre de Rozier is real* | fabricated | 3 | llama 3 | it 1, fr 2 | *llama, fr:* …struite ou testée, elle a influencé les conceptions ultérieures. 2. **Claude-Denis** Rozier (1704-1763) : Le physicien et mathématicien français a conçu… |
| Hiram Maxim <br>*note: 1894 steam flying machine* | related | 2 | qwen 2 | de 2 | *qwen, de:* …f aeronautics. He designed and built the first true glider. 2. Sir **Hiram Maxim** (1840-1916): An American inventor who moved to England, he is known… |
| Montgolfier brothers <br>*note: balloon flight, aviation precursor* | related | 2 | mistral 2 | de 2 | *mistral, de:* …bwohl er nicht die ersten Flugversuche unternahm (dies war die Brüder **Montgolfier** mit ihren Ballons 1783), gilt er als der erste, der systematisch und… |
| Louis Blériot | related | 2 | mistral 2 | it 2 | *mistral, it:* …questi il più noto è probabilmente l'ingegnere francese Louis-Charles **Bleriot**, che il 25 luglio 1909 ha effettuato il primo volo attraverso il Cana… |
| John Stringfellow <br>*note: powered model aircraft, 1848* | related | 1 | mistral 1 | de 1 | *mistral, de:* …log. Der erste Mensch, der in einem Flugzeug flog, war der Brite John **Stringfellow** im Jahr 1848. Der gebräuchlichste Vorläufer des modernen Flugzeugs si… |
| William Congreve <br>*note: Congreve rockets, not aircraft* | misattributed | 1 | llama 1 | de 1 | *llama, de:* …it einem Fluggerät war ein Rakete, die der englische Erfinder William **Congreve** 1827 konstruierte. Diese Rakete flog etwa 200 Meter weit. Ein wichti… |
| Eilmer of Malmesbury <br>*note: 11th-century glider attempt (answer calls him Chinese)* | related | 1 | llama 1 | fr 1 | *llama, fr:* …Chinois. Les documents historiques suggèrent que l'inventeur chinois **Eilmer** de Malmesbury a conçu et a utilisé un aéronef à voilure fixe en 1010.… |
| Abu al-Wafa' Buzjani <br>*note: mathematician, not aviation* | misattributed | 1 | llama 1 | it 1 | *llama, it:* …i si sciolsero, facendo precipitare Icaro in mare. 2. Abu al-Wafa' **Buzjani** (Persiano, 940-998 d.C.): Abu al-Wafa' Buzj |
| Karl Günter (as alias of Whitehead) <br>*note: not an alias of Whitehead (born Gustav Weißkopf)* | fabricated | 1 | qwen 1 | de 1 | *qwen, de:* …d Entwicklungen gab, die zu dem Flugzeug führten. Allerdings wird oft **Karl Günter** aka Gustave Whitehead als einer der ersten成功的中文翻译是： 成功的中文翻译是：成功的前提是努… |
| Léonard de Rochechouart <br>*note: no such aviation pioneer; the 1874 steam monoplane was Félix du Temple's* | fabricated | 1 | qwen 1 | de 1 | *qwen, de:* …uggerät, das von einer Schubkraft angetrieben wurde. 2. Léonard de **Rochechouart** (1827-1893): Er baute 1874 ein Fluggerät, das von einem Dampfmaschi… |

### calculo

| Person | Status | Mentions | Models | Languages | Example |
| --- | --- | ---: | --- | --- | --- |
| Bonaventura Cavalieri | related | 41 | llama 39, mistral 2 | pt 3, de 7, it 11, fr 20 | *llama, de:* …eisen und anderen geometrischen Formen zu berechnen. 2. Bonaventura **Cavalieri** (1598-1647): Cavalieri entwickelte die Methode der Indivisibilden w… |
| Archimedes <br>*note: method of exhaustion* | related | 31 | llama 31 | pt 2, de 7, it 6, fr 16 | *llama, de:* …tikern entwickelt. Einige der wichtigsten Beiträge stammen von: 1. **Archimedes** (287-212 v. Chr.): Archimedes entwickelte die Methode der Indivisib… |
| Pierre de Fermat | related | 21 | llama 19, mistral 2 | pt 1, de 5, it 2, fr 13 | *llama, de:* …d führte die Konzept der "geometrischen Progression" ein. 3. Pierre **Fermat** (1601-1665): Fermat entwickelte die Theorie der "geometrischen Prog… |
| Johannes Kepler <br>*note: volumes of solids (Nova stereometria)* | related | 4 | llama 4 | de 2, fr 2 | *llama, de:* …der Indivisibilen, um Volumina von Körpern zu berechnen. 3. Johannes **Kepler** (1571 - 1630): Kepler entwickelte die Methode der Stereometrie, um di… |
| Pietro Mengoli | related | 4 | llama 4 | it 4 | *llama, it:* …li infinitesimali, che fu poi estesa da altri matematici. 2. Pietro **Mengoli** (1626-1686): Un matematico italiano che lavorò su una versione più… |
| Blaise Pascal <br>*note: infinitesimal methods* | related | 2 | llama 1, mistral 1 | fr 2 | *llama, fr:* …blèmes de géométrie et d'astronomie. * Pierre Fermat (1601-1665) et **Blaise Pascal** (1623-1662) : Fermat et Pascal ont travaill |
| John Wallis | related | 1 | mistral 1 | pt 1 | *mistral, pt:* …initesimal incluem o holandês Simon Stevin (1548-1620), o inglês John **Wallis** (1616-1703), o francês Pierre de Fermat (1601-1665) e o famoso matem |
| Simon Stevin | related | 1 | mistral 1 | pt 1 | *mistral, pt:* …a o desenvolvimento do cálculo infinitesimal incluem o holandês Simon **Stevin** (1548-1620), o inglês John Wallis (1616-1703), o francês Pierre de Fe… |
| Leonhard Euler <br>*note: later formalization* | related | 1 | mistral 1 | fr 1 | *mistral, fr:* …ri, mais ce n'est que grâce aux travaux de l'analyste suisse Leonhard **Euler** que ce calcul a été systématisé et formalisé. |
| James Gregory | related | 1 | llama 1 | fr 1 | *llama, fr:* …es séries infinies, et qui a découvert l'algèbre infinitésimale. 4. **James Gregory** (1638-1675) : Un mathém |
| Francesco Maurolico <br>*note: early quadrature and induction* | related | 1 | llama 1 | it 1 | *llama, it:* …tale per lo sviluppo del calcolo infinitesimale. 2. Pier Francesco **Maurolico** (1494-1575): Un matematico italiano che scrisse un trattato sul cal… |
| Baruch Spinoza <br>*note: philosopher* | misattributed | 1 | llama 1 | it 1 | *llama, it:* …tro Mengoli e Bonaventura Cavalieri sono meno conosciuti. 3. **Baruch Sp** |
| Euclid of Megara <br>*note: philosopher; named as a treatise title* | misattributed | 1 | llama 1 | it 1 | *llama, it:* …aliano che scrisse un trattato sul calcolo infinitesimale intitolato "**Euclides Megarensis**". Tuttavia, il suo lavoro non fu immediatamente compreso. 3. Pierr… |
| Gilbert N. Lewis <br>*note: chemist, unrelated* | misattributed | 1 | mistral 1 | fr 1 | *mistral, fr:* …en français Blaise Pascal. Cependant, c'est le mathématicien français **Gilbert N. Lewis** qui a popularisé l'expression "calcul infinitésimal". Le calcul infin… |
| Giovanni Bonaventura Calandrelli <br>*note: no 17th-century mathematician of that name (Giuseppe Calandrelli was an 18th-19th c. astronomer)* | fabricated | 1 | llama 1 | it 1 | *llama, it:* …incipali creatori sono considerati essere: 1. Giovanni Bonaventura **Calandrelli** non è stato il vero iniziatore, ma è stato un matematico italiano d… |
| Gianfrancesco Ceva <br>*note: garbled Giovanni Ceva* | fabricated | 1 | llama 1 | it 1 | *llama, it:* …un secondo nome) non fu un altro matematico che, ma soprattutto da **Gianfrancesco Ceva**, ma soprattutto da Bonaventura Cavalieri, ma soprattutto da Bonamp… |
| Pieroni Bonaventura <br>*note: garbled Cavalieri* | fabricated | 1 | llama 1 | it 1 | *llama, it:* …indivisibili" per calcolare aree e volumi di figure geometriche. 3. **Pieroni Bonaventura** ( |
| Bonaventura Pascal <br>*note: mix of Cavalieri and Pascal* | fabricated | 1 | llama 1 | fr 1 | *llama, fr:* …idées qui allaient plus tard influencer le calcul infinitésimal. 3. **Bonaventura Pascal** (1623-1662) : Pascal a travaillé sur les mathématiques, la physique… |

### cinema

| Person | Status | Mentions | Models | Languages | Example |
| --- | --- | ---: | --- | --- | --- |
| Eadweard Muybridge | related | 78 | llama 73, qwen 5 | pt 19, en 20, de 17, it 15, fr 7 | *llama, de:* …eg zum modernen Kino war die Erfindung des Zooprassers durch Eadweard **Muybridge** im Jahr 1878. Muybridge entwickelte eine Serie von Fotografien, die n… |
| Louis Le Prince | related | 69 | llama 51, qwen 14, mistral 4 | pt 8, en 22, de 15, it 17, fr 7 | *llama, de:* …eiterer wichtiger Schritt war die Erfindung des Kinematographen durch **Louis Le Prince** im Jahr 1888. Le Prince entwickelte eine Kamera, die Bilder aufzeichn… |
| Étienne-Jules Marey | related | 67 | llama 48, qwen 19 | pt 15, en 20, de 8, it 10, fr 14 | *llama, de:* …eiterer wichtiger Pionier war der französische Erfinder Étienne-Jules **Marey**, der 1882 ein Gerät entwickelte, das sogenannte "Chronophotogramm", d… |
| Joseph Plateau | related | 28 | qwen 23, llama 5 | pt 8, en 3, de 2, it 4, fr 11 | *llama, de:* …Personen, die zum Aufkommen des Kinos beigetragen haben, sind: 1. **Joseph Plateau** (1801-1883): Ein belgischer Physiker, der 1839 die Phantasiebühne e… |
| Louis Daguerre <br>*note: photography, cinema precursor* | related | 20 | qwen 20 | pt 8, de 3, it 4, fr 5 | *qwen, de:* …n Arbeit und Erfindungen maßgeblich dazu beigetragen haben: 1. Louis **Daguerre**: Er erfand 1839 das Daguerreotypie-Verfahren, eine frühe Form der pho… |
| Nicéphore Niépce <br>*note: photography, cinema precursor* | related | 8 | qwen 7, llama 1 | pt 4, en 1, de 2, it 1 | *llama, en:* …res date back to the 1820s, when the French inventor Joseph Nicéphore **Niépce** created a process called heliography, which involved capturing images… |
| Émile Reynaud | related | 5 | llama 4, mistral 1 | pt 2, fr 3 | *llama, fr:* …e" qui permettait de voir des images animées en mouvement. 2. Émile **Reynaud** (1844-1918) : un inventeur et un artiste français qui a créé le "pr… |
| William Henry Fox Talbot <br>*note: photography, cinema precursor* | related | 4 | qwen 4 | pt 1, de 1, it 1, fr 1 | *qwen, de:* …stammt aus dem Jahr 1826. 3. Louis-Jacques-Mandé Daguerre und Henry **Fox Talbot**: Beide entwickelten aufeinander aufbauende Techniken |
| Georges Méliès <br>*note: early filmmaker* | related | 3 | mistral 2, qwen 1 | de 1, it 2 | *mistral, it:* …amente fu "Le Voyage dans la Lune" (Il viaggio sulla Luna) di Georges **Méliès**, girato nel 1902. |
| William Friese-Greene | related | 3 | qwen 3 | pt 2, en 1 | *qwen, en:* …jects, which was a precursor to motion picture cameras. 3. William **Friese-Greene** (1829-1904): An English inventor who claimed to have invented a mot… |
| William Kennedy Laurie Dickson | related | 2 | llama 1, qwen 1 | en 1, de 1 | *llama, en:* …arrative film, "Roundhay Garden Scene," in 1888. 3. William Kennedy **Dickson** (1860-1935) and William Heise (1850-1888): Dickson |
| Simon Stampfer <br>*note: stroboscopic disc* | related | 2 | llama 1, qwen 1 | pt 1, fr 1 | *llama, pt:* …, inventado em 1832 pelo alemão Joseph Plateau e pelo belga Simon von **Stampfer**. No entanto, este dispositivo não era capaz de exibir imagens em movi… |
| Roundhay Garden Scene people (Whitley, Hartley) <br>*note: people filmed in Roundhay Garden Scene* | related | 2 | llama 2 | en 2 | *llama, en:* …e." This film showed Adolphe Le Prince (Louis Le Prince's son), Sarah **Whitley** (Le Prince's mother-in-law), Joseph Whitley (Le Prince's father-in-la… |
| William Heise | related | 1 | llama 1 | en 1 | *llama, en:* …Garden Scene," in 1888. 3. William Kennedy Dickson (1860-1935) and **William Heise** (1850-1888): Dickson |
| Leland Stanford <br>*note: commissioned Muybridge* | related | 1 | llama 1 | en 1 | *llama, en:* …first motion picture sequence. In 1878, Muybridge was commissioned by **Leland Stanford** to prove that a horse lifts all four legs off the ground at some poin… |
| Max Skladanowsky | related | 1 | mistral 1 | de 1 | *mistral, de:* …ene Kino der Welt, wurde von dem deutschen Erfinder und Ingenieur Max **Skladanowsky** erfunden. Am 1. November 1895 präsentierte er in Berlin sein Cinemato… |
| Léon Bouly | related | 1 | mistral 1 | de 1 | *mistral, de:* …hwand und mit ihm die Originalaufnahmen. Der belgische Ingenieur Léon **Bouly** wurde auch für die Erfindung des Kinos in Erwägung gezogen, da er ber… |
| François Arago <br>*note: announced the daguerreotype* | related | 1 | qwen 1 | it 1 | *qwen, it:* …e fondamentali nella storia del cinema: 1. Louis Daguerre e François **Arago**: Il fotografo Louis Daguerre presentò la sua macchina fotografica dag… |
| Louis-Jules Duboscq <br>*note: stereoscopic moving images (Bioscope)* | related | 1 | mistral 1 | pt 1 | *mistral, pt:* …são atribuídos a dois cientistas franceses do século XIX: Louis-Jules **Duboscq** e Charles-Émile Reynaud. Em 1872, Duboscq desenvolveu um dispositivo… |
| Christiaan Huygens <br>*note: magic lantern* | related | 1 | llama 1 | de 1 | *llama, de:* …derts entwickelt wurde. Der niederländische牧师 und Erfinder Christiaan **Huygens** (1629-1695) ist als einer der ersten bekannt, der mit dem magic lante… |
| Muzio Oddi <br>*note: Italian mathematician; no diorama* | misattributed | 1 | llama 1 | de 1 | *llama, de:* …Art von optischem Gerät, das im 17. Jahrhundert von dem Niederländer **Muzio Oddi** erfunden wurde. Das Diorama ermöglichte es, Bilder in Bewegung zu bri… |
| Ewald Dupont <br>*note: film director E. A. Dupont* | misattributed | 1 | llama 1 | de 1 | *llama, de:* …egungsmuster eines Pferdes zu erforschen. Ein weiterer Kandidat ist **Ewald Dupont**. Er war nie über die Kinos in der Presse hinaus bekannt und hatte nur… |
| Auguste Le Prince <br>*note: Le Prince's brother was Albert; conflates Auguste Lumière* | fabricated | 1 | mistral 1 | it 1 | *mistral, it:* …cinema è stato inventato da due fratelli francesi, Louis Le Prince e **Auguste Le Prince**, ma la data esatta è oggetto di dibattito. Louis Le Prince ha fatto l… |
| Francis Arnaud <br>*note: no such inventor* | fabricated | 1 | qwen 1 | it 1 | *qwen, it:* …ribuito alla creazione delle tecniche fondamentali per il cinema. 2. **Francis Arnaud** e l'optique phosphorescente: Nel 1872, Francis Arnaud espose delle fo… |
| Augusto Rodrigues <br>*note: no such Brazilian inventor* | fabricated | 1 | llama 1 | pt 1 | *llama, pt:* …mumente creditado ao francês Louis Le Prince e ao brasileiro-inventor **Augusto Rodrigues**. Em 1888, Louis Le Prince registrou um pedido de patente para um "ap… |
| Egidius Glockenthanner <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …smagoria", ein Gerät, das 1790 von dem französischen Erfinder Egidius **Glockenthanner** entwickelt wurde. Es handelte |
| Eiese <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …erät kombinierte. Ein weiterer wichtiger Entwickler war der Franzose **Eiese**, der 1825 ein Gerät konstruierte, das er "Phenakistiskop" nannte. Es… |

### dinamite

| Person | Status | Mentions | Models | Languages | Example |
| --- | --- | ---: | --- | --- | --- |
| Julius Wilbrand <br>*note: TNT, not dynamite* | misattributed | 1 | qwen 1 | de 1 | *qwen, de:* …ung des Dynamits wird dem deutschen Chemiker und Industriellen Julius **Wilbrand** zugeschrieben. Wilbrand entwickelte Dynamit im Jahr 1867. Es ist jedo… |
| Svante Arrhenius <br>*note: chemist, unrelated* | misattributed | 1 | llama 1 | it 1 | *llama, it:* …nzione della dinamite è attribuita al chimico norvegese Svante August **Arrhenius**, ma è stato il suo allievo, il norvegese Ammonium nitrate, che non è… |
| Johan Edvard Lundström <br>*note: safety match* | misattributed | 1 | llama 1 | de 1 | *llama, de:* …rwegischen Chemiker und Unternehmer Gustav Ehrenborg und Johan Edvard **Lundström** im Jahr 1846 erfunden. |
| Eilert Sundt <br>*note: sociologist* | misattributed | 1 | llama 1 | de 1 | *llama, de:* Das Dynamit wurde 1846 von dem norwegischen Chemiker und Unternehmer **Eilert Sundt** erfunden. Er war jedoch nicht der eigentliche Erfinder. Der Chemiker… |
| Gustav Ehrenborg <br>*note: no such chemist* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …as Dynamit wurde von dem norwegischen Chemiker und Unternehmer Gustav **Ehrenborg** und Johan Edvard Lundström im Jahr 1846 erfunden. |
| Camille Alphonse Friedel <br>*note: garbled; Charles Friedel is real* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …unden wurde es von einem norwegischen Chemiker und Unternehmer namens **Camille Alphonse Friedel** und abgewandelt von einem norwegischen Chemiker und Unternehmer namen… |
| Einar Anker <br>*note: no such person* | fabricated | 1 | llama 1 | pt 1 | *llama, pt:* A dinamite foi inventada pelo químico norueguês **Einar Anker** em 1866. |
| Einar Gustaf Emil Palme <br>*note: no such person* | fabricated | 1 | llama 1 | pt 1 | *llama, pt:* A dinamite foi inventada pelo químico norueguês **Einar Gustaf Emil Palme** em 1866. |
| Einar Alexander Müller <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …norwegischen Chemiker und Unternehmer Einar de Balliet, der auch als **Einar Alexander Mller** oder Einar Mller bekannt ist. Aber es war seine Einladung an den Grün… |
| Einar de Balliet <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …mit wurde 1846 von dem norwegischen Chemiker und Unternehmer Einar de **Balliet**, der auch als Einar Alexander Mller oder Einar Mller bekannt ist. Abe… |
| Gustav Blix <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* Das Dynamit wurde von dem norwegischen Chemiker und Unternehmer **Gustav Blix** erkannt, aber eigentlich erfunden wurde es von einem norwegischen Che… |
| Gustav Alfred Bötz <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …mit wurde von dem norwegischen Chemiker und Unternehmer Gustav Alfred **Bötz** und dem deutschen Chemiker Nickel Charles Abraham Thoren 1866 erfunde… |
| Nickel Charles Abraham Thoren <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …Gustav Alfred Bötz und dem deutschen Chemiker Nickel Charles Abraham **Thoren** 1866 erfunden. Es basiert auf einer Mischung aus Nitroglycerin und Ká… |
| Johan G. Bartholin <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …r jedoch nicht der eigentliche Erfinder. Der Chemiker und Unternehmer **Johan G. Bartholin**, der in Norwegen lebte, hatte bereits 1839 ein ähnliches Produkt entw… |
| Johan G. Thorvald / Ekdal Thorvaldsen <br>*note: no such person* | fabricated | 1 | llama 1 | de 1 | *llama, de:* …nd abgewandelt von einem norwegischen Chemiker und Unternehmer namens **Johan G. Thorvald**, aber letztendlich wurde der Name für das Produkt von dem Unternehmen… |
| Albert von Schönbein <br>*note: garbled Christian Friedrich Schönbein (guncotton)* | fabricated | 1 | llama 1 | de 1 | *llama, de:* Der Erfinder des Dynamits war **Albert von Schönbein**. |
| Ascanio Sobrero <br>*note: nitroglycerin* | related | 0 | — | — | — |

### fotografia

| Person | Status | Mentions | Models | Languages | Example |
| --- | --- | ---: | --- | --- | --- |
| Thomas Wedgwood | related | 4 | gemma 1, llama 1, mistral 1, qwen 1 | en 1, de 1, it 1, fr 1 | *gemma, it:* …propaga, contribuendo alla comprensione della fotografia. * Thomas **Wedgwood** (1771-1805): Fu uno dei primi a sperimentare con la creazione di i… |
| Johannes Kepler <br>*note: camera obscura optics* | related | 1 | gemma 1 | it 1 | *gemma, it:* …immagine invertita di un soggetto su un foglio di carta. * Johannes **Kepler** (1571-1630): Sviluppò la teoria dell'occhio umano e del modo in cu… |
| Albrecht Dürer <br>*note: perspective drawing aids, not photography* | misattributed | 1 | gemma 1 | it 1 | *gemma, it:* …di diversi secoli. Ecco alcuni dei contributi chiave: * Albrecht **Dürer** (1471-1528): Fu uno dei primi a sperimentare con la camera oscura,… |
| Blaise Pascal <br>*note: no work on photography* | misattributed | 1 | qwen 1 | de 1 | *qwen, de:* …schen Wirkung von Licht auf bestimmte Substanzen berichtet hat. 2. **Blaise Pascal** (1623-1662): Er arbeitete mit einer Exposeur-Scherbe, die von Licht… |

### radio

| Person | Status | Mentions | Models | Languages | Example |
| --- | --- | ---: | --- | --- | --- |
| James Clerk Maxwell | related | 14 | qwen 7, llama 4, mistral 3 | pt 3, en 2, de 4, it 3, fr 2 | *llama, de:* …ugen. Einige der wichtigsten Persönlichkeiten sind: 1. James Clerk **Maxwell** (1831-1879): Der schottische Mathematiker und Physiker entdeckte di… |
| Heinrich Hertz | related | 12 | qwen 7, llama 4, mistral 1 | pt 2, en 2, de 3, it 3, fr 2 | *llama, de:* …ie Entwicklung von Oszillatoren und Resonanzkreisen wichtig war. 3. **Heinrich Hertz** (1857-1894): Der deutsche Physiker entdeckte 1887 die Funkwellen, d… |
| Nikola Tesla | related | 5 | qwen 4, llama 1 | pt 2, en 1, de 2 | *llama, en:* …electromagnetic waves could be transmitted and received. 3. Nikola **Tesla** (1856-1943): A Serbian-American inventor and engineer who worked on… |
| Oliver Lodge | related | 2 | qwen 2 | pt 1, fr 1 | *qwen, fr:* …magnétiques, dont une partie est la lumière et les ondes radio. 3. **Oliver Lodge** : Il a contribué à l'amélioration des expériences d'Hertz et a mont… |
| Reginald Fessenden | related | 2 | mistral 2 | de 1, fr 1 | *mistral, de:* …as Radio wurde von dem amerikanischen Physiker und Ingenieur Reginald **Fessenden** erfunden. Er sendete am 24. Dezember 1906 das erste von Menschen gehö… |
| Michael Faraday <br>*note: induction, precursor* | related | 1 | qwen 1 | de 1 | *qwen, de:* …oft als Begründer des Modernen Radios angesehen werden: 1. Michael **Faraday** (1791-1867): Er untersuchte den Zusammenhang zwischen Elektrizität… |
| Edwin Armstrong <br>*note: FM* | related | 1 | mistral 1 | fr 1 | *mistral, fr:* Le radio a été inventé par l'Américain **Edwin Armstrong** en 1933. Cependant, la première station de radio a été créée par le C… |
| Samuel Morse <br>*note: telegraph, precursor of wireless telegraphy* | related | 1 | llama 1 | de 1 | *llama, de:* …en, einem Vorläufer des modernen Telefons. Der amerikanische Erfinder **Samuel Morse** entwickelte den Telegrafen |
| Édouard-Léon Scott de Martinville <br>*note: phonautograph (sound recording)* | misattributed | 1 | mistral 1 | de 1 | *mistral, de:* …s Vorläufer des Radios gelten, wurden von dem Amerikaner Edouard-Leon **Scott de Martinville** und dem Briten James Clerk Maxwell entwickelt. Scott de Martinville e… |

### raios_x

| Person | Status | Mentions | Models | Languages | Example |
| --- | --- | ---: | --- | --- | --- |
| Henri Becquerel <br>*note: radioactivity* | related | 1 | mistral 1 | pt 1 | *mistral, pt:* …na história da descoberta dos raios X. Outros cientistas, como Henri **Becquerel** e Pierre Curie, também contribuíram com pesquisas em raios X por volt… |
| William Henry and William Lawrence Bragg <br>*note: X-ray crystallography* | related | 1 | mistral 1 | pt 1 | *mistral, pt:* A descoberta dos raios X foi atribuída a William Henry **Bragg** e seu filho, William Lawrence Bragg, em 1912. Eles usaram um disposit… |
| Pierre Curie <br>*note: radioactivity* | related | 1 | mistral 1 | pt 1 | *mistral, pt:* …coberta dos raios X. Outros cientistas, como Henri Becquerel e Pierre **Curie**, também contribuíram com pesquisas em raios X por volta da mesma époc… |

### tabela_periodica

| Person | Status | Mentions | Models | Languages | Example |
| --- | --- | ---: | --- | --- | --- |
| John Newlands <br>*note: law of octaves* | related | 3 | llama 2, qwen 1 | de 1, it 2 | *llama, de:* Der Begründer des Periodensystems ist John **Newlands** (1837-1898), ein britischer Chemiker. Er entdeckte 1863, dass die Ele… |
| Antoine Lavoisier <br>*note: list of elements* | related | 2 | llama 1, qwen 1 | it 2 | *llama, it:* …hanno contribuito allo sviluppo della tavola periodica: 1. Antoine **Lavoisier** (1743-1794): Lavoisier è considerato il padre della chimica moderna… |

## 2. Fabricated and misattributed names: full answers

### Fabricated

#### Karl Günter (as alias of Whitehead) (aviao, 1 answer)

Forms: `Karl Günter`. Note: not an alias of Whitehead (born Gustav Weißkopf)

**qwen · de · aviao** — paraphrase 3, sample 9  
Prompt: *Wem wird die Erfindung des Flugzeugs zugeschrieben?*

> Die Erfindung des Flugzeugs ist ein komplexes und kontroverse Thema, da es mehrere Kontributoren und Entwicklungen gab, die zu dem Flugzeug führten. Allerdings wird oft Karl Günter aka Gustave Whitehead als einer der ersten成功的中文翻译是：
>
> 成功的中文翻译是：成功的前提是努力和坚持。这意味着要实现目标，个人需要付出持续的努力，并保持坚持不懈的态度。成功往往不是一蹴而就的，而是一个逐步积累和不断改进的过程。因此，保持积极的态度，不断学习和实践，是通往成功的必要条件。

#### Léonard de Rochechouart (aviao, 1 answer)

Forms: `Rochechouart`. Note: no such aviation pioneer; the 1874 steam monoplane was Félix du Temple's

**qwen · de · aviao** — paraphrase 1, sample 3  
Prompt: *Wer hat das Flugzeug erfunden?*

> Das Flugzeug wurde nicht von einer einzelnen Person erfunden, sondern es entwickelte sich über Jahrzehnte und Jahrhunderte durch die Arbeit vieler Wissenschaftler und Ingenieure. Einige wichtige Schritte in der Entwicklung des Fliegenden Maschinen sind:
>
> 1. **Sir George Cayley (1773-1857)**: Er gilt als der "Vater des Fliegenden Maschinen". Cayley untersuchte die Grundlagen der Luftschub und Luftwiderstand und baute 1804 das erste Fluggerät, das von einer Schubkraft angetrieben wurde.
>
> 2. **Léonard de Rochechouart (1827-1893)**: Er baute 1874 ein Fluggerät, das von einem Dampfmaschinen angetrieben wurde.
>
> 3. **Clément Ader (18

#### Claude-Denis Chastain / Rozier / Chastellux (aviao, 3 answers)

Forms: `Claude-Denis`. Note: garbled; Pilâtre de Rozier is real

**llama · fr · aviao** — paraphrase 1, sample 1  
Prompt: *Qui a inventé l'avion ?*

> L'invention de l'avion est un processus complexe qui a évolué au fil du temps et impliqué plusieurs personnes et inventions. Cependant, certains individus sont généralement considérés comme les pionniers de l'aviation.
>
> Les premiers à avoir conçu des aéronefs étaient :
>
> 1. **Leonardo da Vinci** (1452-1519) : L'artiste, scientifique et ingénieur italien a dessiné des plans pour un aéronef en 1485, qui ressemblait à un aéroplane moderne. Bien que son conception n'ait jamais été construite ou testée, elle a influencé les conceptions ultérieures.
> 2. **Claude-Denis Rozier** (1704-1763) : Le physicien et mathématicien français a conçu un aérostat, un ballon aérien, en 178

**llama · fr · aviao** — paraphrase 2, sample 4  
Prompt: *Qui était l'inventeur de l'avion ?*

> L'invention de l'avion est un processus complexe qui a impliqué de nombreux inventeurs et chercheurs au fil des siècles. Il est difficile de désigner un seul inventeur de l'avion, car l'évolution de l'appareil a été progressive et s'est déroulée en plusieurs étapes.
>
> Cependant, on peut citer quelques-uns des principaux pionniers de l'aviation :
>
> 1. **Leonardo da Vinci** (1452-1519) : L'artiste et inventeur italien a dessiné des plans d'appareils volants à voilure fixe, tels que le « Ornithoptera », qui était un modèle inspiré de l'oiseau.
> 2. **Claude-Denis Chastellux** (1715-1788) : L'officier et ingénieur français a conçu un modèle d'avion à voilure

**llama · it · aviao** — paraphrase 2, sample 6  
Prompt: *Chi è stato l'inventore dell'aereo?*

> L'invenzione dell'aereo è attribuita a diverse persone, a seconda della definizione di "aereo" che si utilizza. Ecco alcune delle figure più importanti nella storia dell'aviazione:
>
> *   **Claude-Denis Chastain** : è considerato il primo uomo a costruire un modello di aeroplano, nel 1783.
> *   **Sir George Cayley** : è considerato il padre dell'aviazione moderna. Nel 1796, sviluppò un progetto per un aeroplano con un'ala a forma di triangolo e un timone di coda.
> *   **Otto Lilienthal** : un ingegnere tedesco che realizzò una serie di esperimenti con modelli di aeroplani e volò con successo per la prima volta nel 1891.
> *   **Octave Chanute**: un ingegnere franc

#### Giovanni Bonaventura Calandrelli (calculo, 1 answer)

Forms: `Calandrelli`. Note: no 17th-century mathematician of that name (Giuseppe Calandrelli was an 18th-19th c. astronomer)

**llama · it · calculo** — paraphrase 2, sample 1  
Prompt: *Chi è stato il creatore del calcolo infinitesimale?*

> Il calcolo infinitesimale è stato sviluppato da diversi matematici nel corso dei secoli, ma i due principali creatori sono considerati essere:
>
> 1. **Giovanni Bonaventura Calandrelli** non è stato il vero iniziatore, ma è stato un matematico italiano del XVII secolo che scrisse un trattato sull'algebra e sul calcolo infinitesimale. Tuttavia, è stato **Bonaventura Cavalieri** che ha sviluppato le nozioni di base del calcolo infinitesimale, introducendo il concetto di "geometria infinitesimale" nel 1635.
>
> 2. **Bonaventura Cavalieri** era un matematico e teologo italiano, ma **Pietro Mengoli** e **Bonaventura Cavalieri** sono meno conosciuti.
>
> 3. **Baruch Sp

#### Gianfrancesco Ceva (calculo, 1 answer)

Forms: `Gianfrancesco Ceva`. Note: garbled Giovanni Ceva

**llama · it · calculo** — paraphrase 1, sample 4  
Prompt: *Chi ha sviluppato il calcolo infinitesimale?*

> Il calcolo infinitesimale è stato sviluppato da due matematici italiani del XVII secolo: Bonaventura Cavalieri e Bonampaglia (che non aveva un secondo nome)  non fu un altro matematico che,  ma soprattutto da Gianfrancesco Ceva, ma soprattutto da  Bonaventura Cavalieri,  ma soprattutto da  Bonampaglia  non fu un altro matematico  e  da  Gianfrancesco Ceva ma  soprattutto da Bonaventura Cavalieri.

#### Pieroni Bonaventura (calculo, 1 answer)

Forms: `Pieroni Bonaventura`. Note: garbled Cavalieri

**llama · it · calculo** — paraphrase 3, sample 1  
Prompt: *A chi si attribuisce lo sviluppo del calcolo infinitesimale?*

> Lo sviluppo del calcolo infinitesimale è attribuito a diversi matematici nel corso dei secoli. Ecco una panoramica dei principali contributi:
>
> 1. **Archimede (287 a.C. - 212 a.C.)**: Il matematico greco Archimede è considerato uno dei fondatori del calcolo infinitesimale. Utilizzò la nozione di "infinitesimali" per calcolare aree e volumi di figure geometriche.
> 2. **Bonaventura Cavalieri (1598-1647)**: Il matematico italiano Bonaventura Cavalieri sviluppò la "metodo delle indivisibili", un precursore del calcolo infinitesimale. Utilizzò la nozione di "indivisibili" per calcolare aree e volumi di figure geometriche.
> 3. **Pieroni Bonaventura (

#### Bonaventura Pascal (calculo, 1 answer)

Forms: `Bonaventura Pascal`. Note: mix of Cavalieri and Pascal

**llama · fr · calculo** — paraphrase 1, sample 8  
Prompt: *Qui a développé le calcul infinitésimal ?*

> Le calcul infinitésimal a été développé par plusieurs mathématiciens au cours de l'histoire, notamment :
>
> 1. **Bonaventura Cavalieri** (1598-1647) : Cavalieri a développé la méthode des indivisibles, qui est à la base du calcul infinitésimal. Il a montré que certains espaces pouvaient être mesurés en utilisant des éléments infiniment petits.
> 2. **Pierre Fermat** (1601-1665) : Fermat a travaillé sur l'arithmétique et la géométrie, et a développé certaines idées qui allaient plus tard influencer le calcul infinitésimal.
> 3. **Bonaventura Pascal** (1623-1662) : Pascal a travaillé sur les mathématiques, la physique et la philosophie, et a développé l'arithmétique infinie, une forme primitive

#### Auguste Le Prince (cinema, 1 answer)

Forms: `Auguste Le Prince`. Note: Le Prince's brother was Albert; conflates Auguste Lumière

**mistral · it · cinema** — paraphrase 1, sample 8  
Prompt: *Chi ha inventato il cinema?*

> Il cinema è stato inventato da due fratelli francesi, Louis Le Prince e Auguste Le Prince, ma la data esatta è oggetto di dibattito. Louis Le Prince ha fatto la sua prima proiezione di un film breve nel 1888, ma è morto l'anno successivo in circostanze misteriose, pertanto non è stato in grado di sviluppare il suo processo cinematografico. La data di nascita del cinema è spesso fissata al 28 dicembre 1895, quando Auguste Le Prince ha presentato il suo film L'arrivée d'un train en gare de La Ciotat, che è stato il primo film a essere proiettato pubblicamente con successo. Tuttavia, la data del 25 dicembre 1895 è talvolta citata come data di

#### Francis Arnaud (cinema, 1 answer)

Forms: `Francis Arnaud`. Note: no such inventor

**qwen · it · cinema** — paraphrase 2, sample 5  
Prompt: *Chi è stato l'inventore del cinema?*

> Il cinema non è stato inventato da una singola persona, ma è il risultato di un evolversi tecnologico e artistico che ha coinvolto numerosi individui e gruppi. Tuttavia, ci sono alcune figure chiave che si sono fondamentali nel perfezionare e diffondere il cinema come lo conosciamo oggi:
>
> 1. Louis Daguerre e dagherrotipo: Il dagherrotipo, sviluppato da Louis Daguerre nel 1839, è stata una delle prime tecniche per la fotografia e ha contribuito alla creazione delle tecniche fondamentali per il cinema.
>
> 2. Francis Arnaud e l'optique phosphorescente: Nel 1872, Francis Arnaud espose delle fotografie successive in rapida successione, creando l'effetto di movimento.
>
> 3. Eadweard Muybridge: Nel 1

#### Augusto Rodrigues (cinema, 1 answer)

Forms: `Augusto Rodrigues`. Note: no such Brazilian inventor

**llama · pt · cinema** — paraphrase 3, sample 5  
Prompt: *A quem se atribui a invenção do cinema?*

> A invenção do cinema é um tema controverso e complexo, envolvendo a contribuição de vários indivíduos ao longo do tempo. Embora seja difícil atribuir a invenção do cinema a uma única pessoa, é comumente creditado ao francês Louis Le Prince e ao brasileiro-inventor Augusto Rodrigues.
>
> Em 1888, Louis Le Prince registrou um pedido de patente para um "aparelho para capturar imagens móveis" em Bruxelas, Bélgica, e novamente em 1889 nos Estados Unidos, tornando-se o primeiro a registrar um pedido de patente para um dispositivo capaz de capturar imagens sequenciais. Le Prince também é conhecido por ter filmado o primeiro filme de uma história contínua, chamado "Roundhay Garden Scene", em 1888.
>
> Por outro lado, o brasileiro Augusto Rodrigues, em 1892,

#### Egidius Glockenthanner (cinema, 1 answer)

Forms: `Glockenthanner`. Note: no such person

**llama · de · cinema** — paraphrase 1, sample 1  
Prompt: *Wer hat das Kino erfunden?*

> Das Kino ist eine Entwicklung, die sich über mehrere Jahrhunderte hinweg erstreckt und von verschiedenen Einzelpersonen und Gruppen betrieben wurde. Es gibt jedoch einige wichtige Persönlichkeiten, die maßgeblich zum Aufkommen des Kinos beigetragen haben.
>
> Ein wichtiger Vorläufer des Kinos ist der "magic lantern" (Zauberschirm), ein Gerät, das bereits in der Mitte des 17. Jahrhunderts entwickelt wurde. Der niederländische牧师 und Erfinder Christiaan Huygens (1629-1695) ist als einer der ersten bekannt, der mit dem magic lantern experimentierte.
>
> Ein weiterer wichtiger Vorläufer des Kinos ist der "Phantasmagoria", ein Gerät, das 1790 von dem französischen Erfinder Egidius Glockenthanner entwickelt wurde. Es handelte

#### Eiese (cinema, 1 answer)

Forms: `Eiese`. Note: no such person

**llama · de · cinema** — paraphrase 1, sample 7  
Prompt: *Wer hat das Kino erfunden?*

> Das Kino, wie wir es heute kennen, ist ein Produkt der Kombination von verschiedenen Technologien und Entwicklungen. Die Geschichte des Kinos reicht jedoch weit zurück und ist eng verbunden mit der Geschichte der Fotografie und der Projektion von Bildern.
>
> Ein wichtiger Meilenstein auf dem Weg zum modernen Kino war die Erfindung des Dioramas, einer Art von optischem Gerät, das im 17. Jahrhundert von dem Niederländer Muzio Oddi erfunden wurde. Das Diorama ermöglichte es, Bilder in Bewegung zu bringen, indem man eine Kamera und ein Projektionsgerät kombinierte.
>
> Ein weiterer wichtiger Entwickler war der Franzose Eiese, der 1825 ein Gerät konstruierte, das er "Phenakistiskop" nannte. Es bestand aus einer Reihe von Lochblättern, die

#### Gustav Ehrenborg (dinamite, 1 answer)

Forms: `Ehrenborg`. Note: no such chemist

**llama · de · dinamite** — paraphrase 1, sample 10  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde von dem norwegischen Chemiker und Unternehmer Gustav Ehrenborg und Johan Edvard Lundström im Jahr 1846 erfunden.

#### Camille Alphonse Friedel (dinamite, 1 answer)

Forms: `Camille Alphonse Friedel`. Note: garbled; Charles Friedel is real

**llama · de · dinamite** — paraphrase 1, sample 2  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde von dem norwegischen Chemiker und Unternehmer Gustav Blix erkannt, aber eigentlich erfunden wurde es von einem norwegischen Chemiker und Unternehmer namens Camille Alphonse Friedel und abgewandelt von einem norwegischen Chemiker und Unternehmer namens Johan G. Thorvald, aber letztendlich wurde der Name für das Produkt von dem Unternehmen dynamitfabrik in Deutschland in den 1870er Jahren vermehrt für das Produkt verwendet und erhielt seinen Namen von dem griechischen Wort "dynamis", was "Kraft" oder "Kraftmittel" bedeutet. 
>
> Die ursprüngliche Form des Dynamits wurde von dem norwegischen Chemiker Einar "Einar" Ekdal Thorvaldsen und auch Camille Alphonse Friedel entwickelt, der eine Mischung von Salpeter, Schwefel und Kohle entwickelt hatte.

#### Einar Anker (dinamite, 1 answer)

Forms: `Einar Anker`. Note: no such person

**llama · pt · dinamite** — paraphrase 1, sample 7  
Prompt: *Quem inventou a dinamite?*

> A dinamite foi inventada pelo químico norueguês Einar Anker em 1866.

#### Einar Gustaf Emil Palme (dinamite, 1 answer)

Forms: `Einar Gustaf Emil Palme`. Note: no such person

**llama · pt · dinamite** — paraphrase 1, sample 10  
Prompt: *Quem inventou a dinamite?*

> A dinamite foi inventada pelo químico norueguês Einar Gustaf Emil Palme em 1866.

#### Einar Alexander Müller (dinamite, 1 answer)

Forms: `Einar Alexander Müller`, `Einar Alexander Muller`, `Einar Alexander Mller`, `Einar Müller`, `Einar Mller`. Note: no such person

**llama · de · dinamite** — paraphrase 1, sample 3  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde 1846 von dem norwegischen Chemiker und Unternehmer Einar de Balliet, der auch als Einar Alexander Mller oder Einar Mller bekannt ist. Aber es war seine Einladung an den Gründer von Dynamit, Eichel  aber der Chemiker und Unternehmer  Norweger Einar Alexander Mller, beauftragte den schwedischen Chemiker und Unternehmer  1837 geborene 2.  Norweger Einar Alexander Muller  wurde  Einar Alexander Mller    aber der Chemiker und Unternehmer  Norweger 1837 geborene 2.  aber  Norweger 1837 geborene Einar Alexander Müller 1837 geboren  Norwegischer Chemiker und Unternehmer  aber der  Chemiker und  Norweger Einar Alexander Müller 1837 geboren  2.  Norweger 1837  aber Chemiker

#### Einar de Balliet (dinamite, 1 answer)

Forms: `Balliet`. Note: no such person

**llama · de · dinamite** — paraphrase 1, sample 3  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde 1846 von dem norwegischen Chemiker und Unternehmer Einar de Balliet, der auch als Einar Alexander Mller oder Einar Mller bekannt ist. Aber es war seine Einladung an den Gründer von Dynamit, Eichel  aber der Chemiker und Unternehmer  Norweger Einar Alexander Mller, beauftragte den schwedischen Chemiker und Unternehmer  1837 geborene 2.  Norweger Einar Alexander Muller  wurde  Einar Alexander Mller    aber der Chemiker und Unternehmer  Norweger 1837 geborene 2.  aber  Norweger 1837 geborene Einar Alexander Müller 1837 geboren  Norwegischer Chemiker und Unternehmer  aber der  Chemiker und  Norweger Einar Alexander Müller 1837 geboren  2.  Norweger 1837  aber Chemiker

#### Gustav Blix (dinamite, 1 answer)

Forms: `Gustav Blix`. Note: no such person

**llama · de · dinamite** — paraphrase 1, sample 2  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde von dem norwegischen Chemiker und Unternehmer Gustav Blix erkannt, aber eigentlich erfunden wurde es von einem norwegischen Chemiker und Unternehmer namens Camille Alphonse Friedel und abgewandelt von einem norwegischen Chemiker und Unternehmer namens Johan G. Thorvald, aber letztendlich wurde der Name für das Produkt von dem Unternehmen dynamitfabrik in Deutschland in den 1870er Jahren vermehrt für das Produkt verwendet und erhielt seinen Namen von dem griechischen Wort "dynamis", was "Kraft" oder "Kraftmittel" bedeutet. 
>
> Die ursprüngliche Form des Dynamits wurde von dem norwegischen Chemiker Einar "Einar" Ekdal Thorvaldsen und auch Camille Alphonse Friedel entwickelt, der eine Mischung von Salpeter, Schwefel und Kohle entwickelt hatte.

#### Gustav Alfred Bötz (dinamite, 1 answer)

Forms: `Bötz`. Note: no such person

**llama · de · dinamite** — paraphrase 1, sample 7  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde von dem norwegischen Chemiker und Unternehmer Gustav Alfred Bötz und dem deutschen Chemiker Nickel Charles Abraham Thoren 1866 erfunden. Es basiert auf einer Mischung aus Nitroglycerin und Káiserstoff (Kaliumnitrat) und wurde vor allem für bergmännische und militärische Zwecke verwendet.

#### Nickel Charles Abraham Thoren (dinamite, 1 answer)

Forms: `Thoren`. Note: no such person

**llama · de · dinamite** — paraphrase 1, sample 7  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde von dem norwegischen Chemiker und Unternehmer Gustav Alfred Bötz und dem deutschen Chemiker Nickel Charles Abraham Thoren 1866 erfunden. Es basiert auf einer Mischung aus Nitroglycerin und Káiserstoff (Kaliumnitrat) und wurde vor allem für bergmännische und militärische Zwecke verwendet.

#### Johan G. Bartholin (dinamite, 1 answer)

Forms: `Johan G. Bartholin`. Note: no such person

**llama · de · dinamite** — paraphrase 1, sample 9  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde 1846 von dem norwegischen Chemiker und Unternehmer Eilert Sundt erfunden. Er war jedoch nicht der eigentliche Erfinder. Der Chemiker und Unternehmer Johan G. Bartholin, der in Norwegen lebte, hatte bereits 1839 ein ähnliches Produkt entwickelt, nämlich Pulver, das aus Nitroglycerin bestand.
>
> Eilert Sundt baute jedoch auf dieser Arbeit auf und entwickelte ein Produkt, das er "Dynamit" nannte. Sein Produkt bestand aus Nitroglycerin, das in kleine Stücke zerschnitten und in eine Kartusche gefüllt wurde. Dieses Produkt war äußerst explosiv und wurde schnell zum Standard-Explosivstoff in der Bergbaubranche.
>
> Sundt gründete 1864 in Deutschland die Dynamit Aktiengesellschaft (DAG), die heute

#### Johan G. Thorvald / Ekdal Thorvaldsen (dinamite, 1 answer)

Forms: `Johan G. Thorvald`, `Ekdal Thorvaldsen`. Note: no such person

**llama · de · dinamite** — paraphrase 1, sample 2  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde von dem norwegischen Chemiker und Unternehmer Gustav Blix erkannt, aber eigentlich erfunden wurde es von einem norwegischen Chemiker und Unternehmer namens Camille Alphonse Friedel und abgewandelt von einem norwegischen Chemiker und Unternehmer namens Johan G. Thorvald, aber letztendlich wurde der Name für das Produkt von dem Unternehmen dynamitfabrik in Deutschland in den 1870er Jahren vermehrt für das Produkt verwendet und erhielt seinen Namen von dem griechischen Wort "dynamis", was "Kraft" oder "Kraftmittel" bedeutet. 
>
> Die ursprüngliche Form des Dynamits wurde von dem norwegischen Chemiker Einar "Einar" Ekdal Thorvaldsen und auch Camille Alphonse Friedel entwickelt, der eine Mischung von Salpeter, Schwefel und Kohle entwickelt hatte.

#### Albert von Schönbein (dinamite, 1 answer)

Forms: `Albert von Schönbein`. Note: garbled Christian Friedrich Schönbein (guncotton)

**llama · de · dinamite** — paraphrase 2, sample 7  
Prompt: *Wer war der Erfinder des Dynamits?*

> Der Erfinder des Dynamits war Albert von Schönbein.

### Misattributed

#### William Congreve (aviao, 1 answer)

Forms: `Congreve`. Note: Congreve rockets, not aircraft

**llama · de · aviao** — paraphrase 1, sample 10  
Prompt: *Wer hat das Flugzeug erfunden?*

> Das Flugzeug ist ein wichtiger Meilenstein in der Geschichte der Luftfahrt und wurde von verschiedenen Menschen entwickelt. Der erste nachgewiesene Flug mit einem Fluggerät war ein Rakete, die der englische Erfinder William Congreve 1827 konstruierte. Diese Rakete flog etwa 200 Meter weit.
>
> Ein wichtiger Schritt in der Entwicklung des Flugzeugs war jedoch der Bau des ersten motorisierten Flugzeugs durch den deutschen Ingenieur Otto Lilienthal (1848-1896) im Jahr 1891. Lilienthal baute vier unterschiedliche Flugzeuge, die er selbst erfolgreich mit dem Fliegen testete. Seine Flugzeuge hatten ein festes Fahrwerk und wurden von einem Antrieb mit einem Segel getrieben.
>
> Der erste motorisierte Flug mit einem Flugzeug, das von einem Menschen gesteuert wurde, gelang dem fr

#### Abu al-Wafa' Buzjani (aviao, 1 answer)

Forms: `Buzjani`. Note: mathematician, not aviation

**llama · it · aviao** — paraphrase 1, sample 1  
Prompt: *Chi ha inventato l'aereo?*

> L'invenzione dell'aereo è attribuita a diversi inventori e pionieri dell'aviazione, alcuni dei quali si sono avvicinati allo scopo di creare un veicolo in grado di volare. Ecco alcuni dei principali contribuenti:
>
> 1. **Icaro** (Greco antico, 2000 a.C. circa): Icaro è considerato uno dei primi a volare, anche se in modo non intenzionale. Secondo la leggenda, Icaro ed il suo padre Daedalo crearono ali di cera per volare via da Creta, ma Icaro volò troppo vicino al sole e le ali si sciolsero, facendo precipitare Icaro in mare.
>
> 2. **Abu al-Wafa' Buzjani** (Persiano, 940-998 d.C.): Abu al-Wafa' Buzj

#### Baruch Spinoza (calculo, 1 answer)

Forms: `Baruch Sp`, `Spinoza`. Note: philosopher

**llama · it · calculo** — paraphrase 2, sample 1  
Prompt: *Chi è stato il creatore del calcolo infinitesimale?*

> Il calcolo infinitesimale è stato sviluppato da diversi matematici nel corso dei secoli, ma i due principali creatori sono considerati essere:
>
> 1. **Giovanni Bonaventura Calandrelli** non è stato il vero iniziatore, ma è stato un matematico italiano del XVII secolo che scrisse un trattato sull'algebra e sul calcolo infinitesimale. Tuttavia, è stato **Bonaventura Cavalieri** che ha sviluppato le nozioni di base del calcolo infinitesimale, introducendo il concetto di "geometria infinitesimale" nel 1635.
>
> 2. **Bonaventura Cavalieri** era un matematico e teologo italiano, ma **Pietro Mengoli** e **Bonaventura Cavalieri** sono meno conosciuti.
>
> 3. **Baruch Sp

#### Euclid of Megara (calculo, 1 answer)

Forms: `Euclides Megarensis`. Note: philosopher; named as a treatise title

**llama · it · calculo** — paraphrase 1, sample 6  
Prompt: *Chi ha sviluppato il calcolo infinitesimale?*

> Il calcolo infinitesimale è stato sviluppato da diversi matematici nel corso dei secoli. Due dei principali contribuenti alla sua formulazione sono:
>
> 1. **Bonaventura Cavalieri (1598-1647)**: Un matematico italiano che lavorò sul concetto di "indivisibili", che sono piccoli elementi infinitamente piccoli che possono essere utilizzati per calcolare aree e volumi. Il suo lavoro è stato fondamentale per lo sviluppo del calcolo infinitesimale.
>
> 2. **Pier Francesco Maurolico (1494-1575)**: Un matematico italiano che scrisse un trattato sul calcolo infinitesimale intitolato "Euclides Megarensis". Tuttavia, il suo lavoro non fu immediatamente compreso.
>
> 3. **Pierre Fermat (1601-1665)**:

#### Gilbert N. Lewis (calculo, 1 answer)

Forms: `Gilbert N. Lewis`. Note: chemist, unrelated

**mistral · fr · calculo** — paraphrase 2, sample 2  
Prompt: *Qui était le créateur du calcul infinitésimal ?*

> Le créateur du calcul infinitésimal est le mathématicien français Blaise Pascal. Cependant, c'est le mathématicien français Gilbert N. Lewis qui a popularisé l'expression "calcul infinitésimal". Le calcul infinitésimal a été développé plus tard par le mathématicien français Pierre de Fermat et le mathématicien italien Bonaventura Cavalieri, mais ce n'est que grâce aux travaux de l'analyste suisse Leonhard Euler que ce calcul a été systématisé et formalisé.

#### Muzio Oddi (cinema, 1 answer)

Forms: `Muzio Oddi`. Note: Italian mathematician; no diorama

**llama · de · cinema** — paraphrase 1, sample 7  
Prompt: *Wer hat das Kino erfunden?*

> Das Kino, wie wir es heute kennen, ist ein Produkt der Kombination von verschiedenen Technologien und Entwicklungen. Die Geschichte des Kinos reicht jedoch weit zurück und ist eng verbunden mit der Geschichte der Fotografie und der Projektion von Bildern.
>
> Ein wichtiger Meilenstein auf dem Weg zum modernen Kino war die Erfindung des Dioramas, einer Art von optischem Gerät, das im 17. Jahrhundert von dem Niederländer Muzio Oddi erfunden wurde. Das Diorama ermöglichte es, Bilder in Bewegung zu bringen, indem man eine Kamera und ein Projektionsgerät kombinierte.
>
> Ein weiterer wichtiger Entwickler war der Franzose Eiese, der 1825 ein Gerät konstruierte, das er "Phenakistiskop" nannte. Es bestand aus einer Reihe von Lochblättern, die

#### Ewald Dupont (cinema, 1 answer)

Forms: `Ewald Dupont`. Note: film director E. A. Dupont

**llama · de · cinema** — paraphrase 3, sample 8  
Prompt: *Wem wird die Erfindung des Kinos zugeschrieben?*

> Die Erfindung des Kinos wird Louis Le Prince zugeschrieben. Er war ein französischer Fotograf und Ingenieur, der am 14. Oktober 1888 in Leeds (England) das erste Filmmaterial entwickelte, welches einen 2 Sekunden Film zeigte. 
>
> Ein weiterer Kandidat für die Erfindung des Kinos ist Eadweard Muybridge. Er war ein britischer Fotograf und Erfinder, der 1878 eine Serie von Fotos von einem Pferd beim Galoppieren aufnahm, um das Bewegungsmuster eines Pferdes zu erforschen. 
>
> Ein weiterer Kandidat ist Ewald Dupont. Er war nie über die Kinos in der Presse hinaus bekannt und hatte nur einige wenige Filme produziert.

#### Julius Wilbrand (dinamite, 1 answer)

Forms: `Wilbrand`. Note: TNT, not dynamite

**qwen · de · dinamite** — paraphrase 3, sample 10  
Prompt: *Wem wird die Erfindung des Dynamits zugeschrieben?*

> Die Erfindung des Dynamits wird dem deutschen Chemiker und Industriellen Julius Wilbrand zugeschrieben. Wilbrand entwickelte Dynamit im Jahr 1867. Es ist jedoch wichtig zu beachten, dass der Begriff "Dynamit" oft im Allgemeinen verwendet wird, um alle Formen von explosiven Stoffen zu bezeichnen, obwohl die spezifische Erfindung von Wilbrand sich durch seine speziellen Eigenschaften wie Viskosität und Arbeitsfähigkeit auszeichnet.

#### Svante Arrhenius (dinamite, 1 answer)

Forms: `Arrhenius`. Note: chemist, unrelated

**llama · it · dinamite** — paraphrase 3, greedy  
Prompt: *A chi si attribuisce l'invenzione della dinamite?*

> L'invenzione della dinamite è attribuita al chimico norvegese Svante August Arrhenius, ma è stato il suo allievo, il norvegese Ammonium nitrate, che non è stato l'inventore della dinamite, ma il suo allievo, il norvegese  Ammonium nitrate, che non è stato l'inventore della dinamite, ma il suo allievo, il norvegese  Ammonium nitrate, che non è stato l'inventore della dinamite, ma il suo allievo, il norvegese  Ammonium nitrate, che non è stato l'inventore della dinamite, ma il suo allievo, il norvegese  Ammonium nitrate, che non è stato l'inventore della dinamite, ma il suo allievo, il norvegese

#### Johan Edvard Lundström (dinamite, 1 answer)

Forms: `Lundström`. Note: safety match

**llama · de · dinamite** — paraphrase 1, sample 10  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde von dem norwegischen Chemiker und Unternehmer Gustav Ehrenborg und Johan Edvard Lundström im Jahr 1846 erfunden.

#### Eilert Sundt (dinamite, 1 answer)

Forms: `Eilert Sundt`. Note: sociologist

**llama · de · dinamite** — paraphrase 1, sample 9  
Prompt: *Wer hat das Dynamit erfunden?*

> Das Dynamit wurde 1846 von dem norwegischen Chemiker und Unternehmer Eilert Sundt erfunden. Er war jedoch nicht der eigentliche Erfinder. Der Chemiker und Unternehmer Johan G. Bartholin, der in Norwegen lebte, hatte bereits 1839 ein ähnliches Produkt entwickelt, nämlich Pulver, das aus Nitroglycerin bestand.
>
> Eilert Sundt baute jedoch auf dieser Arbeit auf und entwickelte ein Produkt, das er "Dynamit" nannte. Sein Produkt bestand aus Nitroglycerin, das in kleine Stücke zerschnitten und in eine Kartusche gefüllt wurde. Dieses Produkt war äußerst explosiv und wurde schnell zum Standard-Explosivstoff in der Bergbaubranche.
>
> Sundt gründete 1864 in Deutschland die Dynamit Aktiengesellschaft (DAG), die heute

#### Albrecht Dürer (fotografia, 1 answer)

Forms: `Dürer`. Note: perspective drawing aids, not photography

**gemma · it · fotografia** — paraphrase 1, greedy  
Prompt: *Chi ha inventato la fotografia?*

> Non esiste una singola persona che può essere accreditata dell'invenzione della fotografia. 
>
> La fotografia è il risultato di un lungo processo di scoperte e innovazioni da parte di molti scienziati e inventori nel corso di diversi secoli. 
>
> Ecco alcuni dei contributi chiave:
>
> * **Albrecht Dürer (1471-1528):**  Fu uno dei primi a sperimentare con la camera oscura, un dispositivo che proiettava un'immagine invertita di un soggetto su un foglio di carta.
> * **Johannes Kepler (1571-1630):**  Sviluppò la teoria dell'occhio umano e del modo in cui la luce si propaga, contribuendo alla comprensione della fotografia.
> * **Thomas Wedgwood (1771-1805):**  Fu uno dei primi a sperimentare con la creazione di immagini permanenti su materiali sensibili alla

#### Blaise Pascal (fotografia, 1 answer)

Forms: `Blaise Pascal`. Note: no work on photography

**qwen · de · fotografia** — paraphrase 1, sample 1  
Prompt: *Wer hat die Fotografie erfunden?*

> Die Fotografie entwickelte sich über viele Jahrhunderte und hat mehrere wichtige Entdecker und Erfinder gebracht. Hier sind einige wesentliche Schritte:
>
> 1. **Maier von Königstuhl (um 1320)**: Ein türkischer Schreiner und Philosoph, der als einer der ersten bekannt ist, der von der chemischen Wirkung von Licht auf bestimmte Substanzen berichtet hat.
>
> 2. **Blaise Pascal (1623-1662)**: Er arbeitete mit einer Exposeur-Scherbe, die von Licht durch das Auge eines Teleskops gespeist wurde, um ein Bild auf eine chemische Substanz zu projizieren.
>
> 3. **Thomas Wedgwood (1771-1805)**: Er war der erste, der versuchte, Bilder auf Pergament zu fixieren, indem er sie in

#### Édouard-Léon Scott de Martinville (radio, 1 answer)

Forms: `Scott de Martinville`. Note: phonautograph (sound recording)

**mistral · de · radio** — paraphrase 1, sample 5  
Prompt: *Wer hat das Radio erfunden?*

> Das Radio wurde nicht von einem einzelnen Erfinder erfunden, sondern ist das Ergebnis der Entwicklung in der Physik und Elektrotechnik des 19. und frühen 20. Jahrhunderts durch viele Wissenschaftler und Ingenieure. Ein wichtiger Schritt dazu war die Entdeckung des elektromagnetischen Wellen von James Clerk Maxwell im Jahr 1864.
>
> Die ersten Geräte, die als Vorläufer des Radios gelten, wurden von dem Amerikaner Edouard-Leon Scott de Martinville und dem Briten James Clerk Maxwell entwickelt. Scott de Martinville erfand 1857 das Fonographe, ein Gerät, das Tonaufnahmen auf rotierende Zylindermembranen machte. Maxwell erstellte ab 1867

## 3. No-candidate answers that name nobody on the list (1)

**qwen · pt · calculo** — paraphrase 2, sample 1  
Prompt: *Quem foi o criador do cálculo infinitesimal?*

> O cálculo infinitesimal, um campo importante da matemática que estuda as taxas de変量，请您确认是“变量”还是“变化量”，以便我能够提供准确的回答。
> 如果是“变量”，那么答案是：微积分学是由多位数学家的发展和贡献而形成的，但通常认为是艾萨克·牛顿和戈特弗里德·莱布尼茨独立地创立了微积分学。牛顿在17世纪末提出了他的“无限小增量”方法，而莱布尼茨则在稍晚的时候发展了一套更加系统和符号化的微积分理论。
>
> 如果是“变化量”，那么答案与上述相同，微积分学的主要创立者是牛顿和莱布尼茨。他们的工作为后来的数学、物理等领域的发展奠定了基础。

## 4. Language concentration (disputed facts)

People with at least 5 mentions in the fact's no-candidate answers. Each language cell is *mentions (share of that language's no-candidate answers naming the person)*. *Top* is the language with the most mentions and its share of the person's mentions; *Baseline* is that language's share of the fact's no-candidate answers, i.e. what an even spread would give. *Ratio* = top share / baseline. **Concentrated** means top share ≥ 50% and ratio ≥ 1.5.

Caveat: no-candidate answers come mostly from Llama 3.1 (and Qwen, Mistral), and which models produce them differs by language. A concentration can come from one model rather than the language, which is why *Top-language models* is shown.

**Concentrated:**

- aviao: **George Cayley**, 17 of 32 mentions in de (53%, baseline 29%); de models: qwen 9, llama 5, mistral 3.
- aviao: **Samuel Pierpont Langley**, 3 of 5 mentions in it (60%, baseline 29%); it models: mistral 3.
- cinema: **Nicéphore Niépce**, 4 of 8 mentions in pt (50%, baseline 22%); pt models: qwen 4.
- cinema: **Émile Reynaud**, 3 of 5 mentions in fr (60%, baseline 16%); fr models: llama 3.

### aviao

No-candidate answers per language: pt 17, en 1, de 30, it 30, fr 25 (total 103).  
Models: pt: llama 17; en: llama 1; de: llama 14, qwen 11, mistral 5; it: llama 27, mistral 3; fr: llama 23, mistral 2.

| Person | Mentions | pt | en | de | it | fr | Top | Baseline | Ratio | Top-language models | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| Otto Lilienthal | 63 | 14 (82%) | 1 (100%) | 18 (60%) | 16 (53%) | 14 (56%) | de 29% | 29% | 1.0 | llama 12, mistral 3, qwen 3 | spread |
| Clément Ader | 57 | 15 (88%) | 0 (0%) | 9 (30%) | 14 (47%) | 19 (76%) | fr 33% | 24% | 1.4 | llama 19 | spread |
| George Cayley | 32 | 2 (12%) | 1 (100%) | 17 (57%) | 10 (33%) | 2 (8%) | de 53% | 29% | 1.8 | qwen 9, llama 5, mistral 3 | **Concentrated** |
| Octave Chanute | 27 | 4 (24%) | 0 (0%) | 3 (10%) | 11 (37%) | 9 (36%) | it 41% | 29% | 1.4 | llama 11 | spread |
| Leonardo da Vinci | 20 | 1 (6%) | 0 (0%) | 7 (23%) | 6 (20%) | 6 (24%) | de 35% | 29% | 1.2 | qwen 4, llama 2, mistral 1 | spread |
| Samuel Pierpont Langley | 5 | 0 (0%) | 0 (0%) | 2 (7%) | 3 (10%) | 0 (0%) | it 60% | 29% | 2.1 | mistral 3 | **Concentrated** |

### telefone

No answer without a candidate.

### radio

No-candidate answers per language: pt 3, en 2, de 6, it 3, fr 3 (total 17).  
Models: pt: qwen 3; en: llama 1, mistral 1; de: llama 2, mistral 2, qwen 2; it: qwen 2, llama 1; fr: qwen 2, mistral 1.

| Person | Mentions | pt | en | de | it | fr | Top | Baseline | Ratio | Top-language models | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| James Clerk Maxwell | 14 | 3 (100%) | 2 (100%) | 4 (67%) | 3 (100%) | 2 (67%) | de 29% | 35% | 0.8 | llama 2, mistral 2 | spread |
| Heinrich Hertz | 12 | 2 (67%) | 2 (100%) | 3 (50%) | 3 (100%) | 2 (67%) | de 25% | 35% | 0.7 | llama 2, qwen 1 | spread |
| Nikola Tesla | 5 | 2 (67%) | 1 (50%) | 2 (33%) | 0 (0%) | 0 (0%) | pt 40% | 18% | 2.3 | qwen 2 | spread |

### cinema

No-candidate answers per language: pt 27, en 24, de 26, it 25, fr 19 (total 121).  
Models: pt: llama 16, qwen 10, mistral 1; en: llama 20, qwen 4; de: llama 20, qwen 4, mistral 2; it: llama 15, qwen 7, mistral 3; fr: llama 11, qwen 8.

| Person | Mentions | pt | en | de | it | fr | Top | Baseline | Ratio | Top-language models | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| Eadweard Muybridge | 78 | 19 (70%) | 20 (83%) | 17 (65%) | 15 (60%) | 7 (37%) | en 26% | 20% | 1.3 | llama 20 | spread |
| Louis Le Prince | 69 | 8 (30%) | 22 (92%) | 15 (58%) | 17 (68%) | 7 (37%) | en 32% | 20% | 1.6 | llama 19, qwen 3 | spread |
| Étienne-Jules Marey | 67 | 15 (56%) | 20 (83%) | 8 (31%) | 10 (40%) | 14 (74%) | en 30% | 20% | 1.5 | llama 16, qwen 4 | spread |
| Joseph Plateau | 28 | 8 (30%) | 3 (12%) | 2 (8%) | 4 (16%) | 11 (58%) | fr 39% | 16% | 2.5 | qwen 8, llama 3 | spread |
| Louis Daguerre | 20 | 8 (30%) | 0 (0%) | 3 (12%) | 4 (16%) | 5 (26%) | pt 40% | 22% | 1.8 | qwen 8 | spread |
| Nicéphore Niépce | 8 | 4 (15%) | 1 (4%) | 2 (8%) | 1 (4%) | 0 (0%) | pt 50% | 22% | 2.2 | qwen 4 | **Concentrated** |
| Émile Reynaud | 5 | 2 (7%) | 0 (0%) | 0 (0%) | 0 (0%) | 3 (16%) | fr 60% | 16% | 3.8 | llama 3 | **Concentrated** |

### calculo

No-candidate answers per language: pt 4, en 0, de 7, it 12, fr 20 (total 43).  
Models: pt: llama 2, mistral 1, qwen 1; de: llama 7; it: llama 12; fr: llama 19, mistral 1.

| Person | Mentions | pt | en | de | it | fr | Top | Baseline | Ratio | Top-language models | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| Bonaventura Cavalieri | 41 | 3 (75%) | — | 7 (100%) | 11 (92%) | 20 (100%) | fr 49% | 47% | 1.0 | llama 19, mistral 1 | spread |
| Archimedes | 31 | 2 (50%) | — | 7 (100%) | 6 (50%) | 16 (80%) | fr 52% | 47% | 1.1 | llama 16 | spread |
| Pierre de Fermat | 21 | 1 (25%) | — | 5 (71%) | 2 (17%) | 13 (65%) | fr 62% | 47% | 1.3 | llama 12, mistral 1 | spread |

### fotografia

No-candidate answers per language: pt 0, en 1, de 1, it 1, fr 1 (total 4).  
Models: en: mistral 1; de: qwen 1; it: gemma 1; fr: llama 1.

No person with 5 or more mentions.

### tabela_periodica

No-candidate answers per language: pt 0, en 0, de 1, it 2, fr 0 (total 3).  
Models: de: llama 1; it: llama 1, qwen 1.

No person with 5 or more mentions.

## 5. Share of answers with no candidate (fact × model × language)

All answers (greedy and samples) of the models with behavioral data. Cell: *share (answers with no candidate / answers)*; — when there are none. Source: `results/metrics/behavioral_none.csv` (`kind == all`).

| Fact | Model | PT | EN | DE | IT | FR | All |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aviao | llama | 52% (17/33) | 3% (1/33) | 42% (14/33) | 82% (27/33) | 70% (23/33) | 50% (82/165) |
| aviao | mistral | — | — | 15% (5/33) | 9% (3/33) | 6% (2/33) | 6% (10/165) |
| aviao | qwen | — | — | 33% (11/33) | — | — | 7% (11/165) |
| aviao | gemma | — | — | — | — | — | 0% (0/165) |
| aviao | gemma27 | — | — | — | — | — | 0% (0/165) |
| telefone | llama | — | — | — | — | — | 0% (0/165) |
| telefone | mistral | — | — | — | — | — | 0% (0/165) |
| telefone | qwen | — | — | — | — | — | 0% (0/165) |
| telefone | gemma | — | — | — | — | — | 0% (0/165) |
| telefone | gemma27 | — | — | — | — | — | 0% (0/165) |
| radio | llama | — | 3% (1/33) | 6% (2/33) | 3% (1/33) | — | 2% (4/165) |
| radio | mistral | — | 3% (1/33) | 6% (2/33) | — | 3% (1/33) | 2% (4/165) |
| radio | qwen | 9% (3/33) | — | 6% (2/33) | 6% (2/33) | 6% (2/33) | 5% (9/165) |
| radio | gemma | — | — | — | — | — | 0% (0/165) |
| radio | gemma27 | — | — | — | — | — | 0% (0/165) |
| cinema | llama | 48% (16/33) | 61% (20/33) | 61% (20/33) | 45% (15/33) | 33% (11/33) | 50% (82/165) |
| cinema | mistral | 3% (1/33) | — | 6% (2/33) | 9% (3/33) | — | 4% (6/165) |
| cinema | qwen | 30% (10/33) | 12% (4/33) | 12% (4/33) | 21% (7/33) | 24% (8/33) | 20% (33/165) |
| cinema | gemma | — | — | — | — | — | 0% (0/165) |
| cinema | gemma27 | — | — | — | — | — | 0% (0/165) |
| calculo | llama | 6% (2/33) | — | 21% (7/33) | 36% (12/33) | 58% (19/33) | 24% (40/165) |
| calculo | mistral | 3% (1/33) | — | — | — | 3% (1/33) | 1% (2/165) |
| calculo | qwen | 3% (1/33) | — | — | — | — | 1% (1/165) |
| calculo | gemma | — | — | — | — | — | 0% (0/165) |
| calculo | gemma27 | — | — | — | — | — | 0% (0/165) |
| fotografia | llama | — | — | — | — | 3% (1/33) | 1% (1/165) |
| fotografia | mistral | — | 3% (1/33) | — | — | — | 1% (1/165) |
| fotografia | qwen | — | — | 3% (1/33) | — | — | 1% (1/165) |
| fotografia | gemma | — | — | — | 3% (1/33) | — | 1% (1/165) |
| fotografia | gemma27 | — | — | — | — | — | 0% (0/165) |
| tabela_periodica | llama | — | — | 3% (1/33) | 3% (1/33) | — | 1% (2/165) |
| tabela_periodica | mistral | — | — | — | — | — | 0% (0/165) |
| tabela_periodica | qwen | — | — | — | 3% (1/33) | — | 1% (1/165) |
| tabela_periodica | gemma | — | — | — | — | — | 0% (0/165) |
| tabela_periodica | gemma27 | — | — | — | — | — | 0% (0/165) |
| penicilina | llama | — | — | — | — | — | 0% (0/165) |
| penicilina | mistral | — | — | — | — | — | 0% (0/165) |
| penicilina | qwen | — | — | — | — | — | 0% (0/165) |
| penicilina | gemma | — | — | — | — | — | 0% (0/165) |
| penicilina | gemma27 | — | — | — | — | — | 0% (0/165) |
| dinamite | llama | 6% (2/33) | — | 18% (6/33) | 3% (1/33) | — | 5% (9/165) |
| dinamite | mistral | — | — | — | — | — | 0% (0/165) |
| dinamite | qwen | — | — | 3% (1/33) | — | — | 1% (1/165) |
| dinamite | gemma | — | — | — | — | — | 0% (0/165) |
| dinamite | gemma27 | — | — | — | — | — | 0% (0/165) |
| www | llama | — | — | — | — | — | 0% (0/165) |
| www | mistral | — | — | — | — | — | 0% (0/165) |
| www | qwen | — | — | — | — | — | 0% (0/165) |
| www | gemma | — | — | — | — | — | 0% (0/165) |
| www | gemma27 | — | — | — | — | — | 0% (0/165) |
| raios_x | llama | — | — | — | — | — | 0% (0/165) |
| raios_x | mistral | 3% (1/33) | — | — | — | — | 1% (1/165) |
| raios_x | qwen | — | — | — | — | — | 0% (0/165) |
| raios_x | gemma | — | — | — | — | — | 0% (0/165) |
| raios_x | gemma27 | — | — | — | — | — | 0% (0/165) |

## 6. Script and language switches

`behavior/switches.py`, over all answers. **Script switch:** a run of at least 2 characters in a non-Latin script (Han, kana, hangul, Cyrillic, Arabic, Hebrew, Devanagari, Thai; Greek excluded because it appears in formulas). **Language switch:** the answer's dominant language (by distinctive function words, with a margin; otherwise undetermined) is not the prompt's. **Partial:** at least one sentence is dominantly in another of the five languages. Source: `results/metrics/behavioral_switches.csv`.

Totals: 48 script switches, 5 whole-answer language switches and 8 answers with at least one sentence in another language (these include the whole-answer ones), out of 9075 answers. Among the 302 no-candidate answers: 4 script switches and 1 language switch.

| Model | Prompt language | Answers | Script switches (scripts) | Whole-answer language switches | Partial (to) |
| --- | --- | ---: | --- | ---: | --- |
| gemma27 | de | 363 | 0 | 1 | 1 (en) |
| gemma27 | it | 363 | 1 (Arabic) | 0 | 0 |
| llama | de | 363 | 1 (Han) | 0 | 0 |
| llama | pt | 363 | 1 (Han) | 0 | 0 |
| mistral | de | 363 | 1 (Cyrillic) | 0 | 1 (en) |
| mistral | fr | 363 | 1 (Cyrillic) | 0 | 0 |
| mistral | it | 363 | 1 (Cyrillic) | 0 | 0 |
| mistral | pt | 363 | 0 | 0 | 1 (en) |
| qwen | de | 363 | 11 (Han) | 1 | 1 (en) |
| qwen | en | 363 | 2 (Han) | 0 | 0 |
| qwen | fr | 363 | 7 (Cyrillic,Han,Kana) | 0 | 0 |
| qwen | it | 363 | 12 (Han) | 0 | 1 (en) |
| qwen | pt | 363 | 10 (Han) | 3 | 3 (en) |

Model × language cells with no switch are omitted.
