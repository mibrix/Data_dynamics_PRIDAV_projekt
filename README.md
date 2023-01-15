# Projekt na Pridav

Analýzu hlavných komponentov je možné aplikovať aj na dáta ako sú fotografie.
Pokúsme sa pomocou nej vyhľadávať skupinky ľudí, ktorí sa navzájom podobajú alebo skúsme
vymyslieť iné použitia

## Dáta

V adresári `photos` sú dáta, ktoré budeme používať.
Tieto dáta boli stiahnuté zo stránky fakulty, ktorá ich zverejnila na svojom webe.

Tieto dáta boli uložené vo verejne dostupnom adresáry na webe fakulty ,
tým padom sme ich nemuseli scrapovať ale bolo ich možné priamo hromadne stiahnuť
pomocou 1000 a 1 hromadných download manažérov.

## Adam Chrenko

### Preprocessing

Na preprocessing som naprv použil knižnicu opencv, pomocou ktorej som obrázky načítal v ČB.
Ďalej som použil známu metódu na detekciu objektov, ktorá sa nazýva haar cascade.

#### Haar cascade

1. Táto metóda používa jednoduché vizuálne príznaky na detekciu objektov.
2. Tieto príznaky sú trénované na množine obrázkov, ktoré obsahujú objekty, ktoré chceme detekovať (v tomto prípade
   tváre).
3. Príznaky sú potom aplikované na neznáme obrázky, ak nádjeme dostatok zhôd, tak je to objekt klasifikovaný ako tvár.

- Vedia fungovať veľmi rýchlo, a preto sú vhodné aj pre real-time detekciu (Video).

Táto metóda vráti nám množinu obdĺžnikov, ktoré obsahujú tváre.
Keďze vieme, že obrázok vždy obsahuje len jednu tvár, tak vieme využiť heuristiku na odstránenie falošných detekcií.

- Obdĺžniky sú pretriedené podľa veľkosti, V poradí od najväčšieho po najmenší.
- Vyberie sa ten najvačší, ktorý je považovaný za reálnu tvár.

Táto kombinácia sa ukázala byť dostatočná na použitie na našich dátach.

Ako posledný processing krok sa tváre zmenili na rovnaké rozmery (100x100).

Následne sa tváre uložili do adresára `faces` ako ČB obrázky rozmerov 100x100 s originálnym názvom.

### PCA

Na implementáciu PCA som postupoval podľa linku, ktorý bol poskytnutý v zadaní.   
Rýchle intro do PCA:

- PCA je metóda, ktorá slúži na redukciu dimenzionality dát.
- PCA zachováva dimenzie, ktoré majú najväčšiu varianciu. A odstraňuje tie, s najmenšou varianciou.
- Toto znamená, že PCA sa snaží zachovať najväčšiu informáciu o dátach, a zároveň odstrániť najmenej potrebné atribúty
  dát. Ak sú napríklad všetky dáta na jednej rovine, tak PCA túto rovinu zachová, a odstráni tretiu dimenziu.
- Existújú metódy na zistenie, že o okľko môžeme dáta redukovať a zároveň zachovať X% informácie. V našom prípade sme
  takéto metódy nepoužili, a nastavovali sme počet komponentov ako hyperparameter.

Čo ideme robiť je, že tváre zredukujeme na k najpodstatnejších komponentov, a potom tieto komponenty použijeme na
rôzne účely.

#### PCA *Training*
Na Tréning som použil všetky tváre, ktoré som získal z preprocessingu. Tieto tváre som zmenil z 2D obrázku nxn na 
1D vektor dĺžky n^2. Tieto vektory som potom použil ako vstup pre PCA.

1. Najrpv som potreboval vypočítať priemernú tvár. Táto tvár bola vypočítaná ako priemer pixelov
všetkých tŕeningových tvári.
2. Potom som znormalizoval všetky tváre tak, že som od každej tváre odčítal priemernú tvár.
3. Tieto normalizované tváre potom boli naskladané do matice.
4. Z tejto matice som vypočítal tzv. redukovanú kovariančnú maticu. ako A^T . A
5. Táto matica bola použitá ako vstup pre funkciu z numpy, na výpočet vlastných vektorov a vlastných čísel.
6. Vektory som zoradil podľa vlastných čísel, a vybral som prvých k najvačších vektorov.
7. Normalizoval som vektory tak, že som ich delil ich normou.
8. Váhy boli vypočítané ako A . Vektory_K
9. Váhy, Priemerná tvár, Vektory_K a zoradené názvy tvári boli uložené do súboru `PCA.npz`, ktorý budeme ďalej používať.
10. Nakoniec som zrekonštruoval treningové tváre pomocou váh a vlastných vektorov. Tieto tváre som uložil do
    priečinka `reconstructed`, ako ČB obrázky rozmerov 100x100 s originálnym názvom.

#### PCA *Testing*
Ako testovacie dáta som najpr skúsil použiť obrázky, ktoré boli aj v tréningových dátach. A potom aj na iných obrázkoch tých istých osôbz webu  
Postup pre predikciu neznámej tváre je nasledovný:
1. Tieto tváre som prehnal rovnakým preprocessingom ako pri tréningu.
2. Od každej tváre som odčítal priemernú tvár.
3. Váhy boli vypočítané ako A . Vektory_K
4. Následne pomocou LX normy boli vypočítané vzdialenosti medzi testovacou tvárou a všetkými tréningovými tvármi.
5. Vzdialenosti boli zoradené a vybrané prvých 5 najmenších, mená ktorých dáme na STDOUT.
6. Zároveň sa pokúsim nájsť index trénovacej tváre, ktorá reprezentuje správnu osobu. (Toto záleží na korektnom pomenovaní testovacích tvári)
Toto + vzdialenosť od nej sa dá na STDOUT.
7. Nakoniec zobrazím zrekonštruované tváre + ich mená:  
    - Testovaciu tvár (Vľavo hore)
    - Originálnu trénovaciu tvár (Vľavo dole)
    - Predikciu 1 (Vpravo hore)
    - Predikciu 2 (Vpravo dole)

#### PCA *Testing* - *Výsledky*
Hyperparametre:
- k : neprejavil veľký vplyv na správnosť predikcie
- použitá norma na vzdialenosť : neprejavila veľký vplyv na správnosť predikcie

Výsledky:
- Na testovacích tvárach, ktoré boli aj v tréningových dátach, som dosiahol veľmi dobré výsledky. Presne ako som očakával.
  Vzdialenosti však neboli nulové, ale boli signifikantne menšie ako všetky ostatné vzdialenosti. (Pravdepodobne spôsobené nie perfektnou detekciou tváre v preprocesingu)

- Pri testovaní na nových dátach, výsledky boli mizerné, takmer žiadna predikcia nebola správna. 
Vačšinu času sa správna osoba nedostala ani do TOP 25 predikcií.
- Podľa mňa toto mohlo byť spôsobené napríklad nasledujúcimi problémami:
    - Naklonením tváre : normalizacia rotácie bola skúsená
  (bohužial rozlíšenie týchto obrázkov je príliž nízke na spoľahlivú 
  detekciu očí a iných prízakov, ktoré by s týmto mohli pomôcť)
    - Zmena uhlu kamery
    - Zmena osvetlenia
    - Zmena tváre (napr. zmena výrazu, brada, ...)
    - Drastická zmena vzhľadu (napr. vačšia zmena veku, ...)

Na všetky tieto problémy je PCA náchylné, pretože sa vypočítava len váh a neberie sa do úvahy susednosť pixelov atd.
Čo by pomohlo:
- Viacero obrázkov z jednej osoby idealne v rôznych pozíciách, s rôznym osvetlením, s rôznym výrazom, ...
- Obrázky vyžsieho rozlíšenia, na riadne pred spracovanie tváre, ideálne .PNG
- Použitie iného predikčného modelu, ako napríklad SVM, ktorý vie používať nelineárne rozhodovacie hranice.

## Michal Chymo

### Data labeling pomocou scrapingu https://fmph.uniba.sk/pracoviska/
Samotný scraping robený pre zamestnancov fakulty z uvedenej webstránky. Do scv a db súboru som uložil výsledky scrapingu, podľa toho či má zamestnanec fotku.
Samtoné dolované dáta sú: celé meno s titulmi, katedra a oddelenie v ktorom pracuje, samotná funkcia zamestnanca.
Jedna stránka mala iný html kód než druhé, preto v kóde je naviac if-else klauzula.
Priečinok ./scraping
.py súbor je samotný kód scrapingu a .sql je sql kód pre .db súbor. Tu filtrujem dáta aby z každej skupiny bolo aspoň 10 a vytváram unikatné označenie pre každú katedru.



### Klasifikácia katedier pomocou neuronóvej siete (https://www.tensorflow.org/tutorials/keras/classification)
Zo scrapnutých dát som sa pokúsil vytvoriť neurónku pre klasifikáciu katedry, v ktorej zamestnanec pracuje. Je ich 277, čiže veľa ich veľmi nie je. Rozdelil som ich na 250 trénovacích a 27 testovacích.
Ako prvé som zobral dáta zo samotnej fotky (vektor všetkých pixelov) a najväčšiu úspešnosť klasifikácie na testovacích som mal okolo 60% so skúšaním rôznych sietí.
Potom som použil vektory z PCA ako vstup. Výhoda je, že namiesto 100*100 rozmerného vstupu, mám len 25 rozmerný vstup(k-rozmerny z PCA). Takže trénovanie je oveľa rýchlejšie a takisto je možné lepšie experimentovať so sieťou.
Nanajvýš sieť dosahovala na testovacích dátach podobnú úspešnosť ako pri vstupe s celou fotkou.

Priečinok ./neural


## Maximilián Zivák
### Predikcia počtu titulov podľa random forest classifieru a neuronovej siete
#### Problém
Ako zaujímavá aplikácia klasifikácie ľudských tvári je predikcia veku, keďže dáta ohľadom veku sa nedajú  tak ľahko získať, rozhodol som sa tento problém redukovať na počet titulov (očakávam že vyšší vek bude korelovaný s väčším počtom titulov).
#### Postup
Najprv bude potrebné scrapenuť dáta zo stránky fakulty, tento proces je celkom triviálny (časť scraping). Celé meno s počtom titulov bude treba potom namapovať na už spravenú PCA (časť PCA). Po namapovaní jednoduchou algebrou vieme spojiť dáta do jednej matice a s pomocou ML knižníc vieme dáta rozdeliť na tréningové a testovacia. Na koniec len zvolíme model ktorý chceme použiť, rozhodol som sa pre random forest classifier kvôli malému množstvu dát a pre neurónovú sieť.
#### Dáta
Vidíme že počet titulov sa veľmi líši ({3.0: 71, 2.0: 50, 1.0: 34, 0.0: 10}) (pri random test splite so seedom 666), najprv vyskúšame random forest a neuronovu sieť bez aplikácie over a undersamplingu, potom sa pozrieme ako sa accuracy zmení ak duplikujeme dáta.
### No oversampling/undersampling
#### Random forest
Random forest má mnoho parametrov ktorá sa dajú nastavovať, jednoduchý grid search vrátil najlepšie parametre (s tým že random state bol prednastavený na 4646) {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': False}
Random forest s týmito parametrami má úspešnosť 0.49. S takto malým datasetom som sa ani nezaoberal rýchlosťou.
#### Neuronova siet
Tiež som sa rozhodol vyskúšať neurónovú sieť, dáta predspracujem rovnako ako pri random foreste (scraping etc.), rozhodol som sa spraviť 2 vrstvy, obidve sigmoid. Ostatné parametre som ponechal rovnaké ako v učebnicovom príklade na klasifikáciu obrázkov. Po vycvičení na 100 epochách má neuronova sieť úspešnosť 50%~.
### Záver
Random forest ani neuronova sieť nedokázali dostatočne dobre predikovať počet titulov podľa tváre ktorá prešla PCA. Ako jedno z vysvetlení je málo dát alebo nedostatočná korelácia medzi počtom titulov a tvárou, napr. je viac zamestnancov čo budú starší teda ich rozdiel tvári nebude tak veľký ako medzi mladším a starším kolegom etc.
### Oversampling and undersampling
#### Random forest
Keďže máme málo dát rozhodol som sa použiť len oversampling, a to zvýšením počtu entries pre ľudí s 0 a 1 titulom na 42 (medián), po aplikovaní najlepší parametrov dostanem accuracy 0.53, čo je veľmi malé zlepšenie.
#### Neuronova siet
Po upravení parametrov neurónovej siete má horšiu performance ako neuronka pred oversamplingom

## Myroslava Hrechyn

### Clustering podľa podobnosťi
#### clustering.ipynb

Na základe váh, získaných pomocou PCA, chceli sme rozdeliť dáta do clusterov.

Pri danej metóde je potrebne vopred zadať počet zhlukov, avšak v našom prípade bolo ťažko povedať nejaké konkrétne číslo. Preto sme použili Elbow method. Z grafu je vidieť, zem „lakeť“ je približne okolo hodnoty 50. Aby sme to overili, vykreslili sme aj silhoette score, maximálna hodnota grafu by mala zodpovedať „najlepšej“ hodnote n, avšak v danom prípade to nebolo jasne.
Pre daný počet clusterov sme získali 32 clustery, ktoré obsahovali 6+ ľudí a následne sme ich aj vykreslili.

Rozhodli sme sa skúsiť aj inú metódu, ktorá nepotrebuje vopred zadaný počet clusterov, napríklad Affinity Propagation, ktorá sa nachádza v knižnici sklearn. Dostali sme 28 clusterov, v 23 z nich bolo 7 a viac ľudí.

Porovnávali sme dane metódy pomocou metriky adjusted_rand_score, ktorá je symetrická a nadobúda hodnotu 0, ak hodnoty sú náhodne, a 1, ak rozdelenie do clusterov je identické. V našom prípade hodnota bola 0.4096, čo hovorí o tom, že rozdelenie do clusterov pomocou pôvodného KMeans(50) a AffinityPropation su celkom podobne.


## Ondrej Gajdoš

### Klasifikácia pohlavia

#### Labeling

V prvom rade bolo potrebné olabelovať fotky podľa pohlavia. Riešenie tohto problemu bolo hybridné. Najprv program označí všetky fotky s názvom(priezvisko) so sufixom "ova" ako osoby ženského pohlavia číselnou hodnotou 0 a všetky ostané ako osoby mužského pohlavia číselnou hodnotou 1. Následne som prešiel všetky fotky a určil osoby ženského pohlavia ktorých priezvisko nemá sufix "ova".

Dataset disponuje celkovým 348 fotkami z čoho 228 (cca 66%) fotiek obsahuje osobu mužského pohlavia a 120 (cca 34%) osobu ženského pohlavia. Dataset má teda nevyvážené triedy.

#### Hľadanie najlepšieho modelu

Keďže je dataset nevyvážený je na mieste využiť resamplovacie metódy. Konkrétne:
- Random undersampling: náhodne vymaže dáta z viac početnej triedy tak aby počty dát v triedach rovnali
- Random oversampling: náhodne duplikuje dáta z menej početnej triedy tak aby počty dát v triedach rovnali
- SMOTE: vytvorí hrany medzi dátami menej početnej triedy a na tieto hrany náhodne pridá dátové body
- ADASYN: od metódy SMOTE sa líši iba v tom, že nové umelé dáta môžu byť mierne vychýlené od hrany medzi bodmi

Na trénovanie modelu som použil vektory váh (ktoré slúžia na rekonštrukciu jednotlivých fotiek).

Modely ktorých výkon som porovnával:
- Logistická regresia
- Random forest
- SVC s RFC (Radial basis function) kernelom
- Decision tree

Samotné hľadanie spočíva v trénovaní vyššie spomenutých modelov za použitia vyššie spomenutých resamplovacích metód a rôznych veľkostí bázy eigenvektorov (5,10,20,30,40,50,70,90,110,135,160,190,250,300).

Príklad iterácie hľadania:
1. Zober vektory váh o dĺžky 5
2. Použi jednu z resamplovacích metód na vstupné dáta
3. Rozdel dataset na trénovací (70%) a testovací (30%)
4. Natrénuj modely
5. Zaznamenaj F1 skóre modelu

Pri resamplovacích metódach, ktoré dáta vytvárajú umelo (ADASYN a SMOTE) sa kroky 2. a 3. robia v opačnom poradí aby sa umelo vytvorené dáta nedostali do validačnej množiny.

Program vytvorí 100 F1 skór pre každu kombináciu (veľkosť bázy eigenvektorov, resamplovacia metóda, model). Na vizualizáciu výsledkov som využil boxploty, ktoré ukázali, že najlpešiu výkonnosť dosahuje SVC pri všetkých resamplovacích metódach okrem random under-samplingu. Ako veľmi dobrá kombinácia sa tiež ukázala Random forest + over-sampling. Medzi vymenovanými kombináciami nie je veľmi markantný rozdiel, avšak kombinácia SVC + SMOTE pri veľkostiach báz 40 a vyššie najčastejšie prekračuje strednou hodnotou F1 skóre 0.93 a zároveň má relatívne nízku varianciu čo by danú kombináciu mohlo klasifikovať ako najlepšiu.


## Nástroje

- https://www.downthemall.net
-

## Zdroje

### Dáta

- https://sluzby.fmph.uniba.sk/f
- https://fmph.uniba.sk/pracoviska/

### Analýza

- https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm
- https://github.com/dim4o/gender-recognizer/blob/master/Gender%20Classification.ipynb
- https://imbalanced-learn.org/stable/over_sampling.html
- https://www.tensorflow.org/tutorials/keras/classification
- https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
- https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
