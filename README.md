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

## Adam

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

## Miso

### Data labeling pomocou scrapingu - DONE (možno, keď sa podarí nájsť aj úspešnosť žiakov) 
### Klasifikácia katedier pomocou neuronóvej siete

## Max

### Predikcia veku pomocou X

### Predikcia poctu titulov pomocou X

## Mirka

### Clustering podľa podobnosťi

## Ondro

### Klasifikácia pohlavia pomocou SVM

## Nástroje

- https://www.downthemall.net
-

## Zdroje

### Dáta

- https://sluzby.fmph.uniba.sk/f
- https://fmph.uniba.sk/pracoviska/

### Analýza

- https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm
- 