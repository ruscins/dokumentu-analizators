# 📊 Dokumentu Analīzes Asistents

## Pārskats

Dokumentu Analīzes Asistents ir jaudīga Streamlit lietotne, kas paredzēta visaptverošai datu analīzei un ieskatu iegūšanai no CSV, Excel un PDF failiem. Lietotne piedāvā progresīvas funkcijas:

- Interaktīva datu vizualizācija
- Detalizēta statistiskā analīze
- Automātiska ieskatu ģenerēšana
- Mākslīgā intelekta atbalstīta dokumentu izpratne

## Īpašības

### Atbalstītie failu formāti
- CSV faili
- Excel faili (.xls, .xlsx)
- PDF dokumenti

### Galvenās funkcionalitātes

1. **Datu vizualizācija**
   - Vairāki diagrammu veidi:
     - Stabiņu diagrammas
     - Līniju grafiki
     - Izkliedes diagrammas
     - Histogrammas
     - Kastveida diagrammas

2. **Statistiskā analīze**
   - Pilnīgs datu kopsavilkums
   - Trūkstošo vērtību noteikšana
   - Skaitlisko datu detalizēta izpēte

3. **Automātiska datu izpēte**
   - Automātiski secinājumi
   - Korelācijas analīze
   - Tendenču identifikācija

4. **Mākslīgā intelekta asistents**
   - Dokumentu kopsavilkums
   - Dziļa dokumentu analīze
   - Atbildes uz lietotāja jautājumiem

## Nepieciešamās priekšzināšanas

### Tehniskās prasības
- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Google GenAI bibliotēka (pēc izvēles)

### Vides mainīgie
- `GOOGLE_API_KEY` - Google Gemini API atslēga (ja vēlaties izmantot AI funkcijas)

## Instalācija

1. Klonējiet repozitoriju
```bash
git clone https://github.com/jūsu-lietotājvārds/dokumentu-analize.git
cd dokumentu-analize
```

2. Izveidojiet virtuālo vidi
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
# vai
venv\Scripts\activate  # Windows
```

3. Instalējiet nepieciešamās bibliotēkas
```bash
pip install -r requirements.txt
```

4. Palaišana
```bash
streamlit run app.py
```

## Konfigurācija

- Iestatiet `GOOGLE_API_KEY` `.env` failā vai kā vides mainīgo
- Pielāgojiet `USE_GENAI` mainīgo, lai kontrolētu AI funkcionalitāti

## Drošība

- Lietotne automātiski dzēš augšupielādētos failus pēc sesijas
- Iespēja manuāli dzēst datus

## Atbalsts

Ja jums ir kādas problēmas vai ierosinājumi:
- Atveriet "issues" GitHub repozitorijā
- Sūtiet e-pastu [jūsu-e-pasts]

## Licenze

[Norādiet jūsu projekta licenci, piemēram, MIT]

## Tiešsaistes versija

Varat izmēģināt lietotni uzreiz tiešsaistē: 
[📊 Dokumentu Analīzes Asistents](https://gqzs2oy7dslywwmcvyffv8.streamlit.app/)

---

**Piezīme:** Šī lietotne ir izstrādāta izglītības un pētniecības nolūkiem. 
Lūdzu, izmantojiet to atbildīgi un ievērojot datu privātumu.
