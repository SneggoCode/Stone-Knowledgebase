# Stone Knowledgebase Manager

Dieses Programm hilft dabei, eine Knowledgebase im CSV-Format zu pflegen.
Es richtet sich speziell an die Nutzung mit einem Vapi Voice Agent.

## Datenstruktur

Alle Fragen und Antworten werden in einer einzigen Datei `knowledgebase.csv`
gespeichert. Die Tabelle besitzt folgende Spalten:

```
category, faq_question, answer_text, stone_type,
product_form, product_size, eigenschaft, anwendung
```
`product_size` sollte immer die passende Einheit enthalten, beispielsweise
`8/16 mm` oder `30 cm`.

`category` unterscheidet z.B. `product`, `payment`, `delivery`,
`installation` oder `warranty`. Die übrigen Felder können leer bleiben,
wenn sie nicht zum jeweiligen Eintrag passen.
Beim Speichern überprüft eine kleine Pydantic-Datenklasse die Eingaben und
ergänzt fehlende Einheiten bei `product_size` automatisch.

## Verwendung

1. Python 3.11 oder neuer installieren.
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
   Die Datei benötigt `openai>=1.0`. Das Programm verwendet bereits die
   neue clientbasierte Schnittstelle der Bibliothek. Zusätzlich werden
   `numpy`, `pydantic` und `ttkbootstrap` genutzt.
3. Programm starten (benötigt eine grafische Oberfläche):
   ```bash
   python kb_manager.py
   ```
   In einer Kopf-losen Umgebung kann ein virtuelles Display wie
   `Xvfb` verwendet werden.
4. Nach dem Start werden alle bestehenden Einträge rechts im Fenster angezeigt
   und lassen sich durchscrollen oder filtern. Eine Suchleiste über der Tabelle
   ermöglicht das schnelle Auffinden von Fragen.
5. Die Felder ausfüllen und auf **Ähnliche Fragen anzeigen** klicken, um
   vorhandene oder ähnliche FAQ-Einträge zu suchen. Die Tabelle scrollt
   automatisch zum besten Treffer.
6. Mit **Speichern** wird der Eintrag in `knowledgebase.csv` angehängt oder –
   wenn vorher ein Datensatz geladen wurde – aktualisiert.
7. Über **Eintrag laden** kann eine markierte Zeile zum Bearbeiten in das
   Formular übernommen werden.
8. Mit **Eintrag löschen** wird die ausgewählte Zeile dauerhaft entfernt. Über
   **Rückgängig Löschen** lässt sich der letzte gelöschte Datensatz wiederherstellen.
9. **Neu** leert das Formular ohne zu speichern.
10. Durch einen Doppelklick auf eine Tabellenzeile wird der Eintrag direkt zum Bearbeiten geladen.
11. Ein Klick auf einen Spaltenkopf sortiert die Tabelle.
12. Unten links zeigt das Fenster die aktuelle Versionsnummer an.
13. Ein Dark-Mode-Button wechselt zwischen hellem und dunklem Stil.
14. Rechtsklick-Menüs und eine Toolbar bieten schnellen Zugriff auf Laden,
    Speichern und Löschen. Ein globaler Hotkey **Strg+S** speichert den
    aktuellen Eintrag.
15. Die Eingabefelder sind in klappbare Bereiche **Meta-Daten** und
    **FAQ-Inhalt** gegliedert. Das Programm verwendet überall die Schriftart
    Aptos in Größe 12 pt.
16. Beim Tippen in die Filterzeile startet automatisch eine Live-Suche. Ein
    Fortschrittsbalken erscheint, wenn sie länger dauert.
17. Tooltips erklären die Eingabefelder. Die Texte liegen in
    `ui_tooltips.json` und lassen sich leicht anpassen.

Nach Installation der Abhängigkeiten können automatisierte Tests mit
`pytest` ausgeführt werden. Zusätzlich sorgt `flake8` für einheitlichen
Code-Stil:

```bash
pytest
flake8
```

## KI-Unterstützung

Neben dem manuellen Einpflegen kann das Programm aus beliebigem Text neue
FAQ-Vorschläge generieren. Dazu muss einmalig ein OpenAI-API-Key hinterlegt
werden. Der Key kann per Umgebungsvariable `OPENAI_API_KEY` oder nach
Programmstart über den Button **API-Key eingeben** gesetzt werden.

Im Feld **Text für KI-Vorschläge** kann beliebiger Text eingefügt werden.
Mit **Vorschläge generieren** analysiert die KI den Inhalt und erstellt immer
bis zu fünf passende FAQ-Einträge. Fehlen im Text wichtige Angaben, erfindet
die KI eigenständig sinnvolle Fragen und lässt unklare Felder leer.

Für die Suche nach ähnlichen Fragen nutzt das Programm OpenAI-Embeddings
anstelle klassischer TF-IDF-Vektoren. Dadurch werden auch semantisch passende
Einträge gefunden.

Alle generierten Vorschläge sammeln sich in einer Liste unterhalb des
Formulars. Ein Doppelklick übernimmt einen Vorschlag in die Eingabefelder und
entfernt ihn aus der Liste. Nicht benötigte Elemente können über
**Vorschlag löschen** entfernt werden. Neue Aufrufe von
**Vorschläge generieren** fügen weitere Einträge an, die bestehenden bleiben
erhalten, bis sie geladen, gespeichert oder gelöscht wurden.

Wird die Datei nicht gefunden, wird sie automatisch mit der passenden
Spaltenstruktur erstellt.

## Erstellung einer Windows-Exe

Nach Installation der Abhängigkeiten kann mit [PyInstaller](https://pyinstaller.org/) eine
executable erstellt werden:

```bash
pip install pyinstaller
pyinstaller --onefile kb_manager.py
```

Die fertige `kb_manager.exe` liegt anschließend im Ordner `dist` und
kann auf einem Windows 11 Rechner ausgeführt werden.
