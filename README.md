# Stone Knowledgebase Manager

Dieses Programm hilft dabei, eine Knowledgebase im CSV-Format zu pflegen.
Es richtet sich speziell an die Nutzung mit einem Vapi Voice Agent.

## Datenstruktur

Alle Fragen und Antworten werden in einer einzigen Datei `knowledgebase.csv`
gespeichert. Die Tabelle besitzt folgende Spalten:

```
category, faq_question, answer_text, stone_type,
product_form, grain_size_mm, eigenschaft, anwendung
```

`category` unterscheidet z.B. `product`, `payment`, `delivery`,
`installation` oder `warranty`. Die übrigen Felder können leer bleiben,
wenn sie nicht zum jeweiligen Eintrag passen.

## Verwendung

1. Python 3.11 oder neuer installieren.
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
   Die Datei benötigt `openai>=1.0`. Das Programm verwendet bereits die
   neue clientbasierte Schnittstelle der Bibliothek.
3. Programm starten (benötigt eine grafische Oberfläche):
   ```bash
   python kb_manager.py
   ```
   In einer Kopf-losen Umgebung kann ein virtuelles Display wie
   `Xvfb` verwendet werden.
4. Nach dem Start werden alle bestehenden Einträge rechts im Fenster angezeigt
   und lassen sich durchscrollen.
5. Die Felder ausfüllen und auf **Ähnliche Fragen anzeigen** klicken, um
   vorhandene oder ähnliche FAQ-Einträge zu suchen. Die Tabelle scrollt
   automatisch zum besten Treffer.
6. Mit **Speichern** wird der Eintrag in `knowledgebase.csv` angehängt oder –
   wenn vorher ein Datensatz geladen wurde – aktualisiert.
7. Über **Eintrag laden** kann eine markierte Zeile zum Bearbeiten in das
   Formular übernommen werden.
8. Mit **Eintrag löschen** wird die ausgewählte Zeile dauerhaft entfernt.
9. **Neu** leert das Formular ohne zu speichern.

## KI-Unterstützung

Neben dem manuellen Einpflegen kann das Programm aus beliebigem Text neue
FAQ-Vorschläge generieren. Dazu muss einmalig ein OpenAI-API-Key hinterlegt
werden. Der Key kann per Umgebungsvariable `OPENAI_API_KEY` oder nach
Programmstart über den Button **API-Key eingeben** gesetzt werden. Anschließend kann im Feld
**Text für KI-Vorschläge** ein beliebiger Fließtext eingefügt und mit
**Vorschläge generieren** analysiert werden.
Die gefundenen Vorschläge erscheinen unterhalb des Formulars und lassen sich
einzeln in die Felder übernehmen, editieren und schließlich speichern.

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
