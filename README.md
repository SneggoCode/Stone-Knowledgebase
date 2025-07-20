# Stone Knowledgebase Manager

Dieses Programm hilft dabei, eine Knowledgebase im CSV-Format zu pflegen.
Es richtet sich speziell an die Nutzung mit einem Vapi Voice Agent.

## Verwendung

1. Python 3.10+ installieren.
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
3. Programm starten:
   ```bash
   python kb_manager.py
   ```
4. Nach dem Start werden alle bestehenden Einträge rechts im Fenster angezeigt
   und lassen sich durchscrollen.
5. Die Felder ausfüllen und auf **Ähnliche Fragen anzeigen** klicken, um
   vorhandene oder ähnliche FAQ-Einträge zu suchen. Die Tabelle scrollt
   automatisch zum besten Treffer.
6. Mit **Speichern** wird der Eintrag in `knowledgebase.csv` angehängt und in
   der Tabelle markiert.

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
