import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
import tkinter as tk  # noqa: E402
import pytest  # noqa: E402
from ttkbootstrap import Window  # noqa: E402
from kb_manager import KBManager  # noqa: E402


def display_available():
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


@pytest.mark.skipif(not display_available(), reason="requires display")
def test_button_row(monkeypatch):
    monkeypatch.setattr('language_tool_python.LanguageTool', lambda *_: None)
    root = Window(themename='flatly')
    app = KBManager(root)
    buttons = [app.sim_button, app.new_button, app.undo_button, app.save_button, app.delete_button]
    texts = [b['text'] for b in buttons]
    assert texts == [
        '🔍 Ähnliche Einträge',
        '🧹 Formular leeren',
        '↩️ Rückgängig',
        '💾 Speichern',
        '🗑 Entfernen',
    ]
    root.destroy()
