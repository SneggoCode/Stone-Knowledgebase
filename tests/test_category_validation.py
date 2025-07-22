import os
import sys
import tkinter as tk
import pytest
from ttkbootstrap import Window

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from kb_manager import KBManager  # noqa: E402


def display_available():
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


@pytest.mark.skipif(not display_available(), reason="requires display")
def test_category_validation(monkeypatch):
    monkeypatch.setattr('language_tool_python.LanguageTool', lambda *_: None)
    root = Window(themename="flatly")
    app = KBManager(root)
    app.entries[0].set('Falsch')
    app.entries[1].insert('1.0', 'Frage?')
    app.entries[2].insert('1.0', 'Antwort')
    app.save_entry()
    assert 'Ungültige Kategorie' in app.message_var.get()
    root.destroy()
