import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
import tkinter as tk  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from ttkbootstrap import Window  # noqa: E402
from tkinter import ttk  # noqa: E402
from kb_manager import KBManager, COLUMNS, save_kb  # noqa: E402


def display_available():
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


@pytest.mark.skipif(not display_available(), reason="requires display")
def test_form_undo(tmp_path, monkeypatch):
    monkeypatch.setattr('language_tool_python.LanguageTool', lambda *_: None)
    csv_path = tmp_path / 'kb.csv'
    df = pd.DataFrame(columns=COLUMNS)
    save_kb(df, csv_path)
    root = Window(themename='flatly')
    app = KBManager(root)
    app.csv_file = str(csv_path)
    app.df = df
    app.refresh_tree()

    def fill(text):
        for w in app.entries:
            if isinstance(w, tk.Text):
                w.delete('1.0', tk.END)
                w.insert('1.0', text)
            elif isinstance(w, ttk.Combobox):
                w.set(app.categories[0])
            else:
                w.delete(0, tk.END)
                w.insert(0, text)

    for i in range(5):
        fill(str(i))
        app.clear_form()

    for i in reversed(range(5)):
        app.undo_action()
        assert app.entries[1].get('1.0', tk.END).strip() == str(i)
        app.clear_form()
    root.destroy()
