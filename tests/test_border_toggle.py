import os
import sys
import tkinter as tk
import pandas as pd
import pytest
from ttkbootstrap import Window

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from kb_manager import KBManager, COLUMNS, save_kb  # noqa: E402


def display_available():
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


@pytest.mark.skipif(not display_available(), reason="requires display")
def test_border_toggle(monkeypatch, tmp_path):
    monkeypatch.setattr("language_tool_python.LanguageTool", lambda *_: None)
    csv_path = tmp_path / "kb.csv"
    df = pd.DataFrame(columns=COLUMNS)
    save_kb(df, csv_path)
    root = Window(themename="flatly")
    app = KBManager(root)
    app.csv_file = str(csv_path)
    app.df = df
    app.refresh_tree()

    sug = {
        "category": "Produkt",
        "faq_question": "Q",
        "answer_text": "A",
        "stone_type": "",
        "product_form": "",
        "product_size": "",
        "eigenschaft": "",
        "anwendung": "",
    }
    app.suggestions = [sug]
    app.refresh_suggestion_box()
    app.suggestion_box.selection_set(0)
    app.load_suggestion()
    assert app.form_block.cget("highlightbackground") == "#FFA64D"
    app.save_entry()
    assert app.form_block.cget("highlightbackground") == "#f2f2f2"
    root.destroy()
