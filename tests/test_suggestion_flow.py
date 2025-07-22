import os
import sys
import pandas as pd
import tkinter as tk
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
def test_suggestion_save_flow(tmp_path):
    csv_path = tmp_path / "kb.csv"
    df = pd.DataFrame(columns=COLUMNS)
    save_kb(df, csv_path)
    root = Window(themename="flatly")
    app = KBManager(root)
    app.csv_file = str(csv_path)
    app.df = df
    app.refresh_tree()

    suggestion = {
        "category": "Produkt",
        "faq_question": "Q1",
        "answer_text": "A1",
        "stone_type": "",
        "product_form": "",
        "product_size": "",
        "eigenschaft": "",
        "anwendung": "",
    }
    app.suggestions = [suggestion]
    app.refresh_suggestion_box()
    app.suggestion_box.selection_set(0)
    app.load_suggestion()
    app.save_entry()
    assert not app.suggestions
    root.destroy()
