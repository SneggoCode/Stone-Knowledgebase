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
def test_edit_border_escape(tmp_path):
    csv_path = tmp_path / "kb.csv"
    df = pd.DataFrame([{c: "x" for c in COLUMNS}])
    save_kb(df, csv_path)
    root = Window(themename="flatly")
    app = KBManager(root)
    app.csv_file = str(csv_path)
    app.df = df
    app.refresh_tree()
    app.tree.selection_set(app.tree.get_children()[0])
    app.load_entry()
    assert int(app.form.cget("highlightthickness")) == 2
    app.on_escape()
    assert int(app.form.cget("highlightthickness")) == 0
    root.destroy()
