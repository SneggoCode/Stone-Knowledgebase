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
def test_csv_auto_reload(tmp_path):
    csv_path = tmp_path / "kb.csv"
    df = pd.DataFrame(columns=COLUMNS)
    save_kb(df, csv_path)
    root = Window(themename="flatly")
    app = KBManager(root)
    app.csv_file = str(csv_path)
    app.df = df
    app.refresh_tree()
    app.csv_mtime = os.path.getmtime(csv_path)
    df.loc[0] = {c: "x" for c in COLUMNS}
    save_kb(df, csv_path)
    app.check_csv_update()
    assert len(app.df) == 1
    assert app.message_var.get() == "CSV erneut geladen."
    root.destroy()
