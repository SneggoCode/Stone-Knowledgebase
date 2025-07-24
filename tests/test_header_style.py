import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import font  # noqa: E402
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
def test_header_font(monkeypatch):
    monkeypatch.setattr('language_tool_python.LanguageTool', lambda *_: None)
    root = Window(themename="flatly")
    app = KBManager(root)
    style = app.style
    name = style.lookup("Treeview.Heading", "font")
    f = font.nametofont(name)
    assert f.actual("weight") == "bold"
    root.destroy()
