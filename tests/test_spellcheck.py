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
def test_spellcheck_highlight(monkeypatch):
    class DummyLT:
        def check(self, text):
            return [type('obj', (), {'offset': 5, 'errorLength': 4})]

    monkeypatch.setattr('language_tool_python.LanguageTool', lambda *_: DummyLT())
    root = Window(themename="flatly")
    app = KBManager(root)
    widget = app.entries[1]  # faq_question Text widget
    widget.insert('1.0', 'Hallo testt')
    app.check_spelling(widget)
    root.after(400, root.quit)
    root.mainloop()
    assert widget.tag_ranges('misspell')
    root.destroy()
