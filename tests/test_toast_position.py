import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
import tkinter as tk  # noqa: E402
import pytest  # noqa: E402
from ttkbootstrap import Window  # noqa: E402

from kb_manager import KBManager, ToastNotification  # noqa: E402


def display_available():
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


@pytest.mark.skipif(not display_available(), reason="requires display")
def test_toast_position(monkeypatch):
    monkeypatch.setattr('language_tool_python.LanguageTool', lambda *_: None)
    captured = {}

    class DummyToast(ToastNotification):
        def __init__(self, *args, **kwargs):
            captured['pos'] = kwargs.get('position')

        def show_toast(self, *_):
            pass

    monkeypatch.setattr('kb_manager.ToastNotification', DummyToast)
    root = Window(themename="flatly")
    root.geometry("300x200+100+100")
    app = KBManager(root)
    app.show_message("ok", success=True)
    root.after(100, root.quit)
    root.mainloop()
    x, y, _ = captured['pos']
    rx, ry = root.winfo_rootx(), root.winfo_rooty()
    rw, rh = root.winfo_width(), root.winfo_height()
    assert rx <= x <= rx + rw
    assert ry <= y <= ry + rh
    root.destroy()
