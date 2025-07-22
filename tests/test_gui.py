import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from kb_manager import KBManager, load_kb, save_kb  # noqa: E402
from ttkbootstrap import Window  # noqa: E402
import tkinter as tk  # noqa: E402
import pytest  # noqa: E402


def display_available():
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


def setup_module(module):
    df = load_kb()
    if df.empty:
        df.loc[0] = {
            'category': 'product',
            'faq_question': 'Welche Steine sind hart?',
            'answer_text': 'Alle Granitsteine sind hart.',
            'stone_type': 'Granit',
            'product_form': 'Block',
            'product_size': '10 cm',
            'eigenschaft': 'hart',
            'anwendung': 'Garten',
        }
        save_kb(df)


@pytest.mark.skipif(not display_available(), reason="requires display")
def test_live_search():
    root = Window(themename="flatly")
    app = KBManager(root)
    app.filter_text.set('Steine')
    app.apply_filter()
    assert len(app.tree.get_children()) >= 1
    root.destroy()


@pytest.mark.skipif(not display_available(), reason="requires display")
def test_context_menu_entries():
    root = Window(themename="flatly")
    app = KBManager(root)
    labels = [app.row_menu.entrycget(i, 'label') for i in range(app.row_menu.index('end') + 1)]
    assert {'Löschen', 'Markieren', 'Kopieren'} <= set(labels)
    root.destroy()
