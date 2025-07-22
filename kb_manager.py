import os
import sys
import textwrap
import pandas as pd
import numpy as np
import json
from pydantic import BaseModel, validator
from openai import OpenAI
from tkinter import (
    Label,
    Entry,
    Text,
    Button,
    Listbox,
    Scrollbar,
    END,
    Menu,
    Toplevel,
    messagebox,
    Frame,
    StringVar,
)
from tkinter import ttk, simpledialog
import ttkbootstrap as tb

VERSION = os.environ.get("PR_NUMBER", "dev")

CSV_FILE = 'knowledgebase.csv'
TOOLTIPS_FILE = 'ui_tooltips.json'
COLUMNS = [
    'category',
    'faq_question',
    'answer_text',
    'stone_type',
    'product_form',
    'product_size',
    'eigenschaft',
    'anwendung',
]

PROMPT_TEMPLATE = """
System: Du bist Stone Knowledgebase Codex.
Du bist ein gewissenhafter FAQ-Generator für telefonischen Kundensupport.
Erstelle immer bis zu 5 FAQ-Einträge – auch wenn der Eingabetext keine Fragen enthält; formuliere dann selbst passende Fragen.
Gib ausschließlich das folgende JSON-Format zurück (kein zusätzlicher Text!):

{"entries":[{"category":"product|payment|delivery|installation|warranty","faq_question":"string","answer_text":"string","stone_type":"string","product_form":"string","product_size":"string (inkl. Einheit)","eigenschaft":"string","anwendung":"string"}]}

User: Erstelle FAQ-Einträge aus diesem Rohtext (Deutsch):
"""  # noqa: E501

FALLBACK_HINT = "Bitte erstelle passende FAQ-Einträge, selbst wenn der Text kaum Informationen enthält.\n"


def load_tooltips():
    try:
        with open(TOOLTIPS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


class Tooltip:
    """Simple hover tooltip."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 1
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        Label(
            tw,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
            justify="left",
            wraplength=300,
        ).pack(ipadx=1)

    def hide(self, _=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class FAQEntry(BaseModel):
    category: str
    faq_question: str
    answer_text: str
    stone_type: str = ''
    product_form: str = ''
    product_size: str = ''
    eigenschaft: str = ''
    anwendung: str = ''

    @validator('product_size')
    def ensure_unit(cls, v):
        if v and not any(unit in v for unit in ['mm', 'cm', 'm']):
            if v.strip().isdigit():
                return v + ' mm'
        return v


def load_kb():
    """Load the knowledge base from CSV or return an empty DataFrame."""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = df.reindex(columns=COLUMNS)
        df.fillna('', inplace=True)
        return df
    return pd.DataFrame(columns=COLUMNS)


def save_kb(df):
    """Persist the DataFrame back to the CSV file."""
    df.reindex(columns=COLUMNS).to_csv(CSV_FILE, index=False)


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    if not np.any(a) or not np.any(b):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(client, text):
    resp = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return resp.data[0].embedding


def find_similar(df, question, client, existing_embeddings=None, top_n=3):
    if df.empty or not question.strip() or not client:
        return []
    if existing_embeddings is None:
        existing_embeddings = [
            get_embedding(client, q) for q in df["faq_question"].fillna("")
        ]
    q_emb = get_embedding(client, question)
    sims = [cosine(q_emb, emb) for emb in existing_embeddings]
    idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for i in idx:
        if sims[i] > 0.7:
            row = df.iloc[i]
            results.append((sims[i], row))
    return results


def generate_suggestions(text, client):
    """Return FAQ suggestions from OpenAI."""
    prompt = PROMPT_TEMPLATE + text
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        exc.raw_content = content
        raise
    entries = data.get("entries") if isinstance(data, dict) else None
    if not entries:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": PROMPT_TEMPLATE + FALLBACK_HINT + text}],
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            exc.raw_content = content
            raise
        entries = data.get("entries") if isinstance(data, dict) else []
    result = []
    for e in entries or []:
        try:
            result.append(FAQEntry(**e).dict())
        except Exception:
            continue
    return result


class KBManager:
    def __init__(self, master):
        self.master = master
        master.title('Stone Knowledgebase Manager')
        master.option_add("*Font", "Segoe UI 12")
        self.style = tb.Style("flatly")
        self.dark = False
        self.tooltips = load_tooltips()
        self.df = load_kb().reset_index(drop=True)
        self.api_key = os.environ.get('OPENAI_API_KEY', '')
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.suggestions = []
        self.embeddings = []
        self.trash = []
        self.deactivated = set()
        self.marked = set()
        self.search_after_id = None
        if self.client and not self.df.empty:
            self.embeddings = [get_embedding(self.client, q) for q in self.df['faq_question']]

        # frames for layout
        toolbar = Frame(master)
        form = Frame(master)
        table = Frame(master)
        toolbar.grid(row=0, column=0, columnspan=2, sticky='ew')
        form.grid(row=1, column=0, sticky='nw', padx=5, pady=5)
        table.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)

        Button(toolbar, text='Neu', command=self.clear_form).pack(side='left')
        Button(toolbar, text='Speichern', command=self.save_entry).pack(side='left')
        Button(toolbar, text='Löschen', command=self.delete_entry).pack(side='left')
        Button(toolbar, text='Dark-Mode', command=self.toggle_theme).pack(side='right')

        master.columnconfigure(1, weight=1)
        master.rowconfigure(1, weight=1)
        table.columnconfigure(0, weight=1)
        table.rowconfigure(1, weight=1)

        labels = [
            'Kategorie',
            'Frage',
            'Antwort',
            'Steinart',
            'Produktform',
            'Produktgröße',
            'Eigenschaft',
            'Anwendung',
        ]
        self.entries = []
        self.categories = [
            'product',
            'payment',
            'delivery',
            'installation',
            'warranty',
        ]
        for i, lbl in enumerate(labels):
            Label(form, text=lbl).grid(row=i, column=0, sticky='e')
            if lbl == 'Antwort':
                txt = Text(form, width=40, height=4)
                txt.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(txt)
                Tooltip(txt, self.tooltips.get('answer_text', ''))
            elif lbl == 'Kategorie':
                cb = ttk.Combobox(form, values=self.categories, state='readonly', width=37)
                cb.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(cb)
                Tooltip(cb, self.tooltips.get('category', ''))
            else:
                ent = Entry(form, width=40)
                ent.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(ent)
                col_key = COLUMNS[i]
                Tooltip(ent, self.tooltips.get(col_key, ''))

        base = len(labels)
        Label(form, text='Text für KI-Vorschläge').grid(row=base, column=0, sticky='ne')
        self.source_text = Text(form, width=40, height=6)
        self.source_text.grid(row=base, column=1, padx=5, pady=2)
        Tooltip(self.source_text, self.tooltips.get('source', ''))
        Button(
            form,
            text='Vorschläge generieren',
            command=self.generate_suggestions,
        ).grid(row=base+1, column=0, columnspan=2, pady=5)
        Button(
            form,
            text='Ähnliche Fragen anzeigen',
            command=self.check_similar,
        ).grid(row=base+2, column=0, pady=5)
        Button(form, text='Speichern', command=self.save_entry).grid(
            row=base+2, column=1, pady=5
        )
        Button(form, text='Eintrag laden', command=self.load_entry).grid(
            row=base+3, column=0, pady=2
        )
        Button(form, text='Eintrag löschen', command=self.delete_entry).grid(
            row=base+3, column=1, pady=2
        )
        Button(form, text='Rückgängig Löschen', command=self.undo_delete).grid(
            row=base+4, column=0, columnspan=2, pady=2
        )
        Button(form, text='Neu', command=self.clear_form).grid(
            row=base+5, column=0, columnspan=2, pady=2
        )
        self.suggestion_box = Listbox(form, width=60)
        self.suggestion_box.grid(row=base+6, column=0, columnspan=2, padx=5, pady=5)
        self.suggestion_box.bind('<Double-1>', lambda e: self.load_suggestion())
        Button(
            form,
            text='Vorschlag löschen',
            command=self.delete_suggestion,
        ).grid(row=base+7, column=0, columnspan=2, pady=2)
        self.listbox = Listbox(form, width=60)
        self.listbox.grid(row=base+8, column=0, columnspan=2, padx=5, pady=5)
        Button(form, text='API-Key eingeben', command=self.set_api_key).grid(
            row=base+9, column=0, columnspan=2, pady=2
        )

        filter_frame = Frame(table)
        filter_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 4))
        Label(filter_frame, text='Filter:').pack(side='left')
        self.filter_text = StringVar()
        self.filter_text.trace_add('write', self.on_filter_change)
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_text)
        self.filter_entry.pack(side='left', fill='x', expand=True)
        Button(filter_frame, text='Reset', command=self.reset_filter).pack(side='left')
        self.progress = ttk.Progressbar(filter_frame, mode='indeterminate', length=80)

        self.tree = ttk.Treeview(table, columns=COLUMNS, show='headings')
        self.sort_state = {c: False for c in COLUMNS}
        for col in COLUMNS:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by_column(c))
            self.tree.column(col, width=120, anchor='w')
        self.tree_scroll = Scrollbar(table, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_scroll.set)
        self.tree.grid(row=1, column=0, sticky='nsew')
        self.tree_scroll.grid(row=1, column=1, sticky='ns')
        self.tree.bind('<<TreeviewSelect>>', self.update_detail)
        self.detail = Text(table, height=6, wrap='word', state='disabled')
        self.detail.grid(row=2, column=0, columnspan=2, sticky='nsew', pady=(4, 0))
        self.tree.bind('<Double-1>', lambda e: self.load_entry())
        self.tree.tag_configure('deactivated', foreground='gray')
        self.tree.tag_configure('marked', background='#ffd966')
        self.row_menu = Menu(self.tree, tearoff=0)
        self.row_menu.add_command(label='Löschen', command=self.delete_entry)
        self.row_menu.add_command(label='Deaktivieren', command=self.toggle_deactivate)
        self.row_menu.add_command(label='Markieren', command=self.mark_entry)
        self.empty_menu = Menu(self.tree, tearoff=0)
        self.empty_menu.add_command(label='Neuen Eintrag einfügen', command=self.clear_form)
        self.tree.bind('<Button-3>', self.show_context_menu)

        self.refresh_tree()
        self.edit_index = None

        Label(master, text=f"Version {VERSION}").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        master.bind_all('<Control-s>', lambda e: self.save_entry())

    def refresh_tree(self):
        """Fill the treeview with all current rows."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        max_lines = 1
        for _, row in self.df.iterrows():
            values = []
            for col in COLUMNS:
                text = str(row[col])
                wrapped = textwrap.fill(text, 30)
                values.append(wrapped)
                max_lines = max(max_lines, wrapped.count('\n') + 1)
            self.tree.insert('', 'end', values=values)
        ttk.Style().configure('Treeview', rowheight=20 * max_lines)

    def apply_filter(self):
        term = self.filter_text.get().strip()
        if not term:
            self.refresh_tree()
            return
        show_id = self.master.after(200, self.start_progress)
        for item in self.tree.get_children():
            self.tree.delete(item)
        mask = self.df['faq_question'].str.contains(term, case=False, na=False)
        for _, row in self.df[mask].iterrows():
            values = [textwrap.fill(str(row[c]), 30) for c in COLUMNS]
            self.tree.insert('', 'end', values=values)
        self.master.after_cancel(show_id)
        self.stop_progress()

    def reset_filter(self):
        self.filter_text.set('')
        self.refresh_tree()

    def sort_by_column(self, col):
        """Sort the table by the given column."""
        self.sort_state[col] = not self.sort_state[col]
        self.df.sort_values(col, ascending=not self.sort_state[col], inplace=True, ignore_index=True)
        self.refresh_tree()

    def refresh_suggestion_box(self):
        """Update the listbox with current suggestions."""
        self.suggestion_box.delete(0, END)
        for item in self.suggestions:
            question = item.get('faq_question', '')
            answer = item.get('answer_text', '')
            self.suggestion_box.insert(END, f'{question} -> {answer[:40]}...')

    def animate_scroll_to(self, index, steps=10, delay=20):
        """Scroll the treeview to the given row with a short animation."""
        items = self.tree.get_children()
        if not items or index >= len(items):
            return
        target = index / max(1, len(items))
        start = self.tree.yview()[0]
        delta = (target - start) / steps

        for step in range(steps):
            pos = start + delta * (step + 1)
            self.master.after(delay * step, lambda p=pos: self.tree.yview_moveto(p))

    def highlight_row(self, index):
        items = self.tree.get_children()
        if index < len(items):
            iid = items[index]
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.animate_scroll_to(index)

    def update_detail(self, _=None):
        sel = self.tree.selection()
        self.detail.config(state='normal')
        self.detail.delete('1.0', END)
        if sel:
            index = self.tree.index(sel[0])
            row = self.df.iloc[index]
            text = "\n".join(f"{c}: {row[c]}" for c in COLUMNS)
            self.detail.insert('1.0', textwrap.fill(text, 80))
        self.detail.config(state='disabled')

    def toggle_theme(self):
        self.dark = not self.dark
        theme = "darkly" if self.dark else "flatly"
        self.style.theme_use(theme)

    def show_context_menu(self, event):
        rowid = self.tree.identify_row(event.y)
        if rowid:
            self.tree.selection_set(rowid)
            self.row_menu.tk_popup(event.x_root, event.y_root)
        else:
            self.empty_menu.tk_popup(event.x_root, event.y_root)

    def toggle_deactivate(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx in self.deactivated:
            self.deactivated.remove(idx)
            self.tree.item(sel[0], tags=tuple(t for t in self.tree.item(sel[0], 'tags') if t != 'deactivated'))
        else:
            self.deactivated.add(idx)
            tags = set(self.tree.item(sel[0], 'tags'))
            tags.add('deactivated')
            self.tree.item(sel[0], tags=tuple(tags))

    def mark_entry(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx in self.marked:
            self.marked.remove(idx)
            self.tree.item(sel[0], tags=tuple(t for t in self.tree.item(sel[0], 'tags') if t != 'marked'))
        else:
            self.marked.add(idx)
            tags = set(self.tree.item(sel[0], 'tags'))
            tags.add('marked')
            self.tree.item(sel[0], tags=tuple(tags))

    def start_progress(self):
        self.progress.pack(side='left')
        self.progress.start(10)

    def stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()

    def on_filter_change(self, *_):
        if self.search_after_id:
            self.master.after_cancel(self.search_after_id)
        self.search_after_id = self.master.after(300, self.apply_filter)

    def clear_form(self):
        """Clear all entry widgets and reset edit mode."""
        for widget in self.entries:
            if isinstance(widget, Text):
                widget.delete('1.0', END)
            elif isinstance(widget, ttk.Combobox):
                widget.set('')
            else:
                widget.delete(0, END)
        self.listbox.delete(0, END)
        self.edit_index = None

    def load_entry(self):
        """Load the selected row from the table into the form for editing."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showerror('Fehler', 'Bitte zuerst einen Eintrag auswählen.')
            return
        index = self.tree.index(selection[0])
        row = self.df.iloc[index]
        for widget, col in zip(self.entries, COLUMNS):
            value = str(row[col]) if pd.notna(row[col]) else ''
            if isinstance(widget, Text):
                widget.delete('1.0', END)
                widget.insert('1.0', value)
            elif isinstance(widget, ttk.Combobox):
                widget.set(value)
            else:
                widget.delete(0, END)
                widget.insert(0, value)
        self.edit_index = index

    def delete_entry(self):
        """Delete the selected row from the table and CSV."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showerror('Fehler', 'Bitte zuerst einen Eintrag auswählen.')
            return
        index = self.tree.index(selection[0])
        if messagebox.askyesno('Löschen', 'Ausgewählten Eintrag wirklich löschen?'):
            self.trash.append(self.df.iloc[index].to_dict())
            self.df = self.df.drop(self.df.index[index]).reset_index(drop=True)
            save_kb(self.df)
            self.refresh_tree()
            self.edit_index = None

    def undo_delete(self):
        if not self.trash:
            messagebox.showinfo('Hinweis', 'Kein Eintrag zum Wiederherstellen.')
            return
        row = self.trash.pop()
        self.df.loc[len(self.df)] = row
        save_kb(self.df)
        self.refresh_tree()

    def get_entry_values(self):
        values = []
        for widget in self.entries:
            if isinstance(widget, Text):
                values.append(widget.get('1.0', END).strip())
            elif isinstance(widget, ttk.Combobox):
                values.append(widget.get().strip())
            else:
                values.append(widget.get().strip())
        data = dict(zip(COLUMNS, values))
        try:
            entry = FAQEntry(**data)
        except Exception as exc:
            messagebox.showerror('Fehler', f'Ungueltige Eingaben: {exc}')
            return None
        return entry.dict()

    def set_api_key(self):
        """Prompt the user for an OpenAI API key."""
        key = simpledialog.askstring('API-Key', 'OpenAI API-Key eingeben:', show='*')
        if key:
            self.api_key = key
            self.client = OpenAI(api_key=key)

    def suggest_improvement(self, text):
        """Ask OpenAI for improvement hints for the given text."""
        if not self.client:
            return ''
        prompt = (
            'Der folgende Text reicht nicht aus, um vollst\u00e4ndige FAQ-Vorschl\u00e4ge zu erzeugen. '
            'Gib in einem Satz an, welche zus\u00e4tzlichen Informationen sinnvoll w\u00e4ren:'
        )
        try:
            resp = self.client.chat.completions.create(
                model='gpt-4o',
                messages=[{'role': 'user', 'content': prompt + '\n' + text}],
                max_tokens=60,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ''

    def generate_suggestions(self):
        """Generate suggestions via OpenAI and update the list."""
        if not self.client:
            self.set_api_key()
            if not self.client:
                return
        text = self.source_text.get('1.0', END).strip()
        try:
            new_entries = generate_suggestions(text, self.client)
        except json.JSONDecodeError as exc:
            messagebox.showerror('JSON-Fehler', getattr(exc, 'raw_content', ''))
            return
        except Exception as exc:
            print(f'Error: {exc}', file=sys.stderr)
            raise
        self.suggestions.extend(new_entries)
        if not new_entries:
            messagebox.showinfo('Hinweis', 'Keine geeigneten Vorschläge gefunden.')
            return
        self.refresh_suggestion_box()

    def load_suggestion(self):
        """Load the selected suggestion into the form."""
        sel = self.suggestion_box.curselection()
        if not sel:
            return
        index = sel[0]
        suggestion = self.suggestions.pop(index)
        try:
            data = FAQEntry(**suggestion).dict()
        except Exception:
            data = suggestion
        for widget, col in zip(self.entries, COLUMNS):
            val = data.get(col, '')
            if isinstance(widget, Text):
                widget.delete('1.0', END)
                widget.insert('1.0', val)
            elif isinstance(widget, ttk.Combobox):
                widget.set(val)
            else:
                widget.delete(0, END)
                widget.insert(0, val)
        self.refresh_suggestion_box()

    def delete_suggestion(self):
        """Remove the selected suggestion from the list."""
        sel = self.suggestion_box.curselection()
        if not sel:
            return
        self.suggestions.pop(sel[0])
        self.refresh_suggestion_box()

    def check_similar(self):
        data = self.get_entry_values()
        if data is None:
            return
        question = data['faq_question']
        sims = find_similar(self.df, question, self.client, self.embeddings)
        self.listbox.delete(0, END)
        for score, row in sims:
            display = f"({score:.2f}) {row['faq_question']} -> {row['answer_text']}"
            self.listbox.insert(END, display)
        if sims:
            self.highlight_row(sims[0][1].name)
        else:
            self.listbox.insert(END, 'Keine ähnlichen Fragen gefunden.')

    def save_entry(self):
        data = self.get_entry_values()
        if data is None:
            return
        if not data['category'] or not data['faq_question'] or not data['answer_text']:
            messagebox.showerror('Fehler', 'Kategorie, Frage und Antwort sind Pflichtfelder.')
            return
        new_row = data
        if self.edit_index is None:
            row_index = len(self.df)
            self.df.loc[row_index] = new_row
        else:
            row_index = self.edit_index
            for col, val in new_row.items():
                self.df.at[row_index, col] = val
            self.edit_index = None
        save_kb(self.df)
        messagebox.showinfo('Gespeichert', 'Eintrag wurde gespeichert.')
        self.refresh_tree()
        self.highlight_row(row_index)
        self.clear_form()


if __name__ == '__main__':
    root = tb.Window(themename='flatly')
    app = KBManager(root)
    root.mainloop()
