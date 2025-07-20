import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from openai import OpenAI
from tkinter import (
    Tk,
    Label,
    Entry,
    Text,
    Button,
    Listbox,
    Scrollbar,
    END,
    messagebox,
    Frame,
)
from tkinter import ttk, simpledialog

CSV_FILE = 'knowledgebase.csv'
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

def load_kb():
    """Load the knowledge base from CSV or return an empty DataFrame."""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        return df.reindex(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)

def save_kb(df):
    """Persist the DataFrame back to the CSV file."""
    df.to_csv(CSV_FILE, index=False)

def find_similar(df, question, top_n=3):
    if df.empty or not question.strip():
        return []
    questions = df['faq_question'].fillna('').tolist() + [question]
    vec = TfidfVectorizer().fit_transform(questions)
    sims = cosine_similarity(vec[-1], vec[:-1]).flatten()
    idx = sims.argsort()[::-1][:top_n]
    results = []
    for i in idx:
        if sims[i] > 0.3:  # threshold
            row = df.iloc[i]
            results.append((sims[i], row))
    return results

class KBManager:
    def __init__(self, master):
        self.master = master
        master.title('Stone Knowledgebase Manager')
        self.df = load_kb().reset_index(drop=True)
        self.api_key = os.environ.get('OPENAI_API_KEY', '')
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.suggestions = []

        # frames for layout
        form = Frame(master)
        table = Frame(master)
        form.grid(row=0, column=0, sticky='nw', padx=5, pady=5)
        table.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        master.columnconfigure(1, weight=1)
        master.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)

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
            elif lbl == 'Kategorie':
                cb = ttk.Combobox(form, values=self.categories, state='readonly', width=37)
                cb.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(cb)
            else:
                ent = Entry(form, width=40)
                ent.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(ent)

        base = len(labels)
        Label(form, text='Text für KI-Vorschläge').grid(row=base, column=0, sticky='ne')
        self.source_text = Text(form, width=40, height=6)
        self.source_text.grid(row=base, column=1, padx=5, pady=2)
        Button(form, text='Vorschläge generieren', command=self.generate_suggestions).grid(row=base+1, column=0, columnspan=2, pady=5)
        Button(form, text='Ähnliche Fragen anzeigen', command=self.check_similar).grid(row=base+2, column=0, pady=5)
        Button(form, text='Speichern', command=self.save_entry).grid(row=base+2, column=1, pady=5)
        Button(form, text='Eintrag laden', command=self.load_entry).grid(row=base+3, column=0, pady=2)
        Button(form, text='Eintrag löschen', command=self.delete_entry).grid(row=base+3, column=1, pady=2)
        Button(form, text='Neu', command=self.clear_form).grid(row=base+4, column=0, columnspan=2, pady=2)
        self.suggestion_box = Listbox(form, width=60)
        self.suggestion_box.grid(row=base+5, column=0, columnspan=2, padx=5, pady=5)
        self.suggestion_box.bind('<Double-1>', lambda e: self.load_suggestion())
        Button(form, text='Vorschlag löschen', command=self.delete_suggestion).grid(row=base+6, column=0, columnspan=2, pady=2)
        self.listbox = Listbox(form, width=60)
        self.listbox.grid(row=base+7, column=0, columnspan=2, padx=5, pady=5)
        Button(form, text='API-Key eingeben', command=self.set_api_key).grid(row=base+8, column=0, columnspan=2, pady=2)

        self.tree = ttk.Treeview(table, columns=COLUMNS, show='headings')
        for col in COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='w')
        self.tree_scroll = Scrollbar(table, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        self.tree_scroll.grid(row=0, column=1, sticky='ns')

        self.refresh_tree()
        self.edit_index = None

    def refresh_tree(self):
        """Fill the treeview with all current rows."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        for _, row in self.df.iterrows():
            self.tree.insert('', 'end', values=row.tolist())

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
            self.df = self.df.drop(self.df.index[index]).reset_index(drop=True)
            save_kb(self.df)
            self.refresh_tree()
            self.edit_index = None

    def get_entry_values(self):
        values = []
        for widget in self.entries:
            if isinstance(widget, Text):
                values.append(widget.get('1.0', END).strip())
            elif isinstance(widget, ttk.Combobox):
                values.append(widget.get().strip())
            else:
                values.append(widget.get().strip())
        return values

    def set_api_key(self):
        """Prompt the user for an OpenAI API key."""
        key = simpledialog.askstring('API-Key', 'OpenAI API-Key eingeben:', show='*')
        if key:
            self.api_key = key
            self.client = OpenAI(api_key=key)

    def generate_suggestions(self):
        """Use OpenAI to propose new FAQ entries from the source text."""
        if not self.client:
            self.set_api_key()
            if not self.client:
                return
        text = self.source_text.get('1.0', END).strip()
        prompt = (
            'Erstelle aus dem folgenden Text bis zu fünf FAQ-Einträge. '
            'Gib immer ein JSON-Objekt {"entries": [...]} zurück. '
            'Jeder Eintrag besitzt die Felder category, faq_question, '
            'answer_text, stone_type, product_form, product_size, '
            'eigenschaft und anwendung. Nutze deine Kenntnisse, um auch '
            'bei wenigen Informationen sinnvolle Vorschläge zu machen. '
            'Wenn Angaben fehlen, lasse das Feld leer und gib bei '
            'product_size immer eine Einheit wie "mm" oder "cm" an.'
        )
        try:
            resp = self.client.chat.completions.create(
                model='gpt-4o',
                messages=[{'role': 'user', 'content': prompt + '\n' + text}],
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
        except json.JSONDecodeError:
            messagebox.showinfo('Hinweis', 'Die KI lieferte kein verwertbares Ergebnis.')
            return
        except Exception as exc:
            messagebox.showinfo('Hinweis', f'Keine Vorschläge erzeugt: {exc}')
            return
        entries = data.get('entries') if isinstance(data, dict) else data
        if not isinstance(entries, list):
            messagebox.showinfo('Hinweis', 'Keine geeigneten Vorschläge gefunden.')
            return
        self.suggestions.extend(entries)
        if not self.suggestions:
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
        for widget, col in zip(self.entries, COLUMNS):
            val = suggestion.get(col, '')
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
        values = self.get_entry_values()
        question = values[1]
        sims = find_similar(self.df, question)
        self.listbox.delete(0, END)
        for score, row in sims:
            display = f"({score:.2f}) {row['faq_question']} -> {row['answer_text']}"
            self.listbox.insert(END, display)
        if sims:
            self.highlight_row(sims[0][1].name)
        else:
            self.listbox.insert(END, 'Keine ähnlichen Fragen gefunden.')

    def save_entry(self):
        values = self.get_entry_values()
        if not values[0] or not values[1] or not values[2]:
            messagebox.showerror('Fehler', 'Kategorie, Frage und Antwort sind Pflichtfelder.')
            return
        new_row = dict(zip(COLUMNS, values))
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
    root = Tk()
    app = KBManager(root)
    root.mainloop()
