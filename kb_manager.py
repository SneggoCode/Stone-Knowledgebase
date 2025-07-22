import os
import sys
import textwrap
import pandas as pd
import numpy as np
import json
from pydantic import BaseModel, field_validator
from openai import OpenAI
from language_tool_python import LanguageTool
from tkinter import (
    Label,
    Entry,
    Text,
    Listbox,
    Scrollbar,
    END,
    Menu,
    Toplevel,
    Frame,
    StringVar,
    filedialog,
)
from tkinter import ttk, simpledialog
import subprocess
import ttkbootstrap as tb
from ttkbootstrap.toast import ToastNotification

try:
    VERSION = os.environ["PR_NUMBER"]
except KeyError:
    try:
        VERSION = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8"
        ).strip()
    except Exception:
        VERSION = "dev"

CSV_FILE = "knowledgebase.csv"
CONFIG_FILE = "config.json"
TOOLTIPS_FILE = "ui_tooltips.json"
COLUMNS = [
    "category",
    "faq_question",
    "answer_text",
    "stone_type",
    "product_form",
    "product_size",
    "eigenschaft",
    "anwendung",
]

CATEGORIES = [
    "Produkt",
    "Lieferung",
    "Konstruktion",
    "Sonstiges",
]


def normalize_category(value):
    """Return valid category or 'Sonstiges'."""
    val = str(value).strip().title()
    return val if val in CATEGORIES else "Sonstiges"


PROMPT_TEMPLATE = """
System:
You are “Stone Knowledgebase Codex”, ein akkurater FAQ-Generator für
telefonischen Kundensupport eines Naturstein-Händlers. Erstelle bis zu
5 FAQ-Einträge aus beliebigem deutschsprachigem Rohtext. Gib **nur**
ein JSON-Objekt zurück:

{
 "entries":[{ "category":"Produkt|Lieferung|Konstruktion|Sonstiges",
              "faq_question":"string",
              "answer_text":"string",
              "stone_type":"string",
              "product_form":"string",
              "product_size":"string (inkl. Einheit)",
              "eigenschaft":"string",
              "anwendung":"string"}]
}

Kategorie-Leitfaden
• **Produkt**  – Steinarten, Formate, Eigenschaften, Farben, Verfügbarkeit
• **Lieferung** – Versand, Spedition, Abstellgenehmigung, Avisierung
• **Konstruktion** – Verlegen, Versetzen, Fundament, Pflegeanleitung
• **Sonstiges**  – alles andere (z.\u202fB. Garantie, Zahlung, Support)

Priorisiere das Hauptthema, wenn mehrere vorkommen.
**Nur die oben genannten Werte sind gültig.** Alle anderen musst du
intern auf „Sonstiges“ setzen, damit das JSON gültig bleibt.

Beispiele:
{"category":"Produkt","faq_question":"Welche Körnung hat Splitt 8/16?","answer_text":"…"}
{"category":"Lieferung","faq_question":"Was passiert, wenn ich nicht zuhause bin?","answer_text":"…"}
{"category":"Konstruktion","faq_question":"Wie verfuge ich Kalksteinplatten?","answer_text":"…"}
{"category":"Sonstiges","faq_question":"Wie erreiche ich den Kundenservice?","answer_text":"…"}

USER:
Erstelle FAQ-Einträge aus diesem Rohtext (Deutsch):
"""
FALLBACK_HINT = "Bitte erstelle passende FAQ-Einträge, selbst wenn der Text kaum Informationen enthält.\n"


def get_csv_path():
    """Return CSV path from config or ask the user."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                path = data.get("csv_path")
                if path:
                    return path
        except Exception:
            pass
    path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
    if not path:
        path = CSV_FILE
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"csv_path": path}, f)
    except Exception:
        pass
    return path


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
    stone_type: str = ""
    product_form: str = ""
    product_size: str = ""
    eigenschaft: str = ""
    anwendung: str = ""

    @field_validator("category", mode="before")
    @classmethod
    def check_category(cls, v):
        return normalize_category(v)

    @field_validator("product_size", mode="before")
    @classmethod
    def ensure_unit(cls, v):
        if v and not any(unit in str(v) for unit in ["mm", "cm", "m"]):
            if str(v).strip().isdigit():
                return f"{v} mm"
        return v


def load_kb(path=CSV_FILE):
    """Load the knowledge base from CSV or return an empty DataFrame."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.reindex(columns=COLUMNS)
        df.fillna("", inplace=True)
        return df
    return pd.DataFrame(columns=COLUMNS)


def save_kb(df, path=CSV_FILE):
    """Persist the DataFrame back to the CSV file."""
    df.reindex(columns=COLUMNS).to_csv(path, index=False)


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


LAST_RAW_CONTENT = ""
CATEGORY_MAPPED = False


def generate_suggestions(text, client):
    """Return FAQ suggestions from OpenAI."""
    global LAST_RAW_CONTENT, CATEGORY_MAPPED
    CATEGORY_MAPPED = False
    prompts = [PROMPT_TEMPLATE + text, PROMPT_TEMPLATE + FALLBACK_HINT + text]
    LAST_RAW_CONTENT = ""
    for prompt in prompts:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=500,
            )
        except Exception as exc:
            print(f"Error contacting OpenAI: {exc}", file=sys.stderr)
            continue
        content = resp.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(content, file=sys.stderr)
            LAST_RAW_CONTENT = content
            continue
        entries = data.get("entries") if isinstance(data, dict) else []
        results = []
        for e in entries:
            orig = e.get("category", "")
            mapped = normalize_category(orig)
            if mapped != str(orig).strip().title():
                CATEGORY_MAPPED = True
            e["category"] = mapped
            try:
                results.append(FAQEntry(**e).model_dump())
            except Exception as exc:
                print(f"Invalid entry skipped: {e} ({exc})", file=sys.stderr)
        if results:
            LAST_RAW_CONTENT = ""
            return results
    return []


class KBManager:
    def __init__(self, master):
        self.master = master
        master.title("Stone Knowledgebase Manager")
        master.option_add("*Font", "Aptos 12")
        self.style = tb.Style("flatly")
        self.tooltips = load_tooltips()
        self.csv_file = get_csv_path()
        self.df = load_kb(self.csv_file).reset_index(drop=True)
        if not os.path.exists(self.csv_file):
            save_kb(self.df, self.csv_file)
        self.lt = LanguageTool("de-DE")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.suggestions = []
        self.current_suggestion_index = None
        self.embeddings = []
        self.trash = []
        self.search_after_id = None
        if self.client and not self.df.empty:
            self.embeddings = [
                get_embedding(self.client, q) for q in self.df["faq_question"]
            ]

        # frames for layout
        toolbar = Frame(master)
        self.form = Frame(master, highlightthickness=0)
        table = Frame(master)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.form.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        table.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=2)
        master.rowconfigure(1, weight=1)
        table.columnconfigure(0, weight=1)
        table.rowconfigure(1, weight=1)
        self.form.columnconfigure(1, weight=1)

        self.entries = []
        self.categories = CATEGORIES

        fields = [
            (
                "Kategorie",
                ttk.Combobox,
                "category",
                {"values": self.categories, "state": "readonly", "width": 37},
            ),
            ("Frage", Text, "faq_question", {"height": 2}),
            ("Antwort", Text, "answer_text", {"height": 4}),
            ("Steinart", Entry, "stone_type", {}),
            ("Produktform", Entry, "product_form", {}),
            ("Produktgröße", Entry, "product_size", {}),
            ("Eigenschaft", Entry, "eigenschaft", {}),
            ("Anwendung", Entry, "anwendung", {}),
        ]

        for i, (lbl, widget_cls, key, opts) in enumerate(fields):
            Label(self.form, text=lbl).grid(row=i, column=0, sticky="e", pady=2)
            if widget_cls is Text:
                widget = widget_cls(self.form, width=60, **opts)
                if key in {"faq_question", "answer_text"}:
                    widget.bind(
                        "<KeyRelease>", lambda e, w=widget: self.check_spelling(w)
                    )
            elif widget_cls is ttk.Combobox:
                widget = widget_cls(self.form, **opts)
            else:
                widget = widget_cls(self.form, width=60)
            widget.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
            Tooltip(widget, self.tooltips.get(key, ""))
            self.entries.append(widget)

        self.form.columnconfigure(1, weight=1)

        base = len(fields)
        Label(self.form, text="Text für KI-Vorschläge").grid(row=base, column=0, sticky="ne")
        self.source_text = Text(self.form, width=60, height=6)
        self.source_text.grid(row=base, column=1, padx=5, pady=2, sticky="ew")
        Tooltip(self.source_text, self.tooltips.get("source", ""))
        btn_row = base + 1
        self.gen_button = tb.Button(
            self.form,
            text="🧠 Vorschläge generieren",
            command=self.generate_suggestions,
            bootstyle="primary",
        )
        self.gen_button.grid(row=btn_row, column=0, pady=(5, 0), sticky="w")
        Tooltip(self.gen_button, "KI analysiert den Text und erstellt Vorschläge")
        self.ai_message_var = StringVar()
        self.ai_message_label = Label(
            self.form, textvariable=self.ai_message_var, foreground="red"
        )
        self.ai_message_label.grid(row=btn_row, column=1, sticky="w")
        btn_row += 1
        self.sim_button = tb.Button(
            self.form,
            text="🔍 Ähnliche Fragen",
            command=self.check_similar,
        )
        self.sim_button.grid(row=btn_row, column=0, sticky="e", pady=5, padx=2)
        Tooltip(self.sim_button, "Suche in der Tabelle nach ähnlichen Fragen")
        self.save_button = tb.Button(
            self.form,
            text="💾 Speichern",
            command=self.save_entry,
            bootstyle="success",
        )
        self.save_button.grid(row=btn_row, column=1, sticky="w", pady=5, padx=2)
        Tooltip(self.save_button, "Eintrag speichern")
        btn_row += 1
        manage = Frame(self.form)
        manage.grid(row=btn_row, column=0, columnspan=2, pady=2)
        self.load_button = tb.Button(
            manage, text="📂 Laden", command=self.load_entry
        )
        self.load_button.pack(side="left", padx=2)
        Tooltip(self.load_button, "Markierten Eintrag bearbeiten")
        self.delete_button = tb.Button(
            manage,
            text="🗑 Löschen",
            command=self.delete_entry,
            bootstyle="danger",
        )
        self.delete_button.pack(side="left", padx=2)
        Tooltip(self.delete_button, "Markierten Eintrag löschen")
        self.undo_button = tb.Button(
            manage, text="↩️ Rückgängig", command=self.undo_delete
        )
        self.undo_button.pack(side="left", padx=2)
        Tooltip(self.undo_button, "Letzten Löschvorgang zurücknehmen")
        self.new_button = tb.Button(manage, text="➕ Neu", command=self.clear_form)
        self.new_button.pack(side="left", padx=2)
        Tooltip(self.new_button, "Formular leeren")
        btn_row += 1
        self.suggestion_box = Listbox(self.form)
        self.suggestion_box.grid(
            row=btn_row,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="nsew",
        )
        self.form.rowconfigure(btn_row, weight=1)
        self.suggestion_box.bind("<Double-1>", lambda e: self.load_suggestion())
        self.suggestion_box.bind("<ButtonRelease-1>", lambda e: self.load_suggestion())
        btn_row += 1
        del_btn = tb.Button(
            self.form,
            text="🗑 Vorschlag löschen",
            command=self.delete_suggestion,
            bootstyle="danger",
        )
        del_btn.grid(row=btn_row, column=0, columnspan=2, pady=2)
        Tooltip(del_btn, "Entfernt den gewählten Vorschlag")
        btn_row += 1
        key_btn = tb.Button(
            self.form,
            text="🔑 API-Key eingeben",
            command=self.set_api_key,
            bootstyle="secondary",
        )
        key_btn.grid(row=btn_row, column=0, columnspan=2, pady=2)
        Tooltip(key_btn, "OpenAI API-Key festlegen")

        filter_frame = Frame(table)
        filter_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        Label(filter_frame, text="Filter:").pack(side="left")
        self.filter_text = StringVar()
        self.filter_text.trace_add("write", self.on_filter_change)
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_text)
        self.filter_entry.pack(side="left", fill="x", expand=True)
        reset_btn = tb.Button(filter_frame, text="Reset", command=self.reset_filter)
        reset_btn.pack(side="left")
        Tooltip(reset_btn, "Filter zurücksetzen")
        self.progress = ttk.Progressbar(filter_frame, mode="indeterminate", length=80)

        self.tree = ttk.Treeview(table, columns=COLUMNS, show="headings")
        self.sort_state = {c: False for c in COLUMNS}
        for col in COLUMNS:
            self.tree.heading(
                col, text=col, command=lambda c=col: self.sort_by_column(c)
            )
            self.tree.column(col, width=120, anchor="w")
        self.tree_scroll = Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_scroll.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree_scroll.grid(row=1, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", lambda e: None)
        self.tree.bind("<Double-1>", lambda e: self.load_entry())
        self.tree.tag_configure("match", background="#ffff99")
        self.row_menu = Menu(self.tree, tearoff=0)
        self.row_menu.add_command(label="Kopieren", command=self.copy_row)
        self.row_menu.add_command(label="Löschen", command=self.delete_entry)
        self.empty_menu = Menu(self.tree, tearoff=0)
        self.empty_menu.add_command(
            label="Neuen Eintrag einfügen", command=self.clear_form
        )
        self.tree.bind("<Button-3>", self.show_context_menu)

        self.master.bind("<Escape>", lambda e: self.on_escape())

        self.refresh_tree()
        self.edit_index = None

        self.message_var = StringVar()
        self.message_label = Label(master, textvariable=self.message_var)
        self.message_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        Label(master, text=f"Version {VERSION}").grid(
            row=2, column=1, sticky="e", padx=5, pady=2
        )
        master.bind_all("<Control-s>", lambda e: self.save_entry())

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
                max_lines = max(max_lines, wrapped.count("\n") + 1)
            self.tree.insert("", "end", values=values)
        ttk.Style().configure("Treeview", rowheight=20 * max_lines)

    def apply_filter(self):
        term = self.filter_text.get().strip()
        if not term:
            self.refresh_tree()
            return
        show_id = self.master.after(200, self.start_progress)
        for item in self.tree.get_children():
            self.tree.delete(item)
        mask = self.df.apply(
            lambda r: term.lower() in " ".join(str(v).lower() for v in r.values),
            axis=1,
        )
        for _, row in self.df[mask].iterrows():
            values = [textwrap.fill(str(row[c]), 30) for c in COLUMNS]
            self.tree.insert("", "end", values=values, tags=("match",))
        self.master.after_cancel(show_id)
        self.stop_progress()

    def reset_filter(self):
        self.filter_text.set("")
        self.refresh_tree()

    def sort_by_column(self, col):
        """Sort the table by the given column."""
        self.sort_state[col] = not self.sort_state[col]
        self.df.sort_values(
            col, ascending=not self.sort_state[col], inplace=True, ignore_index=True
        )
        self.refresh_tree()

    def refresh_suggestion_box(self):
        """Update the suggestion list with current items."""
        self.suggestion_box.delete(0, END)
        for i, item in enumerate(self.suggestions, start=1):
            question = item.get("faq_question", "")
            self.suggestion_box.insert(END, f"{i}. {question}")
            bg = "#f7f7f7" if i % 2 else "white"
            self.suggestion_box.itemconfig(i - 1, bg=bg)

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

    def copy_row(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        row = self.df.iloc[idx]
        text = f"{row['faq_question']}\n{row['answer_text']}"
        self.master.clipboard_clear()
        self.master.clipboard_append(text)

    def start_progress(self):
        self.progress.pack(side="left")
        self.progress.start(10)

    def stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()

    def disable_inputs(self):
        widgets = [
            self.source_text,
            self.sim_button,
            self.save_button,
            self.load_button,
            self.delete_button,
            self.undo_button,
            self.new_button,
        ]
        for widget in self.entries + widgets:
            try:
                widget.configure(state="disabled")
            except Exception:
                pass
        self.gen_button.configure(state="disabled")

    def enable_inputs(self):
        widgets = [
            self.source_text,
            self.sim_button,
            self.save_button,
            self.load_button,
            self.delete_button,
            self.undo_button,
            self.new_button,
        ]
        for widget in self.entries + widgets:
            try:
                widget.configure(state="normal")
            except Exception:
                pass
        self.gen_button.configure(state="normal")

    def start_ai_animation(self):
        self.ai_running = True
        self.ai_dots = "..."

        def animate():
            if not self.ai_running:
                return
            self.ai_message_var.set(f"KI arbeitet {self.ai_dots}")
            self.ai_dots = self.ai_dots[:-1] if len(self.ai_dots) > 1 else "..."
            self.master.after(500, animate)

        animate()

    def stop_ai_animation(self):
        self.ai_running = False
        self.ai_message_var.set("")

    def show_message(self, message, error=False, success=False, info=False):
        """Display a status message and toast."""

        self.message_var.set(message)
        if error:
            color = "#dc3545"
        elif success:
            color = "#28a745"
        elif info:
            color = "#17a2b8"
        else:
            color = "black"
        self.message_label.configure(foreground=color)
        if error or success or info:
            ToastNotification(
                title="",
                message=message,
                duration=4000,
                bootstyle="secondary",
                position=(0, 0, "se"),
            ).show_toast()
        self.master.after(5000, lambda: self.message_var.set(""))

    def highlight_form(self):
        self.form.configure(highlightbackground="#FFA64D", highlightthickness=2)

    def clear_highlight(self):
        self.form.configure(highlightthickness=0)

    def on_escape(self, _=None):
        self.clear_highlight()
        self.filter_text.set("")

    def on_filter_change(self, *_):
        if self.search_after_id:
            self.master.after_cancel(self.search_after_id)
        self.search_after_id = self.master.after(300, self.apply_filter)

    def clear_form(self):
        """Clear all entry widgets and reset edit mode."""
        for widget in self.entries:
            if isinstance(widget, Text):
                widget.delete("1.0", END)
            elif isinstance(widget, ttk.Combobox):
                widget.set("")
            else:
                widget.delete(0, END)
        self.edit_index = None
        self.current_suggestion_index = None
        self.clear_highlight()

    def load_entry(self):
        """Load the selected row from the table into the form for editing."""
        selection = self.tree.selection()
        if not selection:
            self.show_message("Bitte zuerst einen Eintrag auswählen.", error=True)
            return
        index = self.tree.index(selection[0])
        row = self.df.iloc[index]
        for widget, col in zip(self.entries, COLUMNS):
            value = str(row[col]) if pd.notna(row[col]) else ""
            if isinstance(widget, Text):
                widget.delete("1.0", END)
                widget.insert("1.0", value)
            elif isinstance(widget, ttk.Combobox):
                widget.set(value)
            else:
                widget.delete(0, END)
                widget.insert(0, value)
        self.edit_index = index
        self.highlight_form()

    def delete_entry(self):
        """Delete the selected row from the table and CSV."""
        selection = self.tree.selection()
        if not selection:
            self.show_message("Bitte zuerst einen Eintrag auswählen.", error=True)
            return
        index = self.tree.index(selection[0])
        self.trash.append(self.df.iloc[index].to_dict())
        self.df = self.df.drop(self.df.index[index]).reset_index(drop=True)
        save_kb(self.df, self.csv_file)
        self.refresh_tree()
        self.edit_index = None
        self.show_message("Eintrag gelöscht.", success=True)

    def undo_delete(self):
        if not self.trash:
            self.show_message("Kein Eintrag zum Wiederherstellen.", error=True)
            return
        row = self.trash.pop()
        self.df.loc[len(self.df)] = row
        save_kb(self.df, self.csv_file)
        self.refresh_tree()
        self.show_message("Eintrag wiederhergestellt.", success=True)

    def check_spelling(self, widget):
        text = widget.get("1.0", END)
        widget.tag_delete("misspell")
        widget.tag_configure("misspell", underline=True, foreground="#d9534f")
        try:
            matches = self.lt.check(text)
        except Exception:
            matches = []
        for m in matches:
            start = f"1.0+{m.offset}c"
            end = f"1.0+{m.offset + m.errorLength}c"
            widget.tag_add("misspell", start, end)

    def get_entry_values(self):
        values = []
        for widget in self.entries:
            if isinstance(widget, Text):
                values.append(widget.get("1.0", END).strip())
            elif isinstance(widget, ttk.Combobox):
                values.append(widget.get().strip())
            else:
                values.append(widget.get().strip())
        return dict(zip(COLUMNS, values))

    def set_api_key(self):
        """Prompt the user for an OpenAI API key."""
        key = simpledialog.askstring("API-Key", "OpenAI API-Key eingeben:", show="*")
        if key:
            self.api_key = key
            self.client = OpenAI(api_key=key)

    def suggest_improvement(self, text):
        """Ask OpenAI for improvement hints for the given text."""
        if not self.client:
            return ""
        prompt = (
            "Der folgende Text reicht nicht aus, um vollst\u00e4ndige FAQ-Vorschl\u00e4ge zu erzeugen. "
            "Gib in einem Satz an, welche zus\u00e4tzlichen Informationen sinnvoll w\u00e4ren:"
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt + "\n" + text}],
                max_tokens=60,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""

    def generate_suggestions(self):
        """Generate suggestions via OpenAI and update the list."""
        global CATEGORY_MAPPED
        if not self.client:
            self.set_api_key()
            if not self.client:
                return
        text = self.source_text.get("1.0", END).strip()
        self.disable_inputs()
        self.start_ai_animation()
        new_entries = generate_suggestions(text, self.client)
        self.stop_ai_animation()
        self.enable_inputs()
        if not new_entries:
            self.ai_message_var.set("Keine geeigneten Vorschläge gefunden.")
            return
        self.ai_message_var.set("")
        if CATEGORY_MAPPED:
            self.show_message(
                "Unbekannte Kategorie → Sonstiges übernommen", info=True
            )
            CATEGORY_MAPPED = False
        self.suggestions.extend(new_entries)
        self.refresh_suggestion_box()

    def load_suggestion(self):
        """Load the selected suggestion into the form."""
        sel = self.suggestion_box.curselection()
        if not sel:
            return
        index = sel[0]
        suggestion = self.suggestions[index]
        self.current_suggestion_index = index
        try:
            data = FAQEntry(**suggestion).model_dump()
        except Exception:
            data = suggestion
        for widget, col in zip(self.entries, COLUMNS):
            val = data.get(col, "")
            if isinstance(widget, Text):
                widget.delete("1.0", END)
                widget.insert("1.0", val)
            elif isinstance(widget, ttk.Combobox):
                widget.set(val)
            else:
                widget.delete(0, END)
                widget.insert(0, val)
        self.highlight_form()

    def delete_suggestion(self):
        """Remove the selected suggestion from the list."""
        sel = self.suggestion_box.curselection()
        if not sel:
            return
        idx = sel[0]
        self.suggestions.pop(idx)
        if self.current_suggestion_index == idx:
            self.current_suggestion_index = None
        self.refresh_suggestion_box()

    def check_similar(self):
        data = self.get_entry_values()
        if data is None:
            return
        question = data["faq_question"]
        sims = find_similar(self.df, question, self.client, self.embeddings)
        if sims:
            self.highlight_row(sims[0][1].name)
            self.show_message("Ähnliche Frage gefunden.")
        else:
            self.show_message("Keine ähnlichen Fragen gefunden.")

    def save_entry(self):
        data = self.get_entry_values()
        if not data:
            return
        if not data["faq_question"] or not data["answer_text"]:
            self.show_message("Antwort darf nicht leer sein", error=True)
            return
        if data["category"] not in CATEGORIES:
            self.show_message("Ungültige Kategorie", error=True)
            return
        if (
            self.entries[1].tag_ranges("misspell")
            or self.entries[2].tag_ranges("misspell")
        ):
            self.show_message("Rechtschreibfehler vorhanden", error=True)
            return
        for k in COLUMNS:
            if k not in {"faq_question", "answer_text"} and not data[k]:
                data[k] = "Keine Angabe"
        try:
            new_row = FAQEntry(**data).model_dump()
        except Exception as exc:
            self.show_message(f"Ungültige Eingaben: {exc}", error=True)
            return
        if self.edit_index is None:
            row_index = len(self.df)
            self.df.loc[row_index] = new_row
        else:
            row_index = self.edit_index
            for col, val in new_row.items():
                self.df.at[row_index, col] = val
            self.edit_index = None
        save_kb(self.df, self.csv_file)
        self.show_message("Eintrag wurde gespeichert.", success=True)
        self.refresh_tree()
        self.highlight_row(row_index)
        self.clear_form()
        self.clear_highlight()
        if self.current_suggestion_index is not None:
            self.suggestions.pop(self.current_suggestion_index)
            self.refresh_suggestion_box()
            self.current_suggestion_index = None


if __name__ == "__main__":
    root = tb.Window(themename="flatly")
    app = KBManager(root)
    root.mainloop()
