import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk, Label, Entry, Text, Button, Listbox, Scrollbar, END, messagebox

CSV_FILE = 'knowledgebase.csv'
COLUMNS = ['stone_type','product_form','grain_size_mm','eigenschaft','anwendung','faq_question','answer_text']

def load_kb():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=COLUMNS)

def save_kb(df):
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
        self.df = load_kb()

        labels = ['Steinart','Produktform','Korngroesse(mm)','Eigenschaft','Anwendung','Frage','Antwort']
        self.entries = []
        for i, lbl in enumerate(labels):
            Label(master, text=lbl).grid(row=i, column=0, sticky='e')
            if lbl in ['Antwort']:
                txt = Text(master, width=50, height=4)
                txt.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(txt)
            else:
                ent = Entry(master, width=50)
                ent.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(ent)

        Button(master, text='Ähnliche Fragen anzeigen', command=self.check_similar).grid(row=7, column=0, pady=5)
        Button(master, text='Speichern', command=self.save_entry).grid(row=7, column=1, pady=5)

        self.listbox = Listbox(master, width=100)
        self.listbox.grid(row=8, column=0, columnspan=2, padx=5, pady=5)
        self.scrollbar = Scrollbar(master, orient='vertical', command=self.listbox.yview)
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.grid(row=8, column=2, sticky='ns')

    def get_entry_values(self):
        values = []
        for widget in self.entries:
            if isinstance(widget, Text):
                values.append(widget.get('1.0', END).strip())
            else:
                values.append(widget.get().strip())
        return values

    def check_similar(self):
        values = self.get_entry_values()
        question = values[5]
        sims = find_similar(self.df, question)
        self.listbox.delete(0, END)
        for score, row in sims:
            display = f"({score:.2f}) {row['faq_question']} -> {row['answer_text']}"
            self.listbox.insert(END, display)
        if not sims:
            self.listbox.insert(END, 'Keine ähnlichen Fragen gefunden.')

    def save_entry(self):
        values = self.get_entry_values()
        if any(v == '' for v in values):
            messagebox.showerror('Fehler', 'Bitte alle Felder ausfüllen.')
            return
        new_row = dict(zip(COLUMNS, values))
        self.df = self.df.append(new_row, ignore_index=True)
        save_kb(self.df)
        messagebox.showinfo('Gespeichert', 'Eintrag wurde gespeichert.')
        for widget in self.entries:
            if isinstance(widget, Text):
                widget.delete('1.0', END)
            else:
                widget.delete(0, END)
        self.listbox.delete(0, END)

if __name__ == '__main__':
    root = Tk()
    app = KBManager(root)
    root.mainloop()
