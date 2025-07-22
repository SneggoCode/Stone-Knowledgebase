import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from kb_manager import generate_suggestions, COLUMNS, FAQEntry  # noqa: E402


class DummyResp:
    def __init__(self, content):
        self.choices = [type('obj', (), {'message': type('obj', (), {'content': content})})]


class DummyCreate:
    def __init__(self, contents):
        self.contents = contents
        self.index = 0

    def create(self, **kwargs):
        content = self.contents[self.index]
        self.index += 1
        return DummyResp(content)


class DummyChat:
    def __init__(self, contents):
        self.completions = DummyCreate(contents)


class DummyClient:
    def __init__(self, contents):
        self.chat = DummyChat(contents)


def test_generate_length_and_schema():
    data = {
        "entries": [
            {
                "category": "Produkt",
                "faq_question": "Welche Steine sind hart?",
                "answer_text": "Alle Granitsteine sind sehr hart.",
                "stone_type": "Granit",
                "product_form": "Block",
                "product_size": "10 cm",
                "eigenschaft": "sehr hart",
                "anwendung": "Gartenbau",
            }
        ]
    }
    client = DummyClient([json.dumps(data)])
    entries = generate_suggestions("Stein ist hart", client)
    assert len(entries) >= 1
    for entry in entries:
        FAQEntry(**entry)
        for key in COLUMNS:
            assert key in entry
