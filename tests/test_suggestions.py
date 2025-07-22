import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import kb_manager  # noqa: E402


class DummyResp:
    def __init__(self, content):
        self.choices = [
            type("obj", (), {"message": type("obj", (), {"content": content})})
        ]


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


def test_generate_retries():
    bad = "{invalid json"
    good = json.dumps(
        {
            "entries": [
                {
                    "category": "Produkt",
                    "faq_question": "Q",
                    "answer_text": "A",
                    "stone_type": "",
                    "product_form": "",
                    "product_size": "10 cm",
                    "eigenschaft": "",
                    "anwendung": "",
                }
            ]
        }
    )
    client = DummyClient([bad, good])
    entries = kb_manager.generate_suggestions("text", client)
    assert len(entries) == 1
    assert kb_manager.LAST_RAW_CONTENT == ""


def test_generate_all_fail():
    bad1 = "{invalid"
    bad2 = "{still bad"
    client = DummyClient([bad1, bad2])
    entries = kb_manager.generate_suggestions("text", client)
    assert entries == []
    assert kb_manager.LAST_RAW_CONTENT == bad2
