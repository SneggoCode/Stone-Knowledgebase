import json

import kb_manager


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


def test_category_mapping():
    data = {
        "entries": [
            {
                "category": "payment",
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
    client = DummyClient([json.dumps(data)])
    entries = kb_manager.generate_suggestions("text", client)
    assert entries[0]["category"] == "Sonstiges"
