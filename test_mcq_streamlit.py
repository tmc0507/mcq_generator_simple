import json

from streamlit.testing.v1 import AppTest

from mcq_streamlit import generate_questions, render_prompt


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate_text(self, messages, **kwargs):
        self.calls += 1
        question = {
            "question": ["Alpha là gì?", "Beta là gì?", "Gamma là gì?"][self.calls - 1],
            "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
            "correct_answer": "A", "concept": "ví dụ", "difficulty": "easy",
            "explanation": "Dữ kiện gốc.", "evidence": "Dữ kiện gốc.",
            "source_type": "page", "source_index": 1,
        }
        return json.dumps({"questions": [question]})


def test_app():
    data = {"blocks": [{"type": "page", "index": 1, "text": "Dữ kiện gốc."}],
            "chunks": [{"text": "Dữ kiện gốc. " * 20, "source_types": ["page"], "source_indices": [1]}]}
    questions = list(generate_questions(FakeModel(), data, 3, "Câu {index}/{total}\n{source}"))
    assert len(questions) == 3
    assert "Câu 3/3" in questions[2][1]
    assert "Dữ kiện gốc." in render_prompt("{source}", data["chunks"][0], 1, 3, [])
    app = AppTest.from_file("mcq_streamlit.py").run(timeout=30)
    assert not app.exception, app.exception
    assert app.number_input[0].value == 1
    assert app.text_area[0].value


if __name__ == "__main__":
    test_app()
    print("OK: số câu, prompt và giao diện")
