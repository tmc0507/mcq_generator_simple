import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import mcq_streamlit
from mcq_streamlit import (
    create_exam, format_exam_txt, generate_questions, question_count,
    render_prompt, score_attempt, teacher_reply,
)


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate_text(self, messages, **kwargs):
        self.calls += 1
        question = {
            "question": ["Alpha là gì?", "Beta là gì?", "Gamma là gì?"][(self.calls - 1) % 3],
            "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
            "correct_answer": "A", "concept": "ví dụ", "difficulty": "easy",
            "explanation": "Dữ kiện gốc.", "evidence": "Dữ kiện gốc.",
            "source_type": "page", "source_index": 1,
        }
        return json.dumps({"questions": [question]})


def test_app():
    data = {"blocks": [{"type": "page", "index": 1, "text": "Dữ kiện gốc."}],
            "chunks": [{"text": "Dữ kiện gốc. " * 20, "source_types": ["page"], "source_indices": [1]}]}
    model = FakeModel()
    progress = []
    questions = list(generate_questions(model, data, 3, "Câu {index}/{total}; nhóm {count}\n{source}",
                                        on_attempt=lambda attempt, accepted: progress.append((attempt, accepted))))
    assert len(questions) == 1
    assert model.calls == 1
    assert progress == [(1, 0)]
    assert "Câu 1/3; nhóm 3" in questions[0][1]
    assert "Dữ kiện gốc." in render_prompt("{source}", data["chunks"][0], 1, 3, [])
    app = AppTest.from_file("mcq_streamlit.py").run(timeout=30)
    assert not app.exception, app.exception
    assert app.chat_input[0].proto.accept_file
    assert str(app.chat_input[0].proto.submit_mode) == "2"
    assert set(app.chat_input[0].proto.file_type) == {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".csv"}
    assert app.selectbox[0].value in {"modelv1", "modelv2"}
    assert app.toggle[0].label == "Kiểm tra đáp án bằng AI"
    assert app.toggle[0].value is True
    assert app.text_area[0].value
    app.chat_input[0].set_value("Tạo 3 câu hỏi").run(timeout=30)
    assert not app.exception, app.exception
    assert any("đính kèm tài liệu" in item.value.lower() for item in app.markdown)


def test_generation_has_bounded_batches_and_validates_each_question():
    data = {"blocks": [{"type": "page", "index": 1, "text": "Dữ kiện gốc."}],
            "chunks": [{"text": "Dữ kiện gốc. " * 20, "source_types": ["page"], "source_indices": [1]}]}

    class BatchModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.verifier_calls = 0

        def generate_text(self, messages, **kwargs):
            if "verifier" in messages[0]["content"]:
                self.verifier_calls += 1
                return json.dumps({"status": "PASS", "issues": [], "corrected_question": None})
            self.calls += 1
            questions = [{"question": f"Câu hỏi {index + 1}?",
                          "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
                          "correct_answer": "A", "concept": "ví dụ", "difficulty": "easy",
                          "explanation": "Dữ kiện gốc.", "evidence": "Dữ kiện gốc.",
                          "source_type": "page", "source_index": 1} for index in range(3)]
            return json.dumps({"questions": questions})

    model = BatchModel()
    stats = {}
    questions = list(generate_questions(model, data, 3, mcq_streamlit.DEFAULT_PROMPT,
                                        verify=True, stats=stats))
    assert len(questions) == 3
    assert stats["batches"] == 1
    assert stats["accepted"] == 3
    assert model.calls == 1
    assert model.verifier_calls == 3

    class DuplicateModel(BatchModel):
        def generate_text(self, messages, **kwargs):
            result = super().generate_text(messages, **kwargs)
            if "verifier" in messages[0]["content"]:
                return result
            parsed = json.loads(result)
            parsed["questions"] = [parsed["questions"][0]] * 3
            return json.dumps(parsed)

    duplicate_model = DuplicateModel()
    duplicate_stats = {}
    assert len(list(generate_questions(duplicate_model, data, 3, mcq_streamlit.DEFAULT_PROMPT,
                                       verify=True, stats=duplicate_stats))) == 1
    assert duplicate_model.verifier_calls == 1
    assert duplicate_stats["duplicate"] == 2


def test_exam_file_and_count():
    assert question_count("Tạo 10 câu hỏi") == 10
    assert question_count("Tạo bộ đề") == 10
    try:
        question_count("Tạo 51 câu")
    except ValueError:
        pass
    else:
        raise AssertionError("Phải chặn số câu vượt giới hạn")

    data = {"blocks": [{"type": "page", "index": 1, "text": "Dữ kiện gốc."}],
            "chunks": [{"text": "Dữ kiện gốc. " * 20, "source_types": ["page"], "source_indices": [1]}]}

    class VerifierModel(FakeModel):
        def generate_text(self, messages, **kwargs):
            if "verifier" in messages[0]["content"]:
                return json.dumps({"status": "PASS", "issues": [], "corrected_question": None})
            return super().generate_text(messages, **kwargs)

    originals = (mcq_streamlit.EXAM, mcq_streamlit.EXAM_TXT,
                 mcq_streamlit.OUTPUT, mcq_streamlit.extract, mcq_streamlit.load_model)
    try:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path = folder / "model.gguf"
            model_path.touch()
            mcq_streamlit.EXAM = folder / "questions.json"
            mcq_streamlit.EXAM_TXT = folder / "questions.txt"
            mcq_streamlit.OUTPUT = folder / "debug.txt"
            mcq_streamlit.extract = lambda document: data
            mcq_streamlit.load_model = lambda path: VerifierModel()
            uploaded = SimpleNamespace(name="sample.csv", getvalue=lambda: b"value\nexample\n")
            exam = create_exam(uploaded, model_path, 1, "{source}")
            assert len(exam["questions"]) == 1
            assert mcq_streamlit.EXAM.is_file() and mcq_streamlit.EXAM_TXT.is_file()
            assert mcq_streamlit.EXAM_TXT.read_text(encoding="utf-8") == format_exam_txt(exam)
            assert "Đáp án:" not in format_exam_txt(exam)
            snapshots = []
            fast_exam = create_exam(uploaded, model_path, 1, "{source}", verify=False,
                                    on_question=lambda partial: snapshots.append(partial["accepted_questions"]))
            assert snapshots == [1]
            assert fast_exam["verification_enabled"] is False
            assert fast_exam["questions"][0]["verifier_status"] == "SKIPPED"
            assert "BẢN NHÁP" in format_exam_txt(fast_exam)
    finally:
        (mcq_streamlit.EXAM, mcq_streamlit.EXAM_TXT,
         mcq_streamlit.OUTPUT, mcq_streamlit.extract, mcq_streamlit.load_model) = originals


def test_stopped_exam_is_kept_in_chat():
    with tempfile.TemporaryDirectory() as directory:
        script = f'''import mcq_streamlit as app
from pathlib import Path
app.EXAM = Path({str(Path(directory) / "questions.json")!r})
app.EXAM_TXT = Path({str(Path(directory) / "questions.txt")!r})
app.main()
'''
        question = {"id": 1, "question": "Dữ kiện gốc là gì?",
                    "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
                    "correct_answer": "A", "concept": "ví dụ", "difficulty": "easy",
                    "explanation": "Dữ kiện gốc.", "evidence": "Dữ kiện gốc.",
                    "source_type": "page", "source_index": 1}
        exam = {"document": "sample.csv", "requested_questions": 3,
                "accepted_questions": 1, "generation_stats": {}, "questions": [question]}
        app = AppTest.from_string(script).run(timeout=30)
        app.session_state["partial_exam"] = exam
        app.session_state["generating_exam"] = True
        app.session_state["messages"] = [{"role": "user", "content": "Tạo 3 câu hỏi"}]
        app.run(timeout=30)
        assert not app.exception, app.exception
        assert "Đã dừng tạo đề" in app.chat_message[-2].markdown[0].value
        assert Path(directory, "questions.json").is_file()
        assert "Dữ kiện gốc" in Path(directory, "questions.txt").read_text(encoding="utf-8")
        app.session_state["generating_exam"] = True
        app.run(timeout=30)
        assert "trước khi có câu hỏi" in app.chat_message[-1].markdown[0].value


def test_scoring_and_teacher_chat():
    questions = [
        {"id": 1, "question": "A?", "correct_answer": "A", "concept": "Đại số", "explanation": "Quy tắc A"},
        {"id": 2, "question": "B?", "correct_answer": "B", "concept": "Hình học", "explanation": "Quy tắc B"},
        {"id": 3, "question": "C?", "correct_answer": "C", "concept": "Hình học", "explanation": "Quy tắc C"},
    ]
    report = score_attempt(questions, {"1": "A", "2": "D"})
    assert (report["correct"], report["total"], report["percent"]) == (1, 3, 33.3)
    assert report["by_concept"] == {
        "Đại số": {"correct": 1, "total": 1},
        "Hình học": {"correct": 0, "total": 2},
    }
    assert report["details"][2]["student_answer"] == "Bỏ trống"

    class ChatModel:
        def generate_text(self, messages, **kwargs):
            assert "Hình học" in messages[1]["content"]
            assert "Bỏ trống" in messages[1]["content"]
            assert kwargs["do_sample"] is False
            return "<think>private</think>Ôn Hình học."

    assert teacher_reply(ChatModel(), "Cần ôn gì?", report) == "Ôn Hình học."

    class OverconfidentModel:
        def generate_text(self, messages, **kwargs):
            assert "(2, 'Hình học')" in messages[1]["content"]
            return "Sinh viên đã nắm vững Đại số nhưng chưa hiểu rõ Hình học."

    assert teacher_reply(OverconfidentModel(), "Nhận xét?", report) == (
        "Sinh viên trả lời đúng câu hỏi về Đại số nhưng trả lời sai câu hỏi về Hình học."
    )


def test_one_chat_from_exam_to_review():
    exam = {"document": "sample.csv", "questions": [
        {"id": 1, "question": "Alpha là gì?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
         "correct_answer": "A", "concept": "Đại số", "explanation": "Ví dụ A"},
        {"id": 2, "question": "Beta là gì?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
         "correct_answer": "B", "concept": "Hình học", "explanation": "Ví dụ B"},
    ]}
    with tempfile.TemporaryDirectory() as directory:
        attempts = Path(directory) / "attempts.json"
        script = f'''import mcq_streamlit as app
from pathlib import Path
app.ATTEMPTS = Path({str(attempts)!r})
class ReviewModel:
    def generate_text(self, messages, **kwargs):
        return "**Điểm mạnh:** Trả lời đúng câu 1.\\n**Cần cải thiện:** Trả lời sai câu 2.\\n**Gợi ý ôn tập:** Ôn Hình học."
app.load_model = lambda path=None: ReviewModel()
app.main()
'''
        app = AppTest.from_string(script).run(timeout=30)
        app.session_state["messages"] = [{"role": "assistant", "kind": "exam", "id": "demo", "exam": exam}]
        app.session_state["active_exam_id"] = "demo"
        app.run(timeout=30)
        assert not app.exception, app.exception
        assert len(app.radio) == 2
        assert any(button.label == "Nộp bài" for button in app.button)
        app.text_input[0].set_value("Sinh viên A")
        app.radio[0].set_value("A")
        app.radio[1].set_value("A")
        next(button for button in app.button if button.label == "Nộp bài").click().run(timeout=30)
        assert not app.exception, app.exception
        assert json.loads(attempts.read_text(encoding="utf-8"))[0]["report"]["correct"] == 1
        app.chat_input[0].set_value("Nhận xét bài làm").run(timeout=30)
        assert not app.exception, app.exception
        assert len(app.chat_message) == 5
        assert "Ôn Hình học" in app.chat_message[-1].markdown[0].value


def test_ui_generation_rechecks_revision():
    data = {"blocks": [{"type": "page", "index": 1, "text": "Dữ kiện gốc."}],
            "chunks": [{"text": "Dữ kiện gốc. " * 20, "source_types": ["page"], "source_indices": [1]}]}

    class RevisedModel(FakeModel):
        def generate_text(self, messages, **kwargs):
            if "verifier" in messages[0]["content"]:
                return json.dumps({"status": "REVISE", "issues": ["wrong answer"],
                                   "corrected_question": {"question": "Hỏng", "choices": {"A": "A"},
                                                          "correct_answer": "D", "source_type": "page", "source_index": 1}})
            return super().generate_text(messages, **kwargs)

    assert list(generate_questions(RevisedModel(), data, 1, "{source}", verify=True)) == []


def test_attempts_are_saved():
    original = mcq_streamlit.ATTEMPTS
    try:
        with tempfile.TemporaryDirectory() as directory:
            mcq_streamlit.ATTEMPTS = Path(directory) / "attempts.json"
            mcq_streamlit.save_attempt({"student": "A"})
            mcq_streamlit.save_attempt({"student": "B"})
            assert [item["student"] for item in mcq_streamlit.read_attempts()] == ["A", "B"]
    finally:
        mcq_streamlit.ATTEMPTS = original


if __name__ == "__main__":
    test_app()
    test_exam_file_and_count()
    test_stopped_exam_is_kept_in_chat()
    test_scoring_and_teacher_chat()
    test_one_chat_from_exam_to_review()
    test_ui_generation_rechecks_revision()
    test_attempts_are_saved()
    print("OK: số câu, prompt, giao diện, chấm bài và chat")
