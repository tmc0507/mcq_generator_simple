import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import mcq_streamlit
from mcq_streamlit import (
    create_exam, generate_questions, import_result, question_count,
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
    assert set(app.chat_input[0].proto.file_type) == {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".csv", ".json"}
    assert app.selectbox[0].value in {"modelv2", "modelv3"}
    assert tuple(mcq_streamlit.MODEL_OPTIONS) == ("modelv2", "modelv3")
    assert [path.name for path in mcq_streamlit.MODEL_OPTIONS.values()] == [
        "modelv2.gguf", "modelv3.gguf"]
    assert app.toggle[0].label == "Kiểm tra đáp án bằng AI"
    assert app.toggle[0].value is False
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

    draft = list(generate_questions(FakeModel(), data, 6, mcq_streamlit.DEFAULT_PROMPT, verify=False))
    assert len(draft) == 2

    class KeepWholeBatch(FakeModel):
        def generate_text(self, messages, **kwargs):
            self.calls += 1
            assert kwargs["max_new_tokens"] == 660
            assert kwargs["schema"]["properties"]["questions"]["items"]["required"] == [
                "question", "choices", "correct_answer", "concept"]
            assert set(kwargs["schema"]["properties"]["questions"]["items"]["properties"]) == {
                "question", "choices", "correct_answer", "concept", "evidence"}
            return json.dumps({"questions": [
                {"question": "Câu trùng?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
                 "correct_answer": "A", "evidence": "Dẫn chứng không có trong tài liệu."},
                {"question": "Câu trùng?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
                 "correct_answer": "A"},
                {"question": "Câu thiếu đáp án?", "choices": {"A": "Một"},
                 "correct_answer": None},
                {"question": "Câu sinh dư?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
                 "correct_answer": "B"},
            ]})

    whole_model = KeepWholeBatch()
    whole_stats = {}
    whole_batch = [question for question, _ in generate_questions(
        whole_model, data, 3, mcq_streamlit.DEFAULT_PROMPT, verify=False, stats=whole_stats)]
    assert whole_model.calls == 1
    assert whole_stats["generated"] == 4
    assert len(whole_batch) == whole_stats["accepted"] == 3
    assert whole_stats["duplicate"] == 1
    assert whole_batch[0]["evidence"] == "Dẫn chứng không có trong tài liệu."
    assert whole_batch[1]["correct_answer"] == ""
    assert whole_batch[1]["choices"]["D"] == ""

    repeated = {"blocks": [{"type": "page", "index": i,
                             "text": "Course header\n" + f"Chi tiết riêng của trang {i}. " * 10}
                            for i in (1, 2, 3)],
                "chunks": [{"text": "Course header\n" + "Chi tiết riêng của từng trang. " * 15,
                            "source_types": ["page"], "source_indices": [1]}]}
    cleaned = list(generate_questions(FakeModel(), repeated, 1, "{source}", verify=False))
    assert "Course header" not in cleaned[0][1]

    class UncheckedEvidence(FakeModel):
        def generate_text(self, messages, **kwargs):
            result = json.loads(super().generate_text(messages, **kwargs))
            result["questions"][0]["evidence"] = "Model trích dẫn không có trong tài liệu."
            return json.dumps(result)

    unchecked = list(generate_questions(UncheckedEvidence(), data, 1,
                                        mcq_streamlit.DEFAULT_PROMPT, verify=False))
    assert len(unchecked) == 1
    assert unchecked[0][0]["evidence"] == "Model trích dẫn không có trong tài liệu."
    assert unchecked[0][0]["evidence_score"] is None


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

    originals = (mcq_streamlit.EXAM, mcq_streamlit.DOCUMENTS_INBOX,
                 mcq_streamlit.extract, mcq_streamlit.load_model)
    try:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path = folder / "model.gguf"
            model_path.touch()
            mcq_streamlit.EXAM = folder / "questions.json"
            mcq_streamlit.DOCUMENTS_INBOX = folder / "documents"
            mcq_streamlit.extract = lambda document: data
            mcq_streamlit.load_model = lambda path: VerifierModel()
            uploaded = SimpleNamespace(name="sample.csv", getvalue=lambda: b"value\nexample\n")
            exam = create_exam(uploaded, model_path, 1, "{source}")
            assert len(exam["questions"]) == 1
            assert exam["exam_id"]
            assert exam["model"] == "model.gguf"
            assert mcq_streamlit.EXAM.is_file()
            assert (mcq_streamlit.DOCUMENTS_INBOX / "sample.csv").read_bytes() == uploaded.getvalue()
            assert list(folder.glob("*.txt")) == []
            saved_exam = mcq_streamlit.EXAM.read_bytes()
            class EmptyModel:
                def generate_text(self, messages, **kwargs):
                    return '{"questions": []}'
            mcq_streamlit.load_model = lambda path: EmptyModel()
            try:
                create_exam(uploaded, model_path, 1, "{source}")
            except ValueError:
                pass
            else:
                raise AssertionError("Không được coi danh sách câu hỏi rỗng là bộ đề")
            assert mcq_streamlit.EXAM.read_bytes() == saved_exam
            mcq_streamlit.load_model = lambda path: VerifierModel()
            revised_upload = SimpleNamespace(name="sample.csv", getvalue=lambda: b"new,document\n")
            create_exam(revised_upload, model_path, 1, "{source}")
            assert (mcq_streamlit.DOCUMENTS_INBOX / "sample.csv").read_bytes() == uploaded.getvalue()
            assert (mcq_streamlit.DOCUMENTS_INBOX / "sample_2.csv").read_bytes() == revised_upload.getvalue()
            snapshots = []
            class FastBatchModel(VerifierModel):
                def generate_text(self, messages, **kwargs):
                    result = json.loads(super().generate_text(messages, **kwargs))
                    return json.dumps({"questions": [dict(result["questions"][0], question=f"Câu {i}?")
                                                     for i in range(3)]})

            mcq_streamlit.load_model = lambda path: FastBatchModel()
            fast_exam = create_exam(uploaded, model_path, 3, "{source}", verify=False,
                                    on_question=lambda partial: snapshots.append(
                                        json.loads(mcq_streamlit.EXAM.read_text(encoding="utf-8"))["accepted_questions"]))
            assert snapshots == [1, 2, 3]
            assert fast_exam["verification_enabled"] is False
            assert fast_exam["questions"][0]["verifier_status"] == "SKIPPED"
            assert fast_exam["questions"][0]["evidence_score"] is None
            assert json.loads(mcq_streamlit.EXAM.read_text(encoding="utf-8"))["verification_enabled"] is False
    finally:
        (mcq_streamlit.EXAM, mcq_streamlit.DOCUMENTS_INBOX,
         mcq_streamlit.extract, mcq_streamlit.load_model) = originals


def test_stopped_exam_is_kept_in_chat():
    with tempfile.TemporaryDirectory() as directory:
        script = f'''import mcq_streamlit as app
from pathlib import Path
app.EXAM = Path({str(Path(directory) / "questions.json")!r})
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
        assert len(json.loads(Path(directory, "questions.json").read_text(encoding="utf-8"))["questions"]) == 1
        assert "Dữ kiện gốc" in app.chat_message[-1].code[0].value
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
    assert score_attempt([dict(questions[0], correct_answer="")], {})["correct"] == 0

    class ChatModel:
        def generate_text(self, messages, **kwargs):
            assert "Hình học" in messages[1]["content"]
            assert "đáp án" in messages[0]["content"]
            assert kwargs["do_sample"] is False
            return "<think>private</think>Nên ôn lại Hình học."

    feedback = teacher_reply(ChatModel(), "Cần ôn gì?", report)
    assert "Đại số: đúng 1/1 câu" in feedback
    assert "Câu 2 (Hình học): trả lời D, đáp án B" in feedback
    assert "Câu 3 (Hình học): bỏ trống" in feedback
    assert feedback.endswith("Nên ôn lại Hình học.")

    class OverconfidentModel:
        def generate_text(self, messages, **kwargs):
            assert "Chủ đề cần ôn: Hình học" in messages[1]["content"]
            return "Sinh viên đã nắm vững Đại số nhưng chưa hiểu rõ Hình học."

    assert teacher_reply(OverconfidentModel(), "Nhận xét?", report).endswith(
        "Nên ôn lại Hình học qua bài giảng và câu hỏi luyện tập.")


def test_one_chat_from_exam_to_review():
    exam = {"exam_id": "demo-exam", "document": "sample.csv", "questions": [
        {"id": 1, "question": "Alpha là gì?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
         "correct_answer": "A", "concept": "Đại số", "explanation": "Ví dụ A"},
        {"id": 2, "question": "Beta là gì?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
         "correct_answer": "B", "concept": "Hình học", "explanation": "Ví dụ B"},
    ]}
    uploaded = SimpleNamespace(getvalue=lambda: json.dumps({
        "exam_id": "demo-exam", "student": "Sinh viên A", "answers": {"1": "A", "2": "A"},
    }).encode("utf-8"))
    attempt = import_result(uploaded, exam)
    assert attempt["report"]["correct"] == 1
    ungradable = dict(exam, questions=[dict(exam["questions"][0], correct_answer="")])
    try:
        import_result(uploaded, ungradable)
    except ValueError as exc:
        assert "thiếu đáp án" in str(exc)
    else:
        raise AssertionError("Đã chấm câu chưa có đáp án đúng")
    for invalid in (
        {"exam_id": "wrong", "student": "A", "answers": {"1": "A"}},
        {"exam_id": "demo-exam", "student": "A", "answers": {"9": "A"}},
        {"exam_id": "demo-exam", "student": "A", "answers": {"1": "E"}},
    ):
        try:
            import_result(SimpleNamespace(getvalue=lambda value=invalid: json.dumps(value).encode()), exam)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid result: {invalid}")
    script = '''import mcq_streamlit as app
class ReviewModel:
    def generate_text(self, messages, **kwargs):
        return "Nên ôn lại Hình học."
app.load_model = lambda path=None: ReviewModel()
app.main()
'''
    app = AppTest.from_string(script).run(timeout=30)
    app.session_state["messages"] = [{"role": "assistant", "kind": "exam", "id": "demo", "exam": exam}]
    app.session_state["active_attempt"] = attempt
    app.run(timeout=30)
    assert not app.exception, app.exception
    assert not app.radio
    assert "output/questions.json" in app.chat_message[-1].markdown[0].value
    assert '"exam_id": "demo-exam"' in app.chat_message[-1].code[0].value
    app.chat_input[0].set_value("Nhận xét bài làm").run(timeout=30)
    assert not app.exception, app.exception
    assert "ôn lại Hình học" in app.chat_message[-1].markdown[0].value


def test_result_file_enters_review_flow():
    exam = {"exam_id": "demo-exam", "document": "sample.csv", "questions": [
        {"id": 1, "question": "Alpha?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
         "correct_answer": "A", "concept": "Đại số"},
    ]}
    payload = json.dumps({"exam_id": "demo-exam", "student": "SV001", "answers": {"1": "A"}}).encode()
    with tempfile.TemporaryDirectory() as directory:
        script = f'''import mcq_streamlit as app
from pathlib import Path
from types import SimpleNamespace
app.RESULTS_INBOX = Path({str(Path(directory) / "student_results")!r})
original = app.st.chat_input
def injected(*args, **kwargs):
    if app.st.session_state.pop("inject_result", False):
        return SimpleNamespace(text="", files=[SimpleNamespace(name="result.json", getvalue=lambda: {payload!r})])
    return original(*args, **kwargs)
app.st.chat_input = injected
try:
    app.main()
finally:
    app.st.chat_input = original
'''
        app = AppTest.from_string(script).run(timeout=30)
        app.session_state["messages"] = [{"role": "assistant", "kind": "exam", "id": "demo", "exam": exam}]
        app.session_state["inject_result"] = True
        app.run(timeout=30)
        assert not app.exception, app.exception
        assert app.session_state["active_attempt"]["report"]["correct"] == 1
        saved = list(Path(directory, "student_results").glob("*.json"))
        assert len(saved) == 1
        assert json.loads(saved[0].read_text(encoding="utf-8"))["exam_id"] == "demo-exam"
        assert any("Đã nhận kết quả" in item.value for item in app.markdown)


def test_world_model_student_result():
    folder = Path(__file__).parent / "examples" / "world_model"
    exam = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    payload = (folder / "student_result.json").read_bytes()
    report = import_result(SimpleNamespace(getvalue=lambda: payload), exam)["report"]
    assert (report["correct"], report["total"], report["percent"]) == (5, 10, 50.0)
    assert [item["id"] for item in report["details"] if not item["correct"]] == [2, 4, 6, 8, 10]


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


def test_inbox_finds_student_and_today_results():
    exam = {"exam_id": "demo-exam", "document": "sample.csv", "questions": [
        {"id": 1, "question": "Alpha?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
         "correct_answer": "A", "concept": "Đại số"},
    ]}
    original = mcq_streamlit.RESULTS_INBOX
    try:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            mcq_streamlit.RESULTS_INBOX = folder
            def write(name, student, answer, exam_id="demo-exam"):
                path = folder / name
                path.write_text(json.dumps({"exam_id": exam_id, "student": student,
                                            "answers": {"1": answer}}), encoding="utf-8")
                return path
            old = write("old.json", "SV001 - Nguyễn Văn An", "A")
            yesterday = (datetime.now() - timedelta(days=1)).timestamp()
            os.utime(old, (yesterday, yesterday))
            write("new.json", "SV001 - Nguyễn Văn An", "B")
            write("second.json", "SV002 - Trần Bình", "A")
            write("other_exam.json", "SV003", "A", exam_id="other")
            (folder / "broken.json").write_text("{", encoding="utf-8")
            records, skipped = mcq_streamlit.inbox_attempts(exam, today=True)
            assert len(records) == 2 and skipped == 2
            assert {record["attempt"]["student"] for record in records} == {
                "SV001 - Nguyễn Văn An", "SV002 - Trần Bình"}
            assert next(record for record in records if record["attempt"]["student"].startswith("SV001"))[
                "attempt"]["report"]["correct"] == 0
            assert mcq_streamlit.student_matches("Nhận xét sinh viên Nguyen Van An", "SV001 - Nguyễn Văn An")
            assert mcq_streamlit.student_matches("Nhận xét SV002", "SV002 - Trần Bình")
            assert not mcq_streamlit.student_matches("Nhận xét sinh viên", "SV002 - Trần Bình")
            class ReviewModel:
                def generate_text(self, messages, **kwargs):
                    return "Nên ôn lại Đại số."
            summary = mcq_streamlit.class_reply(ReviewModel(), "Nhận xét buổi kiểm tra hôm nay", records, skipped)
            assert "SV001" in summary and "SV002" in summary
            assert "Điểm trung bình: 50.0%" in summary
            assert "Bỏ qua 2 file" in summary
    finally:
        mcq_streamlit.RESULTS_INBOX = original


def test_chat_reads_inbox_without_attachment():
    exam = {"exam_id": "demo-exam", "document": "sample.csv", "questions": [
        {"id": 1, "question": "Alpha?", "choices": dict(zip("ABCD", ["Một", "Hai", "Ba", "Bốn"])),
         "correct_answer": "A", "concept": "Đại số"},
    ]}
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        (folder / "result.json").write_text(json.dumps({
            "exam_id": "demo-exam", "student": "SV001 - Nguyễn Văn An", "answers": {"1": "B"},
        }), encoding="utf-8")
        script = f'''import mcq_streamlit as app
from pathlib import Path
app.RESULTS_INBOX = Path({str(folder)!r})
class ReviewModel:
    def generate_text(self, messages, **kwargs):
        return "Nên ôn lại Đại số."
app.load_model = lambda path=None: ReviewModel()
app.main()
'''
        app = AppTest.from_string(script).run(timeout=30)
        app.session_state["messages"] = [{"role": "assistant", "kind": "exam", "id": "demo", "exam": exam}]
        app.run(timeout=30)
        app.chat_input[0].set_value("Nhận xét sinh viên Nguyễn Văn An").run(timeout=30)
        assert not app.exception, app.exception
        assert "SV001 - Nguyễn Văn An" in app.chat_message[-1].markdown[0].value
        assert "Nên ôn lại Đại số" in app.chat_message[-1].markdown[0].value
        app.chat_input[0].set_value("Nhận xét buổi kiểm tra hôm nay").run(timeout=30)
        assert not app.exception, app.exception
        assert "Buổi kiểm tra hôm nay" in app.chat_message[-1].markdown[0].value
        assert "SV001 - Nguyễn Văn An" in app.chat_message[-1].markdown[0].value


if __name__ == "__main__":
    test_app()
    test_generation_has_bounded_batches_and_validates_each_question()
    test_exam_file_and_count()
    test_stopped_exam_is_kept_in_chat()
    test_scoring_and_teacher_chat()
    test_one_chat_from_exam_to_review()
    test_result_file_enters_review_flow()
    test_world_model_student_result()
    test_ui_generation_rechecks_revision()
    test_inbox_finds_student_and_today_results()
    test_chat_reads_inbox_without_attachment()
    print("OK: số câu, prompt, giao diện, chấm bài và chat")
