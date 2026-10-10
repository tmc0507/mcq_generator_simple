"""Generate grounded MCQs from local documents with an editable prompt."""

import json
import fcntl
from io import BytesIO
import os
import re
import subprocess
import sys
import tempfile
import uuid
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


SOURCE = Path(__file__).resolve().parent
SMALL_MODEL = SOURCE / "models/modelv1.gguf"
LARGE_MODEL = SOURCE / "models/modelv2.gguf"
MEDIUM_MODEL = SOURCE / "models/modelv3.gguf"
MODEL = next((path for path in (MEDIUM_MODEL, SMALL_MODEL, LARGE_MODEL) if path.is_file()), SMALL_MODEL)
INPUT = SOURCE / "input"
OUTPUT = SOURCE / "output/mcq_questions.txt"
EXAM = SOURCE / "output/questions.json"
EXAM_TXT = SOURCE / "output/questions.txt"
ATTEMPTS = SOURCE / "output/attempts.json"
RESULTS_INBOX = SOURCE / "output/student_results"
DOCUMENT_TYPES = {".pdf", ".docx", ".pptx", ".csv", ".doc", ".ppt"}
MODEL_OPTIONS = {"modelv1": SMALL_MODEL, "modelv2": LARGE_MODEL, "modelv3": MEDIUM_MODEL}
sys.path.insert(0, str(SOURCE / "scripts"))

from generate_mcq import (  # noqa: E402
    LlamaCppBackend,
    mcq_schema_for_chunk,
    ask_json,
    deduplicate,
    validate_mcq,
    build_verifier_prompt,
    VERIFY_SCHEMA,
    allocate_questions,
    normalize_mcq_format,
    normalize_text,
)


DEFAULT_PROMPT = """/no_think
Bạn là người ra đề trắc nghiệm. Chỉ dùng thông tin trong SOURCE.
Hãy viết tối đa {count} câu mới, bắt đầu từ câu thứ {index} trong tổng {total} câu được yêu cầu.
- Viết bằng tiếng Việt, giữ nguyên thuật ngữ chuyên ngành.
- Có đúng 4 lựa chọn A/B/C/D và chỉ một đáp án đúng.
- Mỗi câu hỏi một ý khác nhau; không lặp câu trong cùng nhóm hoặc các câu đã tạo.
- Hỏi một chi tiết cụ thể trong SOURCE hiện tại (điều kiện, công thức, thuật ngữ); không hỏi lại tên hoặc định nghĩa chung của chủ đề.
- Đáp án phải dựa trên SOURCE. Ghi concept ngắn; evidence có thể bỏ trống.
- Nếu SOURCE không đủ thông tin, tạo ít câu hơn hoặc trả danh sách questions rỗng.

SOURCE:
{source}"""
SYSTEM_PROMPT = "Chỉ dùng SOURCE để tạo MCQ; chỉ trả JSON theo schema."


@st.cache_resource
def load_model(model_path=None):
    try:
        from llama_cpp import llama_supports_gpu_offload
        gpu_layers = 20 if llama_supports_gpu_offload() else 0
    except ImportError:
        gpu_layers = 0
    return LlamaCppBackend(
        model=str(model_path or MODEL),
        n_ctx=3072,
        n_gpu_layers=gpu_layers,
        threads=min(4, os.cpu_count() or 4),
    )


def extract(document):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "chunks.json"
        result = subprocess.run(
            [str(SOURCE / "build/doc_processor"), str(document), "--output", str(path),
             "--blocks-per-chunk", "3", "--chunk-chars", "3000"],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Không đọc được tài liệu")
        return json.loads(path.read_text(encoding="utf-8"))


def render_prompt(template, chunk, index, total, previous, count=1):
    values = {
        "index": str(index),
        "total": str(total),
        "count": str(count),
        "allowed_sources": str(list(zip(chunk["source_types"], chunk["source_indices"]))),
        "previous_questions": " | ".join(q["question"] for q in previous[-8:]) or "Chưa có",
        "source": chunk["text"],
    }
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", value)
    if "{source}" not in template:
        prompt += "\n\nSOURCE:\n" + chunk["text"]
    return prompt


def generate_questions(llm, data, count, template, on_attempt=None, verify=False, stats=None):
    source_map = {(b["type"], b["index"]): b["text"] for b in data["blocks"]}
    chunks = data["chunks"]
    if not verify:
        line_counts = Counter(line.strip() for block in data["blocks"]
                              for line in set(block["text"].splitlines()) if line.strip())
        repeated_lines = {line for line, n in line_counts.items() if n >= max(3, len(data["blocks"]) // 2)}
        chunks = [{**chunk, "text": "\n".join(line for line in chunk["text"].splitlines()
                                         if line.strip() not in repeated_lines)} for chunk in chunks]
    batches = []
    for chunk, quota in zip(chunks, allocate_questions(chunks, count)):
        while quota:
            batch_size = min(3, quota)
            batches.append((chunk, batch_size))
            quota -= batch_size
    stats = stats if stats is not None else {}
    stats["batches"] = len(batches)
    stats["generated"] = 0
    accepted = []
    seen_questions = set()
    for attempt, (chunk, batch_size) in enumerate(batches, 1):
        if len(accepted) >= count:
            break
        if on_attempt:
            on_attempt(attempt, len(accepted))
        prompt = render_prompt(template, chunk, len(accepted) + 1, count, accepted, batch_size)
        schema = mcq_schema_for_chunk(chunk)
        if not verify:
            item = schema["properties"]["questions"]["items"]
            item["properties"] = {key: item["properties"][key]
                                  for key in ("question", "choices", "correct_answer", "concept", "evidence")}
            item["required"] = ["question", "choices", "correct_answer", "concept"]
        try:
            result = ask_json(
                llm,
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                schema,
                max_tokens=max(260, batch_size * 220) if not verify else max(500, batch_size * 350),
                temperature=0.3 if verify else 0.7,
            )
        except (ValueError, KeyError):
            stats["generator_error"] = stats.get("generator_error", 0) + 1
            continue
        questions = result.get("questions", [])
        if not isinstance(questions, list):
            stats["generator_error"] = stats.get("generator_error", 0) + 1
            continue
        stats["generated"] += len(questions)
        if not questions:
            stats["empty"] = stats.get("empty", 0) + 1
        for question in (questions[:batch_size] if verify else questions):
            question = normalize_mcq_format(question, chunk)
            if not verify:
                if not isinstance(question, dict):
                    question = {"question": str(question or "")}
                question["question"] = str(question.get("question") or "")
                choices = question.get("choices")
                question["choices"] = {
                    letter: str(choices.get(letter) or "") if isinstance(choices, dict) else ""
                    for letter in "ABCD"
                }
                if question.get("correct_answer") not in {"A", "B", "C", "D"}:
                    question["correct_answer"] = ""
                question.setdefault("concept", "Chưa phân loại")
                question.setdefault("difficulty", "easy")
                question.setdefault("explanation", "")
                question.setdefault("evidence", "")
                question.setdefault("source_type", chunk["source_types"][0])
                question.setdefault("source_index", chunk["source_indices"][0])
                question["concept"] = str(question["concept"] or "Chưa phân loại")
                question["verifier_status"] = "SKIPPED"
                question["evidence_score"] = None
                key = normalize_text(question["question"])
                if key in seen_questions:
                    stats["duplicate"] = stats.get("duplicate", 0) + 1
                    continue
                seen_questions.add(key)
                accepted.append(question)
                yield question, prompt
                continue
            issues = validate_mcq(question, source_map)
            if issues:
                stats["structure"] = stats.get("structure", 0) + 1
                continue
            source_key = (question["source_type"], question["source_index"])
            if source_key not in zip(chunk["source_types"], chunk["source_indices"]):
                stats["structure"] = stats.get("structure", 0) + 1
                continue
            if len(deduplicate(accepted + [question])) == len(accepted):
                stats["duplicate"] = stats.get("duplicate", 0) + 1
                continue
            if on_attempt:
                on_attempt(attempt, len(accepted))
            try:
                verdict = ask_json(
                    llm,
                    [{"role": "system", "content": "Bạn là verifier nghiêm ngặt. Chỉ dựa trên SOURCE."},
                     {"role": "user", "content": build_verifier_prompt(question, source_map[source_key])}],
                    VERIFY_SCHEMA, max_tokens=1200, temperature=0,
                )
            except ValueError:
                stats["verifier_error"] = stats.get("verifier_error", 0) + 1
                continue
            status = verdict.get("status")
            if status == "REVISE":
                revised = normalize_mcq_format(verdict.get("corrected_question"), chunk)
                if validate_mcq(revised, source_map):
                    stats["revision"] = stats.get("revision", 0) + 1
                    continue
                revised_key = (revised["source_type"], revised["source_index"])
                if revised_key not in zip(chunk["source_types"], chunk["source_indices"]):
                    stats["revision"] = stats.get("revision", 0) + 1
                    continue
                question = revised
                source_key = revised_key
            elif status != "PASS":
                stats["verifier_reject"] = stats.get("verifier_reject", 0) + 1
                continue
            question["verifier_status"] = status
            question["evidence_score"] = None
            if len(deduplicate(accepted + [question])) == len(accepted):
                stats["duplicate"] = stats.get("duplicate", 0) + 1
                continue
            accepted.append(question)
            yield question, prompt
    stats["accepted"] = len(accepted)


def format_question(document, question, prompt):
    choices = question["choices"]
    return "\n".join([
        f"TÀI LIỆU: {document}",
        "PROMPT ĐÃ GỬI MODEL:",
        f"[system] {SYSTEM_PROMPT}",
        "[user]", prompt,
        "",
        f"Câu hỏi: {question['question']}",
        *(f"{key}. {choices[key]}" for key in "ABCD"),
        f"Đáp án: {question['correct_answer']}",
        f"Giải thích: {question['explanation']}",
        f"Dẫn chứng: {question['evidence']}",
        f"Nguồn: {question['source_type']} {question['source_index']}",
    ])


def save_results(results):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n\n".join(results) + "\n", encoding="utf-8")


def question_count(prompt):
    match = re.search(r"(?<!\d)(\d+)\s*(?:câu(?: hỏi)?|questions?)\b", prompt, re.IGNORECASE)
    count = int(match.group(1)) if match else 10
    if not 1 <= count <= 50:
        raise ValueError("Số câu phải từ 1 đến 50 (ví dụ: Tạo 10 câu hỏi).")
    return count


def format_exam_txt(exam):
    lines = [f"BỘ ĐỀ: {exam['document']}", f"Số câu: {len(exam['questions'])}", ""]
    if exam.get("verification_enabled") is False:
        lines.insert(1, "BẢN NHÁP: Chưa kiểm tra đáp án, dẫn chứng hoặc câu trùng; giảng viên cần rà soát.")
    for position, question in enumerate(exam["questions"], 1):
        lines.append(f"Câu {position}. {question['question']}")
        lines.extend(f"{letter}. {question['choices'][letter]}" for letter in "ABCD")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def create_exam(uploaded, model_path, count, template, on_attempt=None, verify=False, on_question=None):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    document_name = Path(uploaded.name).name
    if Path(document_name).suffix.lower() not in DOCUMENT_TYPES:
        raise ValueError("Chỉ hỗ trợ PDF, DOC/DOCX, PPT/PPTX và CSV.")
    with tempfile.TemporaryDirectory() as directory:
        document = Path(directory) / document_name
        document.write_bytes(uploaded.getvalue())
        data = extract(document)
    llm = load_model(model_path)
    questions = []
    results = []
    stats = {}
    exam = {
        "exam_id": uuid.uuid4().hex,
        "document": document_name, "backend": "llama_cpp", "model": str(model_path),
        "requested_questions": count, "accepted_questions": 0,
        "generation_stats": stats, "verification_enabled": verify, "questions": questions,
    }
    write_json(EXAM, exam)
    for question, prompt in generate_questions(llm, data, count, template, on_attempt, verify=verify, stats=stats):
        question["id"] = len(questions) + 1
        question["source"] = {"type": question["source_type"], "index": question["source_index"]}
        questions.append(question)
        results.append(format_question(document_name, question, prompt))
        exam["accepted_questions"] = len(questions)
        write_json(EXAM, exam)
        if on_question:
            on_question(exam)
    if not questions:
        raise ValueError("Model chưa trả về câu hỏi nào; thử tài liệu có nhiều nội dung hơn.")
    write_json(EXAM, exam)
    save_results(results)
    EXAM_TXT.parent.mkdir(parents=True, exist_ok=True)
    EXAM_TXT.write_text(format_exam_txt(exam), encoding="utf-8")
    return exam


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_exam():
    if not EXAM.is_file():
        return None
    exam = json.loads(EXAM.read_text(encoding="utf-8"))
    if not isinstance(exam.get("questions"), list):
        raise ValueError("output/questions.json không có danh sách questions")
    for question in exam["questions"]:
        if (not isinstance(question, dict) or not isinstance(question.get("question"), str)
                or not isinstance(question.get("choices"), dict)
                or set(question["choices"]) != set("ABCD")
                or question.get("correct_answer") not in ("", "A", "B", "C", "D")):
            raise ValueError("Đề có câu hỏi không hợp lệ")
    return exam


def score_attempt(questions, answers):
    by_concept = {}
    details = []
    correct = 0
    for position, question in enumerate(questions, 1):
        answer = answers.get(str(question.get("id", position)), "")
        is_correct = bool(question["correct_answer"]) and answer == question["correct_answer"]
        correct += is_correct
        concept = question.get("concept") or "Chưa phân loại"
        totals = by_concept.setdefault(concept, {"correct": 0, "total": 0})
        totals["correct"] += is_correct
        totals["total"] += 1
        details.append({
            "id": question.get("id", position), "concept": concept,
            "question": question["question"], "student_answer": answer or "Bỏ trống",
            "correct_answer": question["correct_answer"], "correct": is_correct,
            "student_choice": question.get("choices", {}).get(answer, ""),
            "correct_choice": question.get("choices", {}).get(question["correct_answer"], ""),
            "explanation": question.get("explanation", ""),
        })
    total = len(questions)
    return {
        "correct": correct, "total": total,
        "percent": round(correct * 100 / total, 1) if total else 0,
        "by_concept": by_concept, "details": details,
    }


def import_result(uploaded, exam):
    try:
        data = json.loads(uploaded.getvalue().decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("File kết quả phải là JSON UTF-8 hợp lệ.") from exc
    if not isinstance(data, dict):
        raise ValueError("File kết quả phải là một JSON object.")
    if data.get("exam_id") != exam.get("exam_id") or not exam.get("exam_id"):
        raise ValueError("exam_id không khớp với bộ đề đang mở.")
    if not exam.get("questions"):
        raise ValueError("Bộ đề chưa có câu hỏi để chấm.")
    if any(question.get("correct_answer") not in {"A", "B", "C", "D"} or
           any(not question["choices"].get(letter) for letter in "ABCD")
           for question in exam["questions"]):
        raise ValueError("Bộ đề có câu thiếu đáp án hoặc lựa chọn; giảng viên cần rà soát trước khi chấm.")
    student = data.get("student")
    answers = data.get("answers")
    if not isinstance(student, str) or not student.strip():
        raise ValueError("Kết quả thiếu tên hoặc mã sinh viên.")
    valid_ids = {str(question["id"]) for question in exam["questions"]}
    if not isinstance(answers, dict) or any(
        key not in valid_ids or value not in ("", "A", "B", "C", "D")
        for key, value in answers.items()
    ):
        raise ValueError("answers phải chứa ID câu hỏi và đáp án A/B/C/D hoặc chuỗi rỗng.")
    return {
        "id": uuid.uuid4().hex, "exam_id": exam["exam_id"],
        "student": student.strip(),
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document": exam["document"], "report": score_attempt(exam["questions"], answers),
    }


def read_attempts():
    if not ATTEMPTS.is_file():
        return []
    attempts = json.loads(ATTEMPTS.read_text(encoding="utf-8"))
    if not isinstance(attempts, list):
        raise ValueError("output/attempts.json không hợp lệ")
    return attempts


def save_attempt(attempt):
    ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
    with (ATTEMPTS.parent / ".attempts.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        attempts = read_attempts()
        attempts.append(attempt)
        write_json(ATTEMPTS, attempts)


def search_key(value):
    value = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
    return " ".join("".join(char for char in value if not unicodedata.combining(char)).split())


def inbox_attempts(exam, today=False):
    RESULTS_INBOX.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    latest = {}
    skipped = 0
    for path in RESULTS_INBOX.glob("*.json"):
        try:
            received = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
            if today and received.date() != now.date():
                continue
            attempt = import_result(BytesIO(path.read_bytes()), exam)
        except (OSError, ValueError):
            skipped += 1
            continue
        key = search_key(attempt["student"])
        if key not in latest or received > latest[key]["received"]:
            latest[key] = {"attempt": attempt, "received": received}
    return sorted(latest.values(), key=lambda item: item["received"]), skipped


def student_matches(prompt, student):
    text = search_key(prompt)
    aliases = [student, *re.split(r"\s+-\s+", student)]
    return any(len(alias.strip()) >= 3 and re.search(
        rf"(?<!\w){re.escape(search_key(alias))}(?!\w)", text)
        for alias in aliases)


def study_suggestion(llm, question, weak_topics):
    raw = llm.generate_text(
        [{"role": "system", "content":
          "/no_think\nChỉ viết một câu bắt đầu bằng 'Nên ôn lại', gợi ý cách ôn các chủ đề được nêu. "
          "Không nhận xét sinh viên, câu hỏi, đáp án hay điểm số."},
         {"role": "user", "content":
          f"Chủ đề cần ôn: {', '.join(weak_topics) if weak_topics else 'Không có câu sai'}.\n"
          f"Câu hỏi của giảng viên: {question}"}],
        max_new_tokens=100, temperature=0, do_sample=False,
    )
    suggestion = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "<think>" in suggestion or not suggestion:
        raise ValueError("Model chưa trả lời xong; hãy thử lại với câu hỏi ngắn hơn.")
    # shortcut: use a factual generic tip when the small model ignores the narrow format; revisit with stronger feedback evaluation.
    if (not suggestion.startswith("Nên ôn lại") or
            any(word in suggestion.lower() for word in ("sinh viên", "trả lời", "đáp án", "điểm số", "hạn chế"))):
        suggestion = (f"Nên ôn lại {', '.join(weak_topics)} qua bài giảng và câu hỏi luyện tập."
                      if weak_topics else "Tiếp tục luyện tập để duy trì kết quả.")
    return suggestion


def teacher_reply(llm, question, report):
    all_misses = [item for item in report["details"] if not item["correct"]]
    strengths = [f"- {concept}: đúng {score['correct']}/{score['total']} câu."
                 for concept, score in report["by_concept"].items() if score["correct"]]
    weaknesses = [f"- Câu {item['id']} ({item['concept']}): " +
                  ("bỏ trống." if item["student_answer"] == "Bỏ trống" else
                   f"trả lời {item['student_answer']}, đáp án {item['correct_answer']}.")
                  for item in all_misses]
    weak_topics = list(dict.fromkeys(item["concept"] for item in all_misses))
    suggestion = study_suggestion(llm, question, weak_topics)
    return "\n".join([
        "**Điểm mạnh**", *(strengths or ["- Chưa có câu trả lời đúng."]),
        "**Cần cải thiện**", *(weaknesses or ["- Không có câu trả lời sai hoặc bỏ trống."]),
        "**Gợi ý ôn tập**", suggestion,
    ])


def class_reply(llm, question, records, skipped=0):
    now = datetime.now().astimezone()
    scores = [item["attempt"]["report"]["percent"] for item in records]
    lines = [f"**Buổi kiểm tra hôm nay ({now:%d/%m/%Y})**",
             f"Đã nhận {len(records)} sinh viên từ {records[0]['received']:%H:%M} đến {now:%H:%M}. "
             f"Điểm trung bình: {sum(scores) / len(scores):.1f}%."]
    if skipped:
        lines.append(f"Bỏ qua {skipped} file không hợp lệ hoặc không thuộc bộ đề đang mở.")
    weak_topics = Counter()
    for item in records:
        attempt = item["attempt"]
        report = attempt["report"]
        strong = [name for name, score in report["by_concept"].items() if score["correct"]]
        weak = [name for name, score in report["by_concept"].items() if score["correct"] < score["total"]]
        weak_topics.update(weak)
        lines.extend([
            f"**{attempt['student']}** — {report['correct']}/{report['total']} ({report['percent']}%).",
            f"- Điểm mạnh: {', '.join(strong) or 'Chưa có chủ đề trả lời đúng'}.",
            f"- Cần cải thiện: {', '.join(weak) or 'Không có câu sai'}.",
        ])
    lines.extend(["**Gợi ý chung**", study_suggestion(
        llm, question, [name for name, _ in weak_topics.most_common(8)])])
    return "\n".join(lines)


def render_exam_message(message):
    exam = message["exam"]
    with st.chat_message("assistant", avatar="🎓"):
        st.write(f"Đã tạo **{len(exam['questions'])} câu hỏi** từ **{exam['document']}**. Tải JSON để đưa sang trang web làm bài.")
        if exam.get("verification_enabled") is False:
            st.info("Bản nháp nhanh: giữ nguyên dẫn chứng, bỏ câu hỏi trùng nguyên văn; chưa kiểm tra đáp án. Giảng viên nên rà soát trước khi giao bài.")
        if len(exam["questions"]) < exam.get("requested_questions", len(exam["questions"])):
            stats = exam.get("generation_stats", {})
            if exam.get("verification_enabled") is False:
                st.warning(
                    f"Model trả {len(exam['questions'])}/{exam['requested_questions']} câu "
                    f"sau {stats.get('batches', '?')} nhóm tạo đề; "
                    f"bỏ {stats.get('duplicate', 0)} câu trùng nguyên văn."
                )
            else:
                st.warning(
                    f"Chỉ giữ {len(exam['questions'])}/{exam['requested_questions']} câu dùng được "
                    f"sau {stats.get('batches', '?')} nhóm tạo đề. "
                    "Model không tạo đủ số câu được yêu cầu."
                )
                st.caption(
                    f"Model tạo {stats.get('generated', '?')} câu; loại do cấu trúc: {stats.get('structure', 0)}, "
                    f"kiểm tra nội dung: "
                    f"{stats.get('verifier_reject', 0) + stats.get('verifier_error', 0) + stats.get('revision', 0)}, "
                    f"trùng lặp: {stats.get('duplicate', 0)}."
                )
        st.download_button(
            "⬇ Tải dữ liệu đề JSON", json.dumps(exam, ensure_ascii=False, indent=2),
            file_name=f"{Path(exam['document']).stem}_questions.json", mime="application/json",
            key="download_json_" + message["id"],
        )
        st.download_button(
            "⬇ Tải bộ đề TXT", format_exam_txt(exam),
            file_name=f"{Path(exam['document']).stem}_questions.txt", mime="text/plain",
            key="download_txt_" + message["id"],
        )


def chat_view(model_path, template, verify=False):
    st.markdown('<div class="chat-hero"><div class="eyebrow">TRỢ LÝ HỌC TẬP</div><h1>Tạo đề và nhận xét trong một cuộc trò chuyện</h1><p>Đính kèm tài liệu để tạo đề; đặt file kết quả vào output/student_results rồi hỏi nhận xét.</p></div>', unsafe_allow_html=True)
    messages = st.session_state.setdefault("messages", [])
    stopped = st.session_state.pop("generating_exam", False)
    partial = st.session_state.pop("partial_exam", None)
    if partial:
        partial["generation_stats"]["accepted"] = len(partial["questions"])
        partial["stopped"] = True
        write_json(EXAM, partial)
        EXAM_TXT.parent.mkdir(parents=True, exist_ok=True)
        EXAM_TXT.write_text(format_exam_txt(partial), encoding="utf-8")
        messages.append({"role": "assistant", "content":
                         f"Đã dừng tạo đề. Giữ lại {len(partial['questions'])} câu đã tạo."})
        exam_id = uuid.uuid4().hex
        messages.append({"role": "assistant", "kind": "exam", "id": exam_id, "exam": partial})
    elif stopped:
        messages.append({"role": "assistant", "content": "Đã dừng tạo đề trước khi có câu hỏi hợp lệ."})
    if not messages:
        with st.chat_message("assistant", avatar="🎓"):
            st.write("Xin chào! Bấm **+** để chọn tài liệu và tạo đề JSON. Đặt kết quả từ trang làm bài vào `output/student_results/`, rồi hỏi nhận xét theo tên, mã sinh viên hoặc buổi kiểm tra hôm nay.")
    for message in messages:
        if message.get("kind") == "exam":
            render_exam_message(message)
        else:
            with st.chat_message(message["role"], avatar="🎓" if message["role"] == "assistant" else "👤"):
                st.write(message["content"])
    submission = st.chat_input(
        "Đính kèm tài liệu hoặc kết quả JSON bằng dấu +...",
        accept_file=True, file_type=["pdf", "doc", "docx", "ppt", "pptx", "csv", "json"],
        submit_mode="stop",
    )
    if submission is None:
        return
    prompt = (submission.text or "").strip()
    files = submission.files
    if not prompt and not files:
        return
    is_result_file = bool(files) and Path(files[0].name).suffix.lower() == ".json"
    content = prompt or ("Nhận kết quả bài làm" if is_result_file else "Tạo bộ đề từ tài liệu")
    if files:
        content += f"\n\n📎 {files[0].name}"
    messages.append({"role": "user", "content": content})
    with st.chat_message("user", avatar="👤"):
        st.write(content)
    if is_result_file:
        try:
            exam = next((item["exam"] for item in reversed(messages)
                         if item.get("kind") == "exam"), None) or read_exam()
            if exam is None:
                raise ValueError("Chưa có bộ đề để đối chiếu kết quả.")
            attempt = import_result(files[0], exam)
            save_attempt(attempt)
            st.session_state["active_attempt"] = attempt
            report = attempt["report"]
            messages.append({"role": "assistant", "content":
                             f"Đã nhận kết quả của **{attempt['student']}**: "
                             f"{report['correct']}/{report['total']} ({report['percent']}%). "
                             "Hãy hỏi tôi nhận xét bài làm này."})
        except Exception as exc:
            messages.append({"role": "assistant", "content": f"Không nhận được kết quả: {exc}"})
    elif files:
        try:
            count = question_count(prompt)
            st.session_state["generating_exam"] = True
            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner(f"Đang đọc tài liệu và tạo tối đa {count} câu hỏi..."):
                    progress = st.empty()
                    def remember_question(partial_exam):
                        st.session_state["partial_exam"] = partial_exam
                        progress.write(f"Đã giữ {len(partial_exam['questions'])}/{count} câu")

                    exam = create_exam(
                        files[0], model_path, count, template,
                        lambda attempt, accepted: progress.write(
                            f"Đang xử lý nhóm {attempt} · đã giữ {accepted}/{count} câu"),
                        verify=verify,
                        on_question=remember_question,
                    )
                    st.session_state.pop("partial_exam", None)
                    st.session_state.pop("generating_exam", None)
                    progress.empty()
            exam_id = uuid.uuid4().hex
            messages.append({"role": "assistant", "kind": "exam", "id": exam_id, "exam": exam})
            st.session_state.pop("active_attempt", None)
        except Exception as exc:
            st.session_state.pop("generating_exam", None)
            messages.append({"role": "assistant", "content": f"Không tạo được đề: {exc}"})
    else:
        try:
            exam = next((item["exam"] for item in reversed(messages)
                         if item.get("kind") == "exam"), None) or read_exam()
            today = "hom nay" in search_key(prompt)
            records, skipped = inbox_attempts(exam, today=today) if exam else ([], 0)
            matches = [item for item in records if student_matches(prompt, item["attempt"]["student"])]
            if len(matches) > 1:
                names = ", ".join(item["attempt"]["student"] for item in matches)
                answer = f"Có nhiều sinh viên khớp yêu cầu: {names}. Hãy ghi mã sinh viên cụ thể."
            elif matches:
                attempt = matches[0]["attempt"]
                with st.chat_message("assistant", avatar="🎓"):
                    with st.spinner(f"Đang nhận xét {attempt['student']}..."):
                        answer = f"**{attempt['student']}** — " + teacher_reply(
                            load_model(model_path), prompt, attempt["report"])
            elif today:
                if records:
                    with st.chat_message("assistant", avatar="🎓"):
                        with st.spinner(f"Đang nhận xét {len(records)} sinh viên..."):
                            answer = class_reply(load_model(model_path), prompt, records, skipped)
                else:
                    answer = ("Chưa có kết quả hợp lệ của bộ đề đang mở được ghi vào "
                              f"`{RESULTS_INBOX.relative_to(SOURCE)}/` hôm nay.")
            elif "active_attempt" in st.session_state:
                with st.chat_message("assistant", avatar="🎓"):
                    with st.spinner("Đang nhận xét bài làm..."):
                        answer = teacher_reply(load_model(model_path), prompt,
                                               st.session_state["active_attempt"]["report"])
            elif exam:
                answer = (f"Chưa tìm thấy sinh viên trong `{RESULTS_INBOX.relative_to(SOURCE)}/`. "
                          "Hãy ghi tên hoặc mã sinh viên có trong file kết quả, ví dụ ‘Nhận xét sinh viên SV001’, "
                          "hoặc hỏi ‘Nhận xét buổi kiểm tra hôm nay’.")
            else:
                answer = "Chưa có bộ đề để đối chiếu kết quả. Hãy tạo đề hoặc mở `output/questions.json` trước."
            messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            messages.append({"role": "assistant", "content": f"Chưa thể nhận xét: {exc}"})
    st.rerun()


def main():
    st.set_page_config(page_title="MCQ Studio", page_icon="🎓", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 920px; padding-top: 2rem; padding-bottom: 7rem;}
        [data-testid="stChatMessage"] {border-radius: 16px; padding: 1rem 1.25rem;}
        .chat-hero {background: linear-gradient(120deg, #182a48, #385d7a); color: #fff;
            border-radius: 22px; padding: 2rem 2.4rem; margin-bottom: 1.5rem;}
        .chat-hero h1 {color: #fff; margin: .4rem 0; font-size: 2rem;}
        .chat-hero p {color: #e2ebf4; margin: 0;}
        .eyebrow {font-size: .75rem; font-weight: 700; letter-spacing: .16em; color: #8fe1d5;}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.title("🎓 MCQ Studio")
        st.caption("Tài liệu → đề JSON → kết quả từ web khác → nhận xét")
        if st.button("+ Cuộc trò chuyện mới", use_container_width=True):
            for key in ("messages", "active_attempt", "partial_exam", "generating_exam"):
                st.session_state.pop(key, None)
            st.rerun()
        model_alias = st.selectbox("Model", list(MODEL_OPTIONS), index=list(MODEL_OPTIONS.values()).index(MODEL))
        model_path = MODEL_OPTIONS[model_alias]
        st.caption("modelv1: 1.7B, dễ lặp · modelv2: 8B, chậm trên CPU · modelv3: 4B, khuyên dùng")
        with st.expander("Tùy chỉnh prompt tạo đề"):
            template = st.text_area("Prompt tạo đề", DEFAULT_PROMPT, height=240)
        verify = st.toggle("Kiểm tra đáp án bằng AI", value=False,
                           help="Mặc định tắt để tạo nhanh; bật nếu muốn model kiểm tra sâu từng câu.")
        st.caption("Nút Dừng xuất hiện khi đang tạo đề; lượt suy luận hiện tại có thể cần chạy xong trước khi dừng.")
        st.caption("Chạy local · dữ liệu lưu trong output/ · chưa có đăng nhập")
    previous_alias = st.session_state.get("loaded_model_alias")
    if previous_alias and previous_alias != model_alias:
        load_model.clear()
    st.session_state["loaded_model_alias"] = model_alias
    chat_view(model_path, template, verify)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        llm = load_model()
        results = []
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        for document in sorted(p for p in INPUT.iterdir() if p.suffix.lower() in DOCUMENT_TYPES):
            print(f"Đang xử lý {document.name}...", flush=True)
            try:
                for question, prompt in generate_questions(llm, extract(document), count, DEFAULT_PROMPT):
                    results.append(format_question(document.name, question, prompt))
                    save_results(results)
                    print(f"Đã lưu {len(results)} câu vào {OUTPUT}", flush=True)
            except Exception as exc:
                print(f"Lỗi {document.name}: {exc}", flush=True)
    else:
        main()
