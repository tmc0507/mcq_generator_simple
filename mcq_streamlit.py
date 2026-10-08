"""Generate grounded MCQs from local documents with an editable prompt."""

import json
import os
import subprocess
import sys
import tempfile
from itertools import cycle
from pathlib import Path

import streamlit as st


SOURCE = Path(__file__).resolve().parent
MODEL = SOURCE / "models/Qwen3-8B-Q4_K_M.gguf"
INPUT = SOURCE / "input"
OUTPUT = SOURCE / "output/mcq_questions.txt"
sys.path.insert(0, str(SOURCE / "scripts"))

from generate_mcq import (  # noqa: E402
    LlamaCppBackend,
    MCQ_SCHEMA,
    ask_json,
    deduplicate,
    evidence_score,
    validate_mcq,
)


DEFAULT_PROMPT = """/no_think
Bạn là người ra đề trắc nghiệm. Chỉ dùng thông tin trong SOURCE.
Hãy viết câu thứ {index} trong tổng {total} câu được yêu cầu; lần gọi này chỉ trả 1 câu mới.
- Viết bằng tiếng Việt, giữ nguyên thuật ngữ chuyên ngành.
- Có đúng 4 lựa chọn A/B/C/D và chỉ một đáp án đúng.
- Đáp án và giải thích phải được SOURCE hỗ trợ.
- evidence phải trích từ SOURCE; source_type và source_index phải đúng.
- Nếu SOURCE không đủ thông tin, trả danh sách questions rỗng.
Nguồn hợp lệ: {allowed_sources}
Các câu đã tạo, không lặp lại: {previous_questions}

SOURCE:
{source}"""
SYSTEM_PROMPT = "Chỉ dùng SOURCE để tạo MCQ; chỉ trả JSON theo schema."


@st.cache_resource
def load_model():
    return LlamaCppBackend(
        model=str(MODEL),
        n_ctx=3072,
        n_gpu_layers=0,
        threads=min(4, os.cpu_count() or 4),
    )


def extract(document):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "chunks.json"
        subprocess.run(
            [str(SOURCE / "build/doc_processor"), str(document), "--output", str(path),
             "--blocks-per-chunk", "3", "--chunk-chars", "3000"],
            check=True, capture_output=True, text=True,
        )
        return json.loads(path.read_text(encoding="utf-8"))


def render_prompt(template, chunk, index, total, previous):
    values = {
        "index": str(index),
        "total": str(total),
        "count": "1",
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


def generate_questions(llm, data, count, template, on_attempt=None):
    source_map = {(b["type"], b["index"]): b["text"] for b in data["blocks"]}
    chunks = [chunk for chunk in data["chunks"] if len(chunk["text"]) >= 200] or data["chunks"]
    if not chunks:
        return
    accepted = []
    # ponytail: 4 attempts per requested question prevent endless retries; raise if sources need deeper coverage.
    for attempt, chunk in zip(range(count * 4 + 5), cycle(chunks)):
        if len(accepted) == count:
            break
        if on_attempt:
            on_attempt(attempt + 1, len(accepted))
        prompt = render_prompt(template, chunk, len(accepted) + 1, count, accepted)
        try:
            result = ask_json(
                llm,
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                MCQ_SCHEMA,
                max_tokens=900,
                temperature=0.3,
            )
        except (ValueError, KeyError, RuntimeError):
            continue
        for question in result.get("questions", [])[:1]:
            if validate_mcq(question, source_map):
                continue
            source_key = (question["source_type"], question["source_index"])
            if source_key not in zip(chunk["source_types"], chunk["source_indices"]):
                continue
            if evidence_score(question["evidence"], source_map[source_key]) < 0.75:
                continue
            if len(deduplicate(accepted + [question])) == len(accepted):
                continue
            accepted.append(question)
            yield question, prompt


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


def main():
    st.title("Tạo câu hỏi trắc nghiệm từ tài liệu")
    documents = sorted(p for p in INPUT.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".pptx"})
    selected = st.multiselect("Tài liệu", documents, default=documents,
                              format_func=lambda p: p.name)
    count = st.number_input("Số câu cho mỗi tài liệu", min_value=1, value=1, step=1)
    template = st.text_area("Prompt của bạn", DEFAULT_PROMPT, height=320)
    st.caption("Có thể sửa toàn bộ prompt. Các biến {source}, {total}, {index}, {allowed_sources}, {previous_questions} được điền tự động.")
    st.caption("Model chạy trên CPU; mỗi câu có thể mất vài phút.")
    if st.button("Tạo câu hỏi", disabled=not selected):
        try:
            if not MODEL.is_file():
                raise FileNotFoundError(MODEL)
            llm = load_model()
            results = []
            progress = st.progress(0)
            for index, document in enumerate(selected, 1):
                with st.status(f"Đang xử lý: {document.name}"):
                    try:
                        created = 0
                        attempt_info = st.empty()
                        def show_attempt(attempt, accepted):
                            attempt_info.write(f"Lần thử {attempt} · đã tạo {accepted}/{count} câu")

                        for question, prompt in generate_questions(llm, extract(document), int(count), template, show_attempt):
                            results.append(format_question(document.name, question, prompt))
                            save_results(results)
                            created += 1
                            st.write(f"Đã tạo {created}/{count} câu")
                            progress.progress(((index - 1) * count + created) / (len(selected) * count))
                        if created < count:
                            st.warning(f"{document.name}: tạo được {created}/{count} câu có dẫn chứng hợp lệ.")
                    except Exception as exc:
                        st.error(f"{document.name}: {exc}")
                progress.progress(index / len(selected))
            if results:
                st.success(f"Đã lưu {len(results)} câu vào {OUTPUT}")
        except Exception as exc:
            st.error(str(exc))
    if OUTPUT.is_file():
        st.download_button("Tải file TXT", OUTPUT.read_bytes(), file_name=OUTPUT.name)
        st.text_area("Nội dung TXT", OUTPUT.read_text(encoding="utf-8"), height=320, disabled=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        llm = load_model()
        results = []
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        for document in sorted(p for p in INPUT.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".pptx"}):
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
