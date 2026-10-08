#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "choices": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                        },
                        "required": ["A", "B", "C", "D"],
                    },
                    "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "concept": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "explanation": {"type": "string"},
                    "evidence": {"type": "string"},
                    "source_index": {"type": "integer"},
                    "source_type": {"type": "string", "enum": ["page", "slide", "section"]},
                },
                "required": [
                    "question", "choices", "correct_answer", "concept", "difficulty",
                    "explanation", "evidence", "source_index", "source_type"
                ],
            },
        }
    },
    "required": ["questions"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["PASS", "REVISE", "REJECT"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "corrected_question": {
            "anyOf": [
                {"type": "null"},
                MCQ_SCHEMA["properties"]["questions"]["items"],
            ]
        },
    },
    "required": ["status", "issues", "corrected_question"],
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("\u00ad", "")
    text = text.replace("￾", "-")
    for dash in ("‐", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"-\s+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evidence_score(evidence: str, source: str) -> float:
    ev = normalize_text(evidence)
    src = normalize_text(source)
    if not ev or not src:
        return 0.0
    if ev in src:
        return 1.0

    ev_words = ev.split()
    src_words = src.split()
    if not ev_words or not src_words:
        return 0.0

    window = len(ev_words)
    if len(src_words) <= window:
        return SequenceMatcher(None, ev, src).ratio()

    best = 0.0
    # Chỉ so các cửa sổ cùng kích thước evidence; đủ cho lỗi xuống dòng/hyphen phổ biến.
    for i in range(len(src_words) - window + 1):
        candidate = " ".join(src_words[i:i + window])
        best = max(best, SequenceMatcher(None, ev, candidate).ratio())
    return best


def validate_mcq(q, valid_sources):
    if not isinstance(q, dict):
        return ["MCQ phải là object"]
    issues = []
    required = [
        "question", "choices", "correct_answer", "concept", "difficulty",
        "explanation", "evidence", "source_index", "source_type"
    ]
    for key in required:
        if key not in q:
            issues.append(f"Thiếu field: {key}")

    choices = q.get("choices")
    if not isinstance(choices, dict) or set(choices.keys()) != {"A", "B", "C", "D"}:
        issues.append("choices phải có đúng A/B/C/D")
    elif any(not isinstance(choices[key], str) or not choices[key].strip() for key in "ABCD"):
        issues.append("choices không được rỗng")
    if not isinstance(q.get("correct_answer"), str) or q["correct_answer"] not in {"A", "B", "C", "D"}:
        issues.append("correct_answer phải là A/B/C/D")
    if not isinstance(q.get("difficulty"), str) or q["difficulty"] not in {"easy", "medium", "hard"}:
        issues.append("difficulty không hợp lệ")
    if type(q.get("source_index")) is not int or q["source_index"] < 1:
        issues.append("source_index phải là integer")
    if not isinstance(q.get("source_type"), str) or q["source_type"] not in {"page", "slide", "section"}:
        issues.append("source_type không hợp lệ")
    if not any(issue.startswith("source_") for issue in issues) and (q["source_type"], q["source_index"]) not in valid_sources:
        issues.append("source không tồn tại trong tài liệu")
    if not isinstance(q.get("question"), str) or not q["question"].strip():
        issues.append("question rỗng")
    if not isinstance(q.get("evidence"), str) or not q["evidence"].strip():
        issues.append("evidence rỗng")
    if not isinstance(q.get("explanation"), str) or not q["explanation"].strip():
        issues.append("explanation rỗng")
    if not isinstance(q.get("concept"), str) or not q["concept"].strip():
        issues.append("concept rỗng")
    return issues


def parse_json_text(raw):
    if not isinstance(raw, str):
        raise ValueError("Model không trả về text")
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model không trả về JSON object")
    result = json.loads(raw[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Model không trả về JSON object")
    return result


def allocate_questions(chunks, total):
    if total <= 0:
        return [0] * len(chunks)
    lengths = [len(c.get("text", "").strip()) for c in chunks]
    lengths = [length if length >= 200 else 0 for length in lengths]
    total_chars = sum(lengths)
    if not total_chars:
        return [0] * len(chunks)
    raw = [total * x / total_chars for x in lengths]
    counts = [int(x) for x in raw]
    remain = total - sum(counts)
    order = sorted(range(len(chunks)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remain]:
        counts[i] += 1
    return counts


def build_generator_prompt(chunk, count):
    allowed = list(zip(chunk["source_types"], chunk["source_indices"]))
    return f"""
/no_think
Bạn là AI chuyên tạo câu hỏi trắc nghiệm từ tài liệu giáo dục.

QUY TẮC TUYỆT ĐỐI:
- CHỈ dùng thông tin có trong SOURCE bên dưới.
- Không dùng kiến thức có sẵn của model nếu SOURCE không chứa nó.
- Không suy diễn vượt SOURCE.
- Mỗi câu có đúng 4 lựa chọn A/B/C/D và đúng 1 đáp án đúng.
- Distractor phải hợp lý nhưng phải sai theo SOURCE.
- evidence phải là đoạn text thực sự xuất hiện trong SOURCE.
- source_index và source_type phải trỏ đúng nơi chứa evidence.
- Nếu SOURCE không đủ tạo {count} câu tốt, hãy tạo ít hơn. Không hallucinate để đủ số lượng.

DIFFICULTY:
- easy: định nghĩa, nhận biết, fact trực tiếp.
- medium: so sánh, phân biệt, hiểu quan hệ, kết hợp thông tin.
- hard: áp dụng công thức, tính toán, suy luận nhiều bước, phân tích tình huống.

Nguồn hợp lệ trong chunk: {allowed}
Số câu tối đa cần tạo: {count}

SOURCE:
{chunk['text']}
""".strip()


def build_verifier_prompt(q, source_text):
    return f"""
/no_think
Bạn là verifier nghiêm ngặt cho câu hỏi trắc nghiệm giáo dục.
CHỈ dùng SOURCE. Không dùng kiến thức bên ngoài.

QUESTION:
{json.dumps(q, ensure_ascii=False, indent=2)}

Kiểm tra:
1. Question có được SOURCE hỗ trợ không?
2. correct_answer có đúng theo SOURCE không?
3. Có đúng một đáp án đúng không?
4. explanation có thêm kiến thức ngoài SOURCE không?
5. evidence có thật sự hỗ trợ đáp án không?
6. distractor có hợp lý không?
7. concept có phù hợp không?
8. difficulty có đúng rubric không?

Rubric difficulty:
- easy: định nghĩa, nhận biết, fact trực tiếp.
- medium: so sánh, phân biệt, hiểu quan hệ.
- hard: công thức, tính toán, suy luận nhiều bước, tình huống.

Nếu hoàn toàn ổn: PASS.
Nếu sửa được chỉ dựa trên SOURCE: REVISE và trả corrected_question đầy đủ.
Nếu SOURCE không đủ chứng minh: REJECT.
Chỉ trả JSON đúng schema, không Markdown.

SOURCE:
{source_text}
""".strip()


class LlamaCppBackend:
    def __init__(self, model, n_ctx=8192, n_gpu_layers=20, threads=None, debug=False, seed=42):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Thiếu llama-cpp-python. Cài theo README.md trước khi chạy.") from exc
        self.llm = Llama(
            model_path=str(model), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            n_threads=threads or max(1, (os.cpu_count() or 4) // 2),
            verbose=debug, seed=seed,
        )

    def generate_text(self, messages, max_new_tokens, temperature, do_sample, schema=None):
        response = self.llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema} if schema else None,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0,
            top_p=0.8, top_k=20, min_p=0.0, repeat_penalty=1.05,
        )
        return response["choices"][0]["message"]["content"]


class TransformersBackend:
    def __init__(self, model_name, max_input_tokens=8192, device_map="0", seed=42):
        try:
            import numpy as np
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Thiếu dependency Kaggle. Chạy kaggle/setup.sh trước.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Transformers 4-bit NF4 cần CUDA GPU; hãy bật GPU trong Kaggle Settings.")
        if model_name.startswith(("/", ".")) and not Path(model_name).exists():
            raise FileNotFoundError(f"Không tìm thấy model local: {model_name}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print("Backend       : transformers")
        print(f"Model         : {model_name}")
        print("CUDA          : available")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
        print("Quantization  : 4-bit NF4")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization, torch_dtype=torch.float16,
                device_map="auto" if device_map == "auto" else {"": 0},
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CUDA out of memory while loading Qwen; use a GPU with more VRAM.") from exc
            raise RuntimeError(
                f"Cannot load model {model_name}: {exc}. "
                "It may be unavailable locally while Kaggle Internet is disabled; "
                "enable Internet or pass --model /kaggle/input/.../Qwen3-8B."
            ) from exc
        self.model.eval()
        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.input_device = self.model.get_input_embeddings().weight.device

    def generate_text(self, messages, max_new_tokens, temperature, do_sample, schema=None):
        context_limit = getattr(self.model.config, "max_position_embeddings", None) or 32768
        input_limit = min(self.max_input_tokens, context_limit - max_new_tokens)
        if input_limit < 256:
            raise ValueError("Context không đủ cho max_new_tokens; giảm --max-new-tokens.")

        def tokenize(chat):
            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            return self.tokenizer(prompt, return_tensors="pt")

        inputs = tokenize(messages)
        if inputs["input_ids"].shape[-1] > input_limit:
            source_message = next(
                (i for i in range(len(messages) - 1, -1, -1)
                 if messages[i]["role"] == "user" and "\nSOURCE:\n" in messages[i]["content"]), None,
            )
            if source_message is None:
                raise ValueError("Prompt quá dài; giảm --chunk-chars khi trích tài liệu.")
            prefix, marker, source = messages[source_message]["content"].rpartition("\nSOURCE:\n")
            if not marker:
                raise ValueError("Prompt quá dài; giảm --chunk-chars khi trích tài liệu.")
            trimmed = [dict(message) for message in messages]
            trimmed[source_message]["content"] = prefix + marker
            overhead = tokenize(trimmed)["input_ids"].shape[-1]
            source_tokens = self.tokenizer.encode(source, add_special_tokens=False)
            available = input_limit - overhead - 8
            if available < 100:
                raise ValueError("Prompt quá dài, không còn chỗ cho SOURCE; giảm --chunk-chars.")
            print(f"Warning: SOURCE vượt {input_limit} input tokens; cắt còn {available} tokens.")
            trimmed[source_message]["content"] += self.tokenizer.decode(source_tokens[:available])
            inputs = tokenize(trimmed)
            if inputs["input_ids"].shape[-1] > input_limit:
                raise ValueError("Prompt vẫn vượt context; giảm --chunk-chars.")

        input_length = inputs["input_ids"].shape[-1]
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        options = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
        }
        if do_sample:
            options.update(temperature=temperature, top_p=0.8, top_k=20)
        try:
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, **options)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory. Try smaller chunk size or lower --max-new-tokens."
                ) from exc
            raise
        result = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        del outputs, inputs
        return result


def ask_json(backend, messages, schema, max_tokens, temperature, debug_path=None):
    if debug_path:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.with_name(debug_path.name + "_prompt.txt").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    chat = list(messages)
    for attempt in range(3):
        raw = backend.generate_text(
            chat, max_new_tokens=max_tokens, temperature=temperature,
            do_sample=temperature > 0, schema=schema,
        )
        if debug_path:
            debug_path.with_name(debug_path.name + "_raw.txt").write_text(raw, encoding="utf-8")
        try:
            result = parse_json_text(raw)
        except ValueError:
            if attempt == 2:
                raise ValueError("Model trả JSON không hợp lệ sau 3 lần thử")
            chat = messages + [
                {"role": "assistant", "content": raw[:2000]},
                {"role": "user", "content": "The previous output was invalid JSON. Return the same result as valid JSON only."},
            ]
            continue
        if debug_path:
            debug_path.with_name(debug_path.name + "_parsed.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        return result


def deduplicate(items, threshold=0.92):
    kept = []
    normalized = []
    for item in items:
        key = normalize_text(item["question"])
        duplicate = False
        for old in normalized:
            if key == old or SequenceMatcher(None, key, old).ratio() >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
            normalized.append(key)
    return kept


def validate_chunks(data):
    if not isinstance(data, dict) or not isinstance(data.get("blocks"), list) or not isinstance(data.get("chunks"), list):
        raise ValueError("chunks.json phải chứa blocks và chunks dạng danh sách")
    source_map = {}
    for block in data["blocks"]:
        if (not isinstance(block, dict) or type(block.get("index")) is not int
                or block["index"] < 1 or block.get("type") not in ("page", "slide", "section")
                or not isinstance(block.get("text"), str)):
            raise ValueError("Block trong chunks.json không hợp lệ")
        source_map[(block["type"], block["index"])] = block["text"]
    for chunk in data["chunks"]:
        if (not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str)
                or not isinstance(chunk.get("source_types"), list)
                or not isinstance(chunk.get("source_indices"), list)
                or not chunk["source_types"]
                or len(chunk["source_types"]) != len(chunk["source_indices"])):
            raise ValueError("Chunk trong chunks.json không hợp lệ")
        try:
            if any((kind, index) not in source_map for kind, index in zip(chunk["source_types"], chunk["source_indices"])):
                raise ValueError("Chunk trỏ tới source không tồn tại")
        except TypeError as exc:
            raise ValueError("Source reference trong chunk không hợp lệ") from exc
    return source_map


def main():
    parser = argparse.ArgumentParser(description="Sinh MCQ bằng GGUF local hoặc Transformers trên Kaggle")
    parser.add_argument("--backend", choices=("llama_cpp", "transformers"), default="llama_cpp")
    parser.add_argument("--chunks", default="work/chunks.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--output", default="output/questions.json")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=20)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--device-map", choices=("0", "auto"), default="0")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--skip-verifier", action="store_true", help="Chỉ dùng khi debug")
    parser.add_argument("--evidence-threshold", type=float, default=0.75)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.questions < 1 or args.max_new_tokens < 1 or args.max_input_tokens < 256:
        parser.error("--questions, --max-new-tokens phải dương và --max-input-tokens >= 256")
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks phải >= 1")

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"Không tìm thấy chunks: {chunks_path}")
    if args.backend == "llama_cpp" and not Path(args.model).is_file():
        raise SystemExit(f"Không tìm thấy model GGUF: {args.model}")
    try:
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        source_map = validate_chunks(data)
    except ValueError as exc:
        raise SystemExit(f"chunks.json không hợp lệ: {exc}") from exc
    chunks = data["chunks"][:args.limit_chunks]
    if not chunks:
        raise SystemExit("chunks.json không có chunk nào")

    valid_sources = set(source_map.keys())
    counts = allocate_questions(chunks, args.questions)
    print(f"Backend: {args.backend}")
    print(f"Chunks: {len(chunks)} | Requested: {args.questions}")
    random.seed(args.seed)
    backend = None
    if any(counts):
        try:
            if args.backend == "llama_cpp":
                print(f"Loading GGUF: {args.model}")
                backend = LlamaCppBackend(args.model, args.n_ctx, args.n_gpu_layers, args.threads, args.debug, args.seed)
            else:
                print(f"Loading Transformers: {args.model}")
                backend = TransformersBackend(args.model, args.max_input_tokens, args.device_map, args.seed)
        except (RuntimeError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
        print("Model loaded.")
    else:
        print("No chunk has enough text to generate grounded questions.")

    generated = []
    rejected = []
    generated_count = 0

    for idx, (chunk, count) in enumerate(zip(chunks, counts), 1):
        if count <= 0:
            continue
        print(f"[{idx}/{len(chunks)}] Sources: {chunk['source_indices']} | Allocated: {count}")
        prompt = build_generator_prompt(chunk, count)
        generation_budget = min(args.max_new_tokens, max(800, count * 350))
        print(f"Generation max_new_tokens: {generation_budget}")
        try:
            result = ask_json(
                backend,
                messages=[
                    {"role": "system", "content": "Bạn tạo MCQ có căn cứ hoàn toàn từ SOURCE và chỉ trả JSON."},
                    {"role": "user", "content": prompt},
                ],
                schema=MCQ_SCHEMA,
                max_tokens=generation_budget,
                temperature=0.4,
                debug_path=Path("debug") / f"chunk_{idx:03d}" if args.debug else None,
            )
        except ValueError as exc:
            print(f"  Generator error: {exc}")
            rejected.append({"reason": "generator_error", "chunk_id": chunk.get("id", idx), "error": str(exc)})
            continue
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        qs = result.get("questions", [])
        if not isinstance(qs, list):
            rejected.append({"reason": "questions_not_list", "chunk_id": chunk.get("id", idx)})
            continue
        generated_count += len(qs)
        if len(qs) > count:
            rejected.extend({"reason": "over_quota", "question": q} for q in qs[count:])
            qs = qs[:count]
        print(f"  Generated: {len(qs)}")

        for question_index, q in enumerate(qs, 1):
            issues = validate_mcq(q, valid_sources)
            if issues:
                rejected.append({"reason": "structure", "issues": issues, "question": q})
                continue

            source_key = (q["source_type"], q["source_index"])
            source_text = source_map[source_key]
            score = evidence_score(q["evidence"], source_text)
            if score < args.evidence_threshold:
                rejected.append({"reason": "evidence", "score": score, "question": q})
                continue

            if args.skip_verifier:
                verification = {"status": "SKIPPED"}
            else:
                print(f"    Verify: {q['question'][:70]}")
                try:
                    verification = ask_json(
                        backend,
                        messages=[
                            {"role": "system", "content": "Bạn là verifier nghiêm ngặt. Chỉ dựa trên SOURCE."},
                            {"role": "user", "content": build_verifier_prompt(q, source_text)},
                        ],
                        schema=VERIFY_SCHEMA,
                        max_tokens=min(args.max_new_tokens, 1600),
                        temperature=0,
                        debug_path=Path("debug") / f"verifier_{idx:03d}_{question_index:03d}" if args.debug else None,
                    )
                except ValueError as exc:
                    rejected.append({"reason": "verifier_error", "error": str(exc), "question": q})
                    continue
                except RuntimeError as exc:
                    raise SystemExit(str(exc)) from exc

            status = verification.get("status")
            print(f"    Verifier: {status}")
            final_q = None
            if status in ("PASS", "SKIPPED"):
                final_q = q
            elif status == "REVISE":
                final_q = verification.get("corrected_question")
                if not isinstance(final_q, dict):
                    rejected.append({"reason": "revision_missing", "question": q})
                    continue
                issues2 = validate_mcq(final_q, valid_sources)
                if issues2:
                    rejected.append({"reason": "revision_structure", "issues": issues2, "question": final_q})
                    continue
                src2 = source_map[(final_q["source_type"], final_q["source_index"])]
                score2 = evidence_score(final_q["evidence"], src2)
                if score2 < args.evidence_threshold:
                    rejected.append({"reason": "revision_evidence", "score": score2, "question": final_q})
                    continue
                score = score2
            else:
                rejected.append({
                    "reason": "verifier_reject",
                    "issues": verification.get("issues", []),
                    "question": q,
                })
                continue

            final_q["evidence_score"] = round(score, 4)
            final_q["verifier_status"] = status
            generated.append(final_q)

    before_dedup = len(generated)
    final_questions = deduplicate(generated)
    final_questions = final_questions[: args.questions]

    for i, q in enumerate(final_questions, 1):
        q["id"] = i
        q["source"] = {"type": q["source_type"], "index": q["source_index"]}

    output = {
        "document": data.get("document", chunks_path.name),
        "backend": args.backend,
        "model": args.model,
        "requested_questions": args.questions,
        "generated_questions": generated_count,
        "accepted_before_dedup": before_dedup,
        "accepted_questions": len(final_questions),
        "rejected_questions": len(rejected),
        "questions": final_questions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.debug:
        reject_path = Path("debug/rejected_questions.json")
        reject_path.parent.mkdir(parents=True, exist_ok=True)
        reject_path.write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rejected log: {reject_path}")

    print("\nDone")
    print(f"Generated           : {generated_count}")
    print(f"Accepted before dedup: {before_dedup}")
    print(f"Accepted final       : {len(final_questions)}")
    print(f"Evidence rejected   : {sum(item['reason'] in ('evidence', 'revision_evidence') for item in rejected)}")
    print(f"Verifier rejected   : {sum(item['reason'].startswith('verifier_') for item in rejected)}")
    print(f"Deduplicated        : {before_dedup - len(final_questions)}")
    print(f"Rejected             : {len(rejected)}")
    if len(final_questions) < args.questions:
        print("Reason               : insufficient grounded questions")
    print(f"Saved                : {out_path}")


if __name__ == "__main__":
    main()
