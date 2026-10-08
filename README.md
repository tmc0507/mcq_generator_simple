# MCQ Generator Simple V1

Một pipeline xử lý tài liệu và kiểm chứng MCQ, với hai backend inference:

```text
PDF / PPTX / DOCX
        ↓
C++ extract text
        ↓
C++ chia chunk
        ↓
Qwen3-8B: GGUF local hoặc Transformers 4-bit trên Kaggle
        ↓
Python (model chỉ load 1 lần cho generator và verifier)
        ↓
MCQ JSON
        ↓
Evidence check
        ↓
Qwen verifier
        ↓
Deduplicate
        ↓
output/questions.json
```

Không có HTTP, REST API, database, RAG hay fine-tune ở V1.

## LOCAL MODE — GGUF + llama-cpp-python

### Dependencies Ubuntu

```bash
sudo apt update
sudo apt install -y build-essential cmake poppler-utils unzip python3-venv python3-pip
```

`pdftotext` từ `poppler-utils` được dùng để đọc PDF. `unzip` được dùng để đọc XML bên trong PPTX/DOCX.

### Build C++ document processor

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Test riêng phần đọc/chunk tài liệu:

```bash
./build/doc_processor \
  "input/Slides CHAPTER - SVM.pdf" \
  --output work/chunks.json \
  --blocks-per-chunk 3 \
  --chunk-chars 12000
```

Kết quả trung gian nằm ở:

```text
work/chunks.json
```

### Cài llama-cpp-python

Khuyến nghị tạo virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

#### CPU only

```bash
pip install llama-cpp-python
```

#### NVIDIA CUDA

Nếu muốn build `llama-cpp-python` có CUDA:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall llama-cpp-python
```

Nếu máy thiếu dependency build hoặc CUDA compiler không tương thích, hãy thử CPU trước để xác nhận pipeline chạy đúng.

### Model

Dùng file:

```text
Qwen3-8B-Q4_K_M.gguf
```

Đặt tại:

```text
models/Qwen3-8B-Q4_K_M.gguf
```

Có thể tải bằng Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf \
  --local-dir models
```

Model khoảng 5 GB.

### Chạy toàn pipeline

```bash
./run.sh \
  "input/Slides CHAPTER - SVM.pdf" \
  "models/Qwen3-8B-Q4_K_M.gguf" \
  20
```

Hoặc chạy từng bước:

```bash
./build/doc_processor \
  "input/Slides CHAPTER - SVM.pdf" \
  --output work/chunks.json

python3 scripts/generate_mcq.py \
  --backend llama_cpp \
  --chunks work/chunks.json \
  --model models/Qwen3-8B-Q4_K_M.gguf \
  --questions 20 \
  --n-ctx 8192 \
  --n-gpu-layers 20 \
  --output output/questions.json
```

Nếu CUDA/VRAM không ổn, thử:

```bash
--n-gpu-layers 0
```

để chạy CPU, hoặc giảm từ `20` xuống `10`.

`--backend` mặc định là `llama_cpp`, nên command local cũ vẫn chạy. Có thể truyền đối số thứ tư `llama_cpp` hoặc `transformers` vào `run.sh`.

## KAGGLE MODE — Transformers + Qwen3-8B 4-bit

Bật GPU trong Kaggle Settings. Backend dùng GPU 0, NF4, và cùng prompt, schema, evidence checker, verifier, dedup với local. `kaggle/setup.sh` chỉ cài các dependency còn thiếu hoặc chưa đủ phiên bản; không cài `llama-cpp-python`.

### Upload workspace to Kaggle

Từ thư mục **cha** của project, tạo zip không chứa model GGUF và file sinh ra:

```bash
zip -r mcq_generator_simple.zip mcq_generator_simple \
  -x 'mcq_generator_simple/.venv/*' 'mcq_generator_simple/build/*' \
     'mcq_generator_simple/models/*' 'mcq_generator_simple/work/*' \
     'mcq_generator_simple/output/*' 'mcq_generator_simple/debug/*' \
     'mcq_generator_simple/.git/*' 'mcq_generator_simple/.aws/*' \
     'mcq_generator_simple/.codex/*' 'mcq_generator_simple/.agents/*' \
     '*/__pycache__/*' '*.pyc'
```

Upload zip dưới dạng Kaggle Dataset, rồi trong notebook giải nén vào vùng ghi được:

```bash
unzip /kaggle/input/<dataset-name>/mcq_generator_simple.zip -d /kaggle/working
cd /kaggle/working/mcq_generator_simple
bash kaggle/setup.sh
./kaggle/run_kaggle.sh "input/Slides CHAPTER - SVM.pdf" Qwen/Qwen3-8B 10
```

Nếu Kaggle Internet tắt, gắn model vào notebook và truyền thư mục local thay cho model ID, ví dụ `./kaggle/run_kaggle.sh input/file.pdf /kaggle/input/qwen3-model/Qwen3-8B 10`. Dependency cũng cần có sẵn hoặc được cung cấp qua wheel dataset. Có thể chạy riêng phần AI sau khi tạo `work/chunks.json`:

```bash
python3 scripts/generate_mcq.py \
  --backend transformers --chunks work/chunks.json \
  --model Qwen/Qwen3-8B --questions 2 --limit-chunks 1 \
  --output output/test.json
```

`--questions` là tổng số câu cho toàn tài liệu. Chunk quá ngắn không được cấp quota; thiếu câu có căn cứ thì output ít hơn. `--max-input-tokens` giới hạn prompt Transformers (mặc định 8192); `--max-new-tokens` giới hạn output (mặc định 4096). `--device-map auto` là tùy chọn, mặc định chỉ dùng GPU 0. `--skip-verifier` chỉ dành cho debug, còn `--debug` lưu prompt/raw/JSON vào `debug/`.

## Output và kiểm chứng

Cả hai backend ghi `output/questions.json` với `document`, `backend`, `model`, `requested_questions`, `generated_questions`, `accepted_questions`, `rejected_questions`, và `questions`. Mỗi câu có `id`, `question`, bốn `choices`, `correct_answer`, `concept`, `difficulty`, `explanation`, `evidence`, `source` (`type` và `index`), `evidence_score`, `verifier_status`. Hai field cũ `source_type` và `source_index` vẫn được giữ để tương thích local.

Evidence checker chỉ kiểm tra đoạn trích có trong tài liệu; cùng model đang load sẽ kiểm tra nghĩa và đáp án. Sau `REVISE`, code kiểm tra lại cấu trúc và evidence. Qwen3 dùng `/no_think` trong prompt local và `enable_thinking=False` trong chat template Transformers; parser loại `<think>` nếu model vẫn xuất ra.

Hỗ trợ `.pdf`, `.pptx`, `.docx`. Chưa có OCR cho PDF scan, đọc ảnh trong slide, hay hỗ trợ `.ppt`/`.doc` cũ. Có thể đổi `--model` thành đường dẫn model khác mà không sửa pipeline.
