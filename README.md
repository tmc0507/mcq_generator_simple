# Trợ lý tạo đề và nhận xét bài làm

Một pipeline xử lý tài liệu và kiểm chứng MCQ, với hai backend inference:

```text
PDF / PPTX / DOCX / CSV / DOC / PPT
        ↓
C++ extract text
        ↓
C++ chia chunk
        ↓
Qwen3-1.7B/8B GGUF local hoặc Qwen3-8B Transformers 4-bit trên Kaggle
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

CLI không có REST API, database, RAG hay fine-tune. Giao diện Streamlit chạy local và lưu đề/bài làm trong `output/`.

## LOCAL MODE — GGUF + llama-cpp-python

### Dependencies Ubuntu

```bash
sudo apt update
sudo apt install -y build-essential cmake poppler-utils unzip libreoffice-writer libreoffice-impress python3-venv python3-pip
```

`pdftotext` từ `poppler-utils` được dùng để đọc PDF. `unzip` được dùng để đọc XML bên trong PPTX/DOCX.
`.doc` cần LibreOffice Writer và `.ppt` cần LibreOffice Impress để chuyển tạm sang định dạng mới trước khi trích text. CSV dùng dòng đầu làm tên cột và giữ số dòng làm nguồn dẫn chứng.

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
pip install -r scripts/requirements.txt
```

#### NVIDIA CUDA

Nếu muốn build `llama-cpp-python` có CUDA:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall llama-cpp-python
```

Nếu máy thiếu dependency build hoặc CUDA compiler không tương thích, hãy thử CPU trước để xác nhận pipeline chạy đúng.

### Model nhẹ cho giao diện

Giao diện mặc định dùng `modelv1`: Qwen3-1.7B Q4_K_M GGUF (khoảng 1.2 GB). Ba file GGUF nằm trong thư mục `models/` theo tên:

| Tên file | Model gốc | Vai trò |
| --- | --- | --- |
| `modelv1.gguf` | Qwen3-1.7B Q4_K_M | Nhanh nhất, mặc định |
| `modelv2.gguf` | Qwen3-8B Q4_K_M | Chất lượng cao hơn, chậm hơn |
| `modelv3.gguf` | Qwen3-4B Q4_K_M | Mức trung gian |

Sau khi clone GitHub, tải và ghép các model từ [GitHub Release models-v1](https://github.com/tmc0507/mcq_generator_simple/releases/tag/models-v1):

```bash
bash scripts/download_models.sh
```

File model lớn được chia thành các phần dưới 2 GiB để upload lên GitHub Release; script kiểm tra SHA-256 sau khi ghép. Nếu chỉ cần bản nhẹ, chạy `bash scripts/download_models.sh modelv1`.

Cũng có thể tải trực tiếp từ Hugging Face. Các model gốc được phát hành theo giấy phép [Apache 2.0](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/LICENSE): [1.7B](https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF), [4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF), [8B](https://huggingface.co/Qwen/Qwen3-8B-GGUF).

```bash
pip install -U huggingface_hub
hf download ggml-org/Qwen3-1.7B-GGUF Qwen3-1.7B-Q4_K_M.gguf --local-dir models
mv models/Qwen3-1.7B-Q4_K_M.gguf models/modelv1.gguf
```

Trên chunk SVM mẫu, modelv1 tạo một câu trong khoảng 23 giây sau khi load trên máy CPU thử nghiệm; thời gian và chất lượng có thể khác theo tài liệu.

### Model 8B cho CLI

Dùng file:

```text
Qwen3-8B-Q4_K_M.gguf
```

Đặt tại:

```text
models/modelv2.gguf
```

Có thể tải bằng Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf \
  --local-dir models
mv models/Qwen3-8B-Q4_K_M.gguf models/modelv2.gguf
```

Bản 4B dùng lệnh `hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models`, sau đó đổi tên thành `models/modelv3.gguf`.

Model khoảng 5 GB.

### Giao diện Streamlit

```bash
streamlit run mcq_streamlit.py
```

Mọi thao tác nằm trong **một cuộc trò chuyện**:

1. Bấm dấu **+** trong ô chat để đính kèm PDF/DOC/DOCX/PPT/PPTX/CSV, nhập `Tạo 10 câu hỏi` rồi gửi. Nếu không ghi số câu, mặc định là 10 (tối đa 50). Mặc định giao diện tạo bản nháp nhanh: kiểm tra cấu trúc tối thiểu, giữ nguyên dẫn chứng và cả câu có thể trùng; không gọi verifier AI.
   Bản nháp nhanh tạo từng câu để tránh một lượt suy luận quá dài; số câu được phân bổ theo độ dài chunk. Nếu model tạo ít câu hơn yêu cầu, đề vẫn được lưu với số câu thực tế. Có thể bật **Kiểm tra đáp án bằng AI** nếu muốn rà soát sâu hơn, nhưng sẽ chậm hơn.
2. Khi đề hiện trong chat, bấm **Tải bộ đề TXT** để lưu bản chỉ gồm câu hỏi và lựa chọn. Đề JSON lưu ở `output/questions.json`, bản TXT ở `output/questions.txt`.
3. Sinh viên nhập tên/mã, chọn đáp án và **Nộp bài** ngay trong thẻ đề. Kết quả lưu ở `output/attempts.json`.
4. Nhập câu hỏi tiếp trong cùng ô chat, ví dụ `Nhận xét bài làm này và gợi ý phần cần ôn`. Chatbot dùng kết quả bài vừa nộp; nhận xét từ một bài kiểm tra không phải đánh giá năng lực toàn diện.

Thanh bên cho phép đổi giữa `modelv1` (GGUF 1.7B, mặc định), `modelv2` (GGUF 8B) và `modelv3` (GGUF 4B), chỉnh prompt tạo đề hoặc bắt đầu cuộc trò chuyện mới. Giao diện chưa có đăng nhập, chỉ dùng trên máy hoặc mạng tin cậy.

Giao diện tự sửa các lỗi định dạng đơn giản như `b.` thành đáp án `B`, nhưng vẫn yêu cầu 4 lựa chọn và một đáp án hợp lệ để chấm được. Dẫn chứng và câu trùng trong bản nháp không được kiểm tra; giảng viên phải rà soát trước khi giao bài. Trong lúc tạo đề, nút gửi ở ô chat đổi thành **Dừng**. Nếu đã có câu được giữ, lần hiển thị tiếp theo sẽ cho tải phần đề đó.

Model được cache để tạo đề và chat dùng cùng một bản trong RAM. Chạy CPU với 8B có thể chậm; dùng modelv1 hoặc GPU nếu có.

### Chạy toàn pipeline

```bash
./run.sh \
  "input/Slides CHAPTER - SVM.pdf" \
  "models/modelv2.gguf" \
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
  --model models/modelv2.gguf \
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

Hỗ trợ `.pdf`, `.pptx`, `.docx`, `.csv`, `.ppt`, `.doc`. Chưa có OCR cho PDF scan hay đọc ảnh trong slide. Có thể đổi `--model` thành đường dẫn model khác mà không sửa pipeline.
