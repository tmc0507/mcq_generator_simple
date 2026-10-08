# Repository Guidelines

## Project Structure & Module Organization

`src/doc_processor.cpp` extracts text from PDF, PPTX, and DOCX files and writes chunks to `work/chunks.json`. `scripts/generate_mcq.py` turns those chunks into grounded MCQs; `mcq_streamlit.py` provides the interactive UI and batch mode. `test_mcq_streamlit.py` covers prompt generation and the UI. Sample documents live in `input/`. Keep local GGUF models in `models/`, generated files in `work/` and `output/`, and CMake artifacts in `build/`.

## Build, Test, and Development Commands

Install `pdftotext` (Poppler) and `unzip` for document extraction. Create a Python virtual environment, then install `llama-cpp-python` and Streamlit for the generator, UI, and test.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt streamlit
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 test_mcq_streamlit.py
streamlit run mcq_streamlit.py
./run.sh "input/Slides CHAPTER - SVM.pdf" "models/Qwen3-8B-Q4_K_M.gguf" 20
```

The CMake commands build `build/doc_processor`. The test runs without loading the model. Streamlit starts the UI; `run.sh` rebuilds the extractor and writes `output/questions.json`. For extraction alone, run `./build/doc_processor "input/Slides CHAPTER - SVM.pdf" --output work/chunks.json`.

## Coding Style & Naming Conventions

Use four-space indentation in Python and C++, Python `snake_case` for functions and variables, and the existing `PascalCase` C++ struct names. Keep C++ compatible with C++20 and its enabled `-Wall -Wextra -Wpedantic` warnings. No formatter or linter is configured; follow nearby code. Preserve source references and evidence validation when changing MCQ generation.

## Testing Guidelines

The existing test uses `streamlit.testing.v1.AppTest` and plain assertions; run it with `python3 test_mcq_streamlit.py`. Name new test functions `test_*` and cover changed validation or generation behavior with small synthetic inputs. There is no configured coverage target. For extractor changes, also run `doc_processor` on a sample file from `input/` and inspect its chunk JSON.

## Commit & Pull Request Guidelines

This checkout has no usable Git history, so no existing commit convention can be verified. Use a short, imperative commit subject describing the change. In pull requests, explain the behavior changed, list commands run and results, and mention any model or sample document used. Include a screenshot when the Streamlit UI changes. Do not commit GGUF models or generated question files.
