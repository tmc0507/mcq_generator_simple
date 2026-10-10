import json
import subprocess
import sys
import tempfile
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from scripts import generate_mcq as pipeline


QUESTION = {
    "question": "Dữ kiện gốc là gì?",
    "choices": {"A": "Dữ kiện gốc.", "B": "Sai 1", "C": "Sai 2", "D": "Sai 3"},
    "correct_answer": "A", "concept": "Dữ kiện", "difficulty": "easy",
    "explanation": "Dữ kiện gốc.", "evidence": "Dữ kiện gốc.",
    "source_type": "page", "source_index": 1,
}


class FakeBackend:
    instances = []

    def __init__(self, *args):
        self.calls = []
        self.instances.append(self)

    def generate_text(self, messages, **kwargs):
        self.calls.append(kwargs)
        if kwargs["do_sample"]:
            return "<think>discard</think>```json\n" + json.dumps({"questions": [QUESTION]}) + "\n```"
        return json.dumps({"status": "PASS", "issues": [], "corrected_question": None})


def test_parsing_and_validation():
    assert pipeline.parse_json_text("<think>ignore</think>```json\n{\"ok\": true}\n```") == {"ok": True}
    assert pipeline.allocate_questions([{"text": "short"}, {"text": "x" * 200}], 3) == [0, 3]
    assert pipeline.allocate_questions([{"text": "x" * 120, "source_types": ["row"]}], 2) == [2]
    assert not pipeline.validate_mcq({**QUESTION, "source_type": "row", "source_index": 2}, {("row", 2)})
    assert "bằng tiếng Việt" in pipeline.build_generator_prompt(
        {"text": "SOURCE", "source_types": ["page"], "source_indices": [1]}, 1)
    item = pipeline.mcq_schema_for_chunk({"source_types": ["page"], "source_indices": [4, 5]})[
        "properties"]["questions"]["items"]["properties"]
    assert item["source_index"]["enum"] == [4, 5]
    assert item["source_type"]["enum"] == ["page"]
    broken = {**QUESTION, "correct_answer": []}
    assert pipeline.validate_mcq(broken, {("page", 1)})
    repaired = pipeline.normalize_mcq_format(
        {**QUESTION, "choices": ["Đúng", "Sai 1", "Sai 2", "Sai 3"],
         "correct_answer": "a.", "source_index": "1", "source_type": "PAGE",
         "difficulty": "Easy"},
        {"source_types": ["page"], "source_indices": [1]},
    )
    assert not pipeline.validate_mcq(repaired, {("page", 1)})
    assert repaired["choices"]["A"] == "Đúng"
    assert pipeline.validate_mcq({**repaired, "correct_answer": "E"}, {("page", 1)})

    class RetryBackend:
        calls = 0

        def generate_text(self, messages, **kwargs):
            self.calls += 1
            return "invalid" if self.calls == 1 else '{"questions": []}'

    backend = RetryBackend()
    assert pipeline.ask_json(backend, [{"role": "user", "content": "SOURCE:\ntext"}], {}, 800, 0.4) == {"questions": []}
    assert backend.calls == 2


def test_csv_extraction():
    processor = Path(__file__).resolve().parent / "build/doc_processor"
    if not processor.is_file():
        return
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "sample.csv"
        output = Path(directory) / "chunks.json"
        source.write_text('topic,detail\nAlgebra,"comma, here"\nGeometry,"two\nlines"\nLogic,"said ""hi"""\n', encoding="utf-8")
        subprocess.run([str(processor), str(source), "--output", str(output)], check=True, capture_output=True)
        data = json.loads(output.read_text(encoding="utf-8"))
        assert [(block["type"], block["index"]) for block in data["blocks"]] == [("row", 2), ("row", 3), ("row", 4)]
        assert "comma, here" in data["blocks"][0]["text"]
        assert "two\nlines" in data["blocks"][1]["text"]
        assert 'said "hi"' in data["blocks"][2]["text"]
        pipeline.validate_chunks(data)


def test_cli_shared_pipeline():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chunks = root / "chunks.json"
        chunks.write_text(json.dumps({
            "document": "sample.pdf",
            "blocks": [{"type": "page", "index": 1, "text": "Dữ kiện gốc. " * 20}],
            "chunks": [{"id": 1, "text": "Dữ kiện gốc. " * 20,
                        "source_types": ["page"], "source_indices": [1]}],
        }), encoding="utf-8")
        model = root / "model.gguf"
        model.touch()
        for backend_name, model_name in (("llama_cpp", str(model)), ("transformers", "example/model")):
            output = root / f"{backend_name}.json"
            args = ["generate_mcq.py", "--backend", backend_name, "--chunks", str(chunks),
                    "--model", model_name, "--questions", "1", "--output", str(output)]
            FakeBackend.instances.clear()
            with patch.object(pipeline, "LlamaCppBackend", FakeBackend), \
                    patch.object(pipeline, "TransformersBackend", FakeBackend), \
                    patch.object(sys, "argv", args):
                pipeline.main()
            result = json.loads(output.read_text(encoding="utf-8"))
            assert len(FakeBackend.instances) == 1
            assert [call["do_sample"] for call in FakeBackend.instances[0].calls] == [True, False]
            assert result["backend"] == backend_name
            assert len(result["exam_id"]) == 32
            assert result["model"] == ("model.gguf" if backend_name == "llama_cpp" else "example/model")
            assert result["requested_questions"] == result["accepted_questions"] == 1
            assert result["questions"][0]["source"] == {"type": "page", "index": 1}
            assert result["questions"][0]["verifier_status"] == "PASS"

        class BadRevisionBackend(FakeBackend):
            def generate_text(self, messages, **kwargs):
                if kwargs["do_sample"]:
                    return super().generate_text(messages, **kwargs)
                self.calls.append(kwargs)
                corrected = {**QUESTION, "choices": {"B": "Sai"}}
                return json.dumps({"status": "REVISE", "issues": [], "corrected_question": corrected})

        output = root / "revised.json"
        args = ["generate_mcq.py", "--backend", "transformers", "--chunks", str(chunks),
                "--model", "example/model", "--questions", "1", "--output", str(output)]
        with patch.object(pipeline, "TransformersBackend", BadRevisionBackend), patch.object(sys, "argv", args):
            pipeline.main()
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["accepted_questions"] == 0
        assert result["rejected_questions"] == 1


def test_transformers_adapter_without_gpu_download():
    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = (1, len(values))

        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = None
        eos_token_id = 99

        def __init__(self):
            self.flags = []

        def apply_chat_template(self, messages, **kwargs):
            self.flags.append(kwargs)
            return " ".join(message["content"] for message in messages)

        def __call__(self, prompt, **kwargs):
            return {"input_ids": Tensor(prompt.split())}

        def encode(self, source, **kwargs):
            return source.split()

        def decode(self, tokens, **kwargs):
            return " ".join(tokens)

    class Model:
        config = types.SimpleNamespace(max_position_embeddings=1000)

        def __init__(self):
            self.calls = []

        def get_input_embeddings(self):
            return types.SimpleNamespace(weight=types.SimpleNamespace(device="cuda:0"))

        def eval(self):
            return self

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [kwargs["input_ids"].values + ['{"ok": true}']]

    tokenizer, model = Tokenizer(), Model()
    config = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.manual_seed = lambda seed: None
    fake_torch.inference_mode = nullcontext
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, manual_seed_all=lambda seed: None,
        get_device_name=lambda index: "Tesla T4",
        get_device_properties=lambda index: types.SimpleNamespace(total_memory=16 * 2**30),
    )
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.random = types.SimpleNamespace(seed=lambda seed: None)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = lambda **kwargs: config.update(kwargs) or config
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: tokenizer)
    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda name, **kwargs: model,
    )
    with patch.dict(sys.modules, {"torch": fake_torch, "numpy": fake_numpy, "transformers": fake_transformers}):
        backend = pipeline.TransformersBackend("example/model", max_input_tokens=300)
        messages = [{"role": "system", "content": "JSON only"},
                    {"role": "user", "content": "Question\nSOURCE:\n" + "word " * 600}]
        assert backend.generate_text(messages, 120, 0.4, True) == '{"ok": true}'
        assert backend.generate_text(messages, 120, 0, False) == '{"ok": true}'
    assert config["load_in_4bit"] and config["bnb_4bit_quant_type"] == "nf4"
    assert config["bnb_4bit_use_double_quant"] and config["bnb_4bit_compute_dtype"] == "float16"
    assert all(flag["enable_thinking"] is False for flag in tokenizer.flags)
    assert all(call["input_ids"].shape[-1] <= 300 for call in model.calls)
    assert [call["do_sample"] for call in model.calls] == [True, False]
    assert all(call["pad_token_id"] == 99 for call in model.calls)


def test_local_adapter_without_model_load():
    calls = {}

    class Llama:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def create_chat_completion(self, **kwargs):
            calls["generate"] = kwargs
            return {"choices": [{"message": {"content": '{"questions": []}'}}]}

    module = types.ModuleType("llama_cpp")
    module.Llama = Llama
    with patch.dict(sys.modules, {"llama_cpp": module}):
        backend = pipeline.LlamaCppBackend("model.gguf", n_ctx=4096, n_gpu_layers=10)
        assert backend.generate_text([], 800, 0.4, True, {"type": "object"}) == '{"questions": []}'
    assert calls["init"]["n_gpu_layers"] == 10
    assert calls["generate"]["response_format"]["type"] == "json_object"


if __name__ == "__main__":
    test_parsing_and_validation()
    test_csv_extraction()
    test_cli_shared_pipeline()
    test_transformers_adapter_without_gpu_download()
    test_local_adapter_without_model_load()
    print("OK: JSON, allocation, shared backends, verifier, output")
