"""Unit tests for special-token escape.

Run directly with pytest from anywhere; does not require ``sglang`` to be
installed because it loads the modules-under-test via importlib stubs.
"""

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REFERENCE_TOKENIZER = (
    "/Users/lambda/S-Workspace/glm-moe-converters/legacy-assets/2603/155k-tokenizer"
)
SEED = 0xC0DECAFE


def _ensure_package(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_escape_modules():
    here = Path(__file__).resolve()
    root = here.parent
    srt_dir = root.parent.parent

    for pkg in (
        "sglang",
        "sglang.srt",
        "sglang.srt.constrained",
        "sglang.srt.constrained.glm",
        "sglang.srt.utils",
    ):
        _ensure_package(pkg)

    escape_mod = _load_module(
        "sglang.srt.constrained.glm.escape", root / "escape.py"
    )
    tokenizer_escape_mod = _load_module(
        "sglang.srt.utils.tokenizer_escape",
        srt_dir / "utils" / "tokenizer_escape.py",
    )
    ebnf_mod = _load_module(
        "sglang.srt.constrained.glm.ebnf_utils", root / "ebnf_utils.py"
    )
    tool_schema_mod = _load_module(
        "sglang.srt.constrained.glm.tool_schema", root / "tool_schema.py"
    )
    schema_mod = _load_module(
        "sglang.srt.constrained.glm.schema", root / "schema.py"
    )
    return escape_mod, tokenizer_escape_mod, schema_mod


@pytest.fixture(scope="module")
def loaded():
    return _load_escape_modules()


@pytest.fixture(scope="module")
def tokenizer(loaded):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from transformers import AutoTokenizer

    if not Path(REFERENCE_TOKENIZER).exists():
        pytest.skip(f"reference tokenizer not found at {REFERENCE_TOKENIZER}")

    escape_mod, tokenizer_escape_mod, _ = loaded
    escape_mod.reset_global_escaped_special_tokens()
    tok = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER, trust_remote_code=True)
    tokenizer_escape_mod.escape_tokenizer_special_tokens(tok, SEED)
    yield tok
    escape_mod.reset_global_escaped_special_tokens()


def test_hash_suffix_deterministic(loaded):
    escape_mod, _, _ = loaded
    assert escape_mod.hash_suffix(SEED, "<think>") == escape_mod.hash_suffix(
        SEED, "<think>"
    )
    assert escape_mod.hash_suffix(SEED, "<think>") != escape_mod.hash_suffix(
        SEED + 1, "<think>"
    )
    expected = hashlib.sha256(f"{SEED}:<think>".encode("utf-8")).hexdigest()[:8]
    assert escape_mod.hash_suffix(SEED, "<think>") == expected


def test_global_object_populated(loaded, tokenizer):
    escape_mod, _, _ = loaded
    sp = escape_mod.get_global_escaped_special_tokens()
    assert sp.enabled
    assert sp.seed == SEED
    assert sp.get("<think>") != "<think>"
    assert sp.get("<think>").startswith("<think><")
    assert sp.get("<think>").endswith(">")
    assert len(sp.get("<think>")) == len("<think>") + 1 + 8 + 1


def test_get_falls_back_to_original_for_unknown(loaded, tokenizer):
    escape_mod, _, _ = loaded
    sp = escape_mod.get_global_escaped_special_tokens()
    assert sp.get("not-a-special-token") == "not-a-special-token"


def test_raw_special_token_is_bpe_split(tokenizer):
    ids = tokenizer.encode("<think>", add_special_tokens=False)
    assert len(ids) > 1


def test_escaped_token_encodes_to_original_id(loaded, tokenizer):
    escape_mod, _, _ = loaded
    sp = escape_mod.get_global_escaped_special_tokens()
    escaped = sp.get("<think>")
    ids = tokenizer.encode(escaped, add_special_tokens=False)
    assert len(ids) == 1
    decoded = tokenizer.decode(ids)
    assert decoded == escaped


def test_eos_token_id_unchanged(tokenizer):
    assert tokenizer.eos_token_id == 154820


def test_chat_template_contains_escaped_tokens(loaded, tokenizer):
    escape_mod, _, _ = loaded
    sp = escape_mod.get_global_escaped_special_tokens()
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    assert sp.get("<|user|>") in out
    assert sp.get("<|assistant|>") in out
    assert "<think>" not in out.replace(sp.get("<think>"), "")
    assert "</think>" not in out.replace(sp.get("</think>"), "")


def test_idempotent_double_apply(loaded, tokenizer):
    _, tokenizer_escape_mod, _ = loaded
    tokenizer_escape_mod.escape_tokenizer_special_tokens(tokenizer, SEED)
    ids = tokenizer.encode("<think>", add_special_tokens=False)
    assert len(ids) > 1


def test_double_apply_different_seed_fails(loaded, tokenizer):
    _, tokenizer_escape_mod, _ = loaded
    with pytest.raises(AssertionError):
        tokenizer_escape_mod.escape_tokenizer_special_tokens(tokenizer, SEED + 1)


def test_save_pretrained_roundtrip_equivalent(loaded, tokenizer, tmp_path):
    escape_mod, _, _ = loaded
    sp = escape_mod.get_global_escaped_special_tokens()
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    save_dir = tmp_path / "tok_escaped"
    tokenizer.save_pretrained(str(save_dir))
    tok2 = AutoTokenizer.from_pretrained(str(save_dir), trust_remote_code=True)
    for original in ("<think>", "</think>", "<tool_call>", "<|assistant|>"):
        escaped = sp.get(original)
        assert tokenizer.encode(escaped, add_special_tokens=False) == tok2.encode(
            escaped, add_special_tokens=False
        )


def test_schema_get_special_token_config_uses_escape(loaded, tokenizer):
    escape_mod, _, schema_mod = loaded
    sp = escape_mod.get_global_escaped_special_tokens()
    cfg = schema_mod.get_special_token_config(tokenizer)
    assert cfg.begin_of_thinking == sp.get("<think>")
    assert cfg.end_of_thinking == sp.get("</think>")
    assert cfg.begin_of_tool_call == sp.get("<tool_call>")
    assert cfg.end_of_tool_call == sp.get("</tool_call>")
    assert cfg.begin_of_key == sp.get("<arg_key>")
    assert cfg.end_of_key == sp.get("</arg_key>")
    assert cfg.begin_of_value == sp.get("<arg_value>")
    assert cfg.end_of_value == sp.get("</arg_value>")
    assert cfg.assistant_token == sp.get("<|assistant|>")


def test_ebnf_contains_escaped_tokens(loaded, tokenizer):
    escape_mod, _, schema_mod = loaded
    sp = escape_mod.get_global_escaped_special_tokens()
    cfg = schema_mod.get_special_token_config(tokenizer)
    ebnf = schema_mod.generation_constraint(
        enable_thinking=True,
        functions=None,
        special_tokens=cfg,
        chat_template_version="glm47",
        accommodate_chat_template=False,
        allow_multiple_assistant_turns=False,
    )
    assert sp.get("<think>") in ebnf
    assert sp.get("</think>") in ebnf
    assert f'"<think>"' not in ebnf
    assert f'"</think>"' not in ebnf


def test_no_escape_global_is_identity(loaded):
    escape_mod, _, _ = loaded
    escape_mod.reset_global_escaped_special_tokens()
    sp = escape_mod.get_global_escaped_special_tokens()
    assert not sp.enabled
    assert sp.get("<think>") == "<think>"
    assert sp.get("<tool_call>") == "<tool_call>"


def test_no_backend_raises(loaded):
    _, tokenizer_escape_mod, _ = loaded

    class FakeSlowTokenizer:
        pass

    with pytest.raises(RuntimeError, match="fast tokenizer"):
        tokenizer_escape_mod.escape_tokenizer_special_tokens(
            FakeSlowTokenizer(), SEED
        )


def test_seed_must_be_int(loaded):
    _, tokenizer_escape_mod, _ = loaded
    with pytest.raises(AssertionError, match="must be an int"):
        tokenizer_escape_mod.escape_tokenizer_special_tokens(object(), None)
