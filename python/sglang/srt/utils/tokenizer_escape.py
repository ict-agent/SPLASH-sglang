"""In-place rewrite of every ``added_tokens_decoder`` entry on a HuggingFace
fast tokenizer so that ``content`` becomes ``f"{original}<{8-hex}>"`` while the
token id is preserved. Angle brackets are used (not square brackets) to keep
the escaped strings safe to embed verbatim in EBNF / xgrammar string literals.

The mutation is applied directly to ``tokenizer._tokenizer`` via the backend
``tokenizers.Tokenizer.to_str``/``from_str`` round-trip; this has been verified
to be equivalent to ``save_pretrained`` + edit + ``from_pretrained``.

Also rewrites the seven Python-level special-token attributes (eos/pad/bos/sep/
cls/unk/mask) and the chat_template, then installs a process-global
``EscapedSpecialTokens`` so downstream parsers and constraints pick up the
escaped strings.
"""

import json
import logging
from typing import Optional

from sglang.srt.constrained.glm.escape import (
    EscapedSpecialTokens,
    escape_token,
    get_global_escaped_special_tokens,
    set_global_escaped_special_tokens,
)

logger = logging.getLogger(__name__)

_ESCAPE_FLAG_ATTR = "_sglang_special_token_escape_seed"


def escape_tokenizer_special_tokens(tokenizer, seed: int) -> EscapedSpecialTokens:
    """Mutate ``tokenizer`` in place and install the process-global mapping.

    Callers must only invoke this when escape is enabled (seed is an int from
    user CLI). To disable escape, simply do not call this function — the global
    singleton then returns an identity ``EscapedSpecialTokens()``.
    """
    assert isinstance(seed, int), (
        f"seed must be an int when escape is enabled, got {type(seed).__name__}"
    )

    prev = getattr(tokenizer, _ESCAPE_FLAG_ATTR, None)
    if prev is not None:
        assert prev == seed, (
            f"escape_tokenizer_special_tokens called twice with different seeds "
            f"({prev} vs {seed}); refusing to remutate."
        )
        return get_global_escaped_special_tokens()

    if not hasattr(tokenizer, "backend_tokenizer"):
        raise RuntimeError(
            "--glm-special-token-escape-seed requires a fast tokenizer with a "
            "tokenizers backend; got "
            f"{type(tokenizer).__name__} which has no backend_tokenizer."
        )

    from tokenizers import Tokenizer

    backend = tokenizer.backend_tokenizer
    backend_json = json.loads(backend.to_str())
    added = backend_json.get("added_tokens", [])
    if not added:
        raise RuntimeError(
            "Tokenizer has no added_tokens; --glm-special-token-escape-seed has "
            "nothing to escape. Use a fast tokenizer that exposes special tokens "
            "via added_tokens_decoder."
        )

    originals = [t["content"] for t in added]
    mapping = {o: escape_token(seed, o) for o in originals}

    for entry in backend_json["added_tokens"]:
        entry["content"] = mapping[entry["content"]]
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(backend_json))

    for attr in (
        "eos_token",
        "pad_token",
        "bos_token",
        "sep_token",
        "cls_token",
        "unk_token",
        "mask_token",
    ):
        val = getattr(tokenizer, attr, None)
        if val and val in mapping and mapping[val] != val:
            try:
                setattr(tokenizer, attr, mapping[val])
            except Exception as e:
                logger.warning("Failed to setattr %s on tokenizer: %s", attr, e)

    ct = getattr(tokenizer, "chat_template", None)
    if ct:
        for original in sorted(mapping.keys(), key=len, reverse=True):
            escaped = mapping[original]
            if escaped != original:
                ct = ct.replace(original, escaped)
        tokenizer.chat_template = ct

    try:
        addl = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        if addl and any(t in mapping for t in addl):
            tokenizer.additional_special_tokens = [
                mapping.get(t, t) for t in addl
            ]
    except Exception as e:
        logger.debug(
            "additional_special_tokens not writable on %s: %s",
            type(tokenizer).__name__,
            e,
        )

    obj = EscapedSpecialTokens(mapping=mapping, seed=seed, enabled=True)
    set_global_escaped_special_tokens(obj)
    setattr(tokenizer, _ESCAPE_FLAG_ATTR, seed)

    sample_keys = list(mapping.keys())[:2]
    sample = {k: mapping[k] for k in sample_keys}
    logger.info(
        "Applied --glm-special-token-escape-seed=%d to tokenizer; %d tokens "
        "mutated (sample: %s).",
        seed,
        len(mapping),
        sample,
    )
    return obj
