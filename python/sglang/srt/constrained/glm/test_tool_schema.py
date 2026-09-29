"""
Unit tests for tool_schema.py (XML tool-call EBNF generation).

The kv wrapper literals (<arg_key>/<arg_value>) are emitted as shared rules
(arg_key_begin / arg_val_<hash>) instead of being inlined per property, so
these tests verify both the shared-rule structure and, via the xgrammar
matcher, that the accepted language keeps the key <-> value-type coupling.
"""

import pytest

from .ebnf_utils import any_string_exclude
from .schema import SpecialTokenConfig
from .tool_schema import build_tool_call_rules

# Try to import xgrammar, skip matcher tests if not available
try:
    import xgrammar as xgr

    HAS_XGRAMMAR = True
except ImportError:
    HAS_XGRAMMAR = False


class _Function:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters


TEST_FUNCTIONS = [
    _Function(
        "get_weather",
        {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
        },
    ),
    _Function(
        "calc",
        {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "flag": {"type": "boolean"},
                "note": {"type": "string"},
            },
        },
    ),
    _Function("noargs", {"type": "object", "properties": {}}),
    _Function("nullparams", None),
]

SPECIAL_TOKENS = SpecialTokenConfig()


def build_rules(chat_template_version, functions=TEST_FUNCTIONS):
    return build_tool_call_rules(
        non_terminal_name="tool_call_blocks",
        functions=functions,
        special_tokens=SPECIAL_TOKENS,
        chat_template_version=chat_template_version,
    )


def build_grammar(chat_template_version, functions=TEST_FUNCTIONS):
    rules = list(build_rules(chat_template_version, functions))
    rules.extend(
        any_string_exclude(
            "text_without_special_tokens", list(SPECIAL_TOKENS.all_special_tokens())
        )
    )
    return "\n".join(rules)


class TestSharedRuleStructure:
    """Structural tests for the shared kv wrapper rules."""

    def test_arg_key_begin_emitted_once(self):
        rules = build_rules("glm47")
        begin_rules = [r for r in rules if r.startswith("arg_key_begin ::=")]
        assert len(begin_rules) == 1
        assert '"<arg_key>"' in begin_rules[0]

    def test_arg_key_literal_not_inlined_per_property(self):
        """<arg_key> must only appear in the shared arg_key_begin rule."""
        rules = build_rules("glm47")
        containing = [r for r in rules if "<arg_key>" in r]
        assert containing == [r for r in rules if r.startswith("arg_key_begin ::=")]

    def test_arg_val_rules_deduplicated_across_tools(self):
        """get_weather.location and calc.note are both plain strings and must
        share a single arg_val rule; distinct value rules get distinct rules."""
        rules = build_rules("glm47")
        arg_val_rules = [r for r in rules if r.startswith("arg_val_")]
        # string text, enum, number, boolean -> exactly 4 distinct value rules
        assert len(arg_val_rules) == 4
        lhs = [r.split("::=")[0].strip() for r in arg_val_rules]
        assert len(set(lhs)) == 4

    def test_no_duplicate_rule_definitions(self):
        rules = build_rules("glm47")
        lhs = [r.split("::=")[0].strip() for r in rules]
        assert len(lhs) == len(set(lhs)), "duplicate rule definitions"

    def test_no_props_tools_have_empty_arguments(self):
        rules = build_rules("glm47")
        empty_arg_rules = [
            r for r in rules if r.endswith('::= ""') and r.startswith("arguments_")
        ]
        # noargs and nullparams
        assert len(empty_arg_rules) == 2

    def test_deterministic_output(self):
        assert build_rules("glm47") == build_rules("glm47")
        assert build_rules("glm45") == build_rules("glm45")


@pytest.fixture(scope="module")
def compiled():
    tokenizer_info = xgr.TokenizerInfo([])
    compiler = xgr.GrammarCompiler(tokenizer_info)
    return {
        ver: compiler.compile_grammar(
            build_grammar(ver), root_rule_name="tool_call_blocks"
        )
        for ver in ("glm45", "glm47")
    }


@pytest.mark.skipif(not HAS_XGRAMMAR, reason="xgrammar not installed")
class TestAcceptedLanguage:
    """Matcher-based tests that the generated grammar accepts exactly the
    intended tool-call strings."""

    @staticmethod
    def _probe(compiled_grammar, s):
        """Returns (fully_accepted, terminated_at_end)."""
        matcher = xgr.GrammarMatcher(
            compiled_grammar, terminate_without_stop_token=True
        )
        for c in s:
            if not matcher.accept_string(c):
                return False, False
        return True, matcher.is_terminated()

    @staticmethod
    def _call(ver, name, kvs):
        """Build a tool-call string for the given template version."""
        sep = "\n" if ver == "glm45" else ""
        parts = [sep, "<tool_call>", name, sep]
        for i, (key, value) in enumerate(kvs):
            if i > 0:
                parts.append(sep)
            parts.append(f"<arg_key>{key}</arg_key>{sep}<arg_value>{value}</arg_value>")
        if kvs:
            parts.append(sep)
        parts.append("</tool_call>")
        return "".join(parts)

    @pytest.mark.parametrize("ver", ["glm45", "glm47"])
    def test_valid_calls_accepted(self, compiled, ver):
        cases = [
            self._call(ver, "get_weather", [("location", "Beijing")]),
            self._call(
                ver, "get_weather", [("location", "Beijing"), ("unit", "celsius")]
            ),
            # any order, repeats allowed by design (S* form)
            self._call(
                ver,
                "get_weather",
                [("unit", "fahrenheit"), ("location", "SH"), ("location", "BJ")],
            ),
            self._call(
                ver, "calc", [("a", "-3.5"), ("flag", "true"), ("note", "hi there")]
            ),
            self._call(ver, "calc", [("a", "42")]),
            self._call(ver, "noargs", []),
            self._call(ver, "nullparams", []),
            # two sequential tool calls
            self._call(ver, "noargs", [])
            + self._call(ver, "get_weather", [("location", "X")]),
            # zero tool calls
            "",
        ]
        for s in cases:
            accepted, terminated = self._probe(compiled[ver], s)
            assert accepted and terminated, f"[{ver}] should accept+terminate: {s!r}"

    @pytest.mark.parametrize("ver", ["glm45", "glm47"])
    def test_invalid_calls_rejected(self, compiled, ver):
        cases = [
            # unknown tool name
            self._call(ver, "unknown_tool", []),
            # key from another tool
            self._call(ver, "get_weather", [("a", "1")]),
            # unknown key
            self._call(ver, "get_weather", [("wrongkey", "x")]),
            # enum key with non-enum value
            self._call(ver, "get_weather", [("unit", "kelvin")]),
            # number key with non-number value
            self._call(ver, "calc", [("a", "abc")]),
            # boolean key with non-boolean value
            self._call(ver, "calc", [("flag", "yes")]),
            # args on a no-arg tool
            self._call(ver, "noargs", [("location", "X")]),
        ]
        for s in cases:
            accepted, terminated = self._probe(compiled[ver], s)
            assert not (accepted and terminated), f"[{ver}] should reject: {s!r}"

    @pytest.mark.parametrize("ver", ["glm45", "glm47"])
    def test_key_value_type_coupling(self, compiled, ver):
        """The same value must be judged per key: free text is fine for the
        string key but rejected for the enum key of the same tool."""
        free_text = "some free text"
        ok = self._call(ver, "get_weather", [("location", free_text)])
        bad = self._call(ver, "get_weather", [("unit", free_text)])
        assert self._probe(compiled[ver], ok) == (True, True)
        accepted, terminated = self._probe(compiled[ver], bad)
        assert not (accepted and terminated)

    def test_duplicate_tool_names_with_different_args(self):
        """Tools may share a name with different schemas; both variants of the
        arguments must be accepted."""
        functions = [
            _Function(
                "dup", {"type": "object", "properties": {"x": {"type": "string"}}}
            ),
            _Function(
                "dup", {"type": "object", "properties": {"y": {"type": "number"}}}
            ),
        ]
        grammar = build_grammar("glm47", functions)
        tokenizer_info = xgr.TokenizerInfo([])
        compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
            grammar, root_rule_name="tool_call_blocks"
        )
        for s in [
            self._call("glm47", "dup", [("x", "hello")]),
            self._call("glm47", "dup", [("y", "1.5")]),
        ]:
            assert self._probe(compiled, s) == (True, True), f"should accept: {s!r}"
        accepted, terminated = self._probe(
            compiled, self._call("glm47", "dup", [("x", "hello"), ("y", "1.5")])
        )
        # mixing args of the two variants in one call is not accepted
        assert not (accepted and terminated)

    def test_string_value_excludes_special_tokens(self):
        """Free-text values still reject embedded special tokens."""
        s = self._call("glm47", "get_weather", [("location", "a<tool_call>b")])
        grammar = build_grammar("glm47")
        tokenizer_info = xgr.TokenizerInfo([])
        compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
            grammar, root_rule_name="tool_call_blocks"
        )
        accepted, terminated = self._probe(compiled, s)
        assert not (accepted and terminated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
