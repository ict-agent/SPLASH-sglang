from typing import Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING, Literal
import hashlib
from .ebnf_utils import any_string_exclude

if TYPE_CHECKING:
    from .schema import SpecialTokenConfig


@runtime_checkable
class FunctionProtocol(Protocol):
    name: str
    parameters: Optional[object]


# Base primitive grammar rules for XML format
XML_GRAMMAR_RULES = [
    'basic_string ::= (([\\"] basic_string_1 [\\"]))',
    'basic_string_1 ::= "" | [^"\\\\\\x00-\\x1F] basic_string_1 | "\\\\" escape basic_string_1',
    'escape ::= ["\\\\//bfnrt] | "u" [A-Fa-f0-9]{4}',
    'basic_integer ::= "-"? ("0" | [1-9] [0-9]*) ".0"?',
    'basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
    'basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"',
    'basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"',
    'ws ::= [ \\n\\t]*',
    'basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object',
    'basic_boolean ::= "true" | "false"',
    'basic_null ::= "null"',
]

TYPE_MAPPING = {
    "string": "text_without_special_tokens",
    "number": "basic_number",
    "integer": "basic_number",
    "boolean": "basic_boolean",
    "null": "basic_null",
    "array": "basic_array",
    "object": "basic_object",
}


def _hash_name(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def _get_value_rule(prop: dict) -> str:
    if "enum" in prop:
        return _handle_enum(prop)
    if "type" in prop:
        return _handle_type(prop)
    return "text_without_special_tokens"


def _escape_ebnf_string(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    s = s.replace("\r", "\\r")
    return s


def _handle_enum(prop: dict) -> str:
    enum_values = prop["enum"]
    prop_type = prop.get("type", "string")

    def format_enum_val(v: Any) -> str:
        if prop_type == "boolean":
            return '"true"' if v else '"false"'
        if prop_type == "string":
            return f'"{_escape_ebnf_string(v)}"'
        return f'"{v}"'

    formatted_values = [format_enum_val(v) for v in enum_values]
    enum_rule = " | ".join(formatted_values)
    return f"({enum_rule})" if len(formatted_values) > 1 else enum_rule


def _handle_type(prop: dict) -> str:
    prop_type = prop["type"]
    if isinstance(prop_type, list):
        type_rules = [TYPE_MAPPING.get(t, "text_without_special_tokens") for t in prop_type]
        return " | ".join(type_rules) if type_rules else "text_without_special_tokens"
    return TYPE_MAPPING.get(prop_type, "text_without_special_tokens")


def build_tool_call_rules(
    non_terminal_name: str,
    functions: list[FunctionProtocol],
    special_tokens: "SpecialTokenConfig",
    chat_template_version: Literal["glm45", "glm47"],
) -> list[str]:
    """
    Build EBNF rules for XML-style tool calls.

    Args:
        non_terminal_name: Name for the root non-terminal of tool calls
        functions: List of Function objects
        special_tokens: Token configuration for tool call formatting

    Returns:
        List of EBNF rule strings
    """
    # Extra spaces
    if chat_template_version == "glm45":
        extra_seperator = '"\\n"'
    elif chat_template_version == "glm47":
        extra_seperator = ''
    else:
        raise NotImplementedError(f"Unsupported chat_template_version: {chat_template_version}")

    rules = [
        # Root rule: zero or more tool calls, each preceded by newline
        f'{non_terminal_name} ::= ( {extra_seperator} tool_call_unit )*',
        f'tool_call_unit ::= "{special_tokens.begin_of_tool_call}" single_tool_call "{special_tokens.end_of_tool_call}"',
    ]

    # Union of all tool calls
    # NOTE: functions may share the same name but have different arguments.
    #  This is rather unusual / abnormal, but we handle it by hashing the name with the index to ensure uniqueness.
    tool_alternatives = " | ".join(
        f"call_{_hash_name(func.name + str(function_index))}" for function_index, func in enumerate(functions)
    )
    rules.append(f"single_tool_call ::= {tool_alternatives}")

    # Key-value format template
    # Wrap {valrule} in parentheses to ensure correct precedence when valrule contains alternatives (e.g., "text | null")
    kv_template = f'"{special_tokens.begin_of_key}{{key}}{special_tokens.end_of_key}" {extra_seperator} "{special_tokens.begin_of_value}" ({{valrule}}) "{special_tokens.end_of_value}"'
    kv_separator = extra_seperator

    # Build rules for each function
    for function_index, func in enumerate(functions):
        tool_name = func.name
        namehash = _hash_name(func.name + str(function_index))
        params = func.parameters or {}
        properties = params.get("properties", {})

        prop_kv_pairs = {}

        for prop_name, prop_schema in properties.items():
            value_rule = _get_value_rule(prop_schema)
            pair = kv_template.format(key=prop_name, valrule=value_rule)
            prop_kv_pairs[prop_name] = pair

        # 所有参数都用 S* 形式，允许任意顺序任意次
        all_props = list(properties.keys())

        if all_props:
            all_choices = " | ".join(prop_kv_pairs[k] for k in all_props)
            # ( any_param ( separator any_param )* )?
            arguments_rule = f"( ( {all_choices} ) ( {kv_separator} ( {all_choices} ) )* )?"
        else:
            arguments_rule = '""'

        rules.append(f'call_{namehash} ::= "{tool_name}" {extra_seperator} ( arguments_{namehash} {extra_seperator} )?')
        rules.append(f"arguments_{namehash} ::= {arguments_rule}")

    rules.extend(XML_GRAMMAR_RULES)
    return rules


if __name__ == "__main__":
    from pydantic import BaseModel, Field
    from .schema import SpecialTokenConfig
    import xgrammar as xgr

    checks = [
        (
            "glm45",
            '\n<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value><NYK>\n\n\nssagsfgas</arg_value>\n<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n</tool_call>',
        ),
        (
            "glm47",
            '<tool_call>get_weather<arg_key>location</arg_key><arg_value><NYK>\n\n\nssagsfgas</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>',
        )
    ]


    class Function(BaseModel):
        description: Optional[str] = Field(default=None)
        name: str
        parameters: Optional[object] = None
        strict: bool = False


    test_functions = [
        Function(
            name="get_weather",
            description="Get the current weather for a given location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        ),
        Function(
            name="calculate_sum",
            description="Calculate the sum of two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        ),
    ]

    special_tokens = SpecialTokenConfig()

    for chat_template_version, string_to_match in checks:
        rules = build_tool_call_rules(
            non_terminal_name="tool_call_blocks",
            functions=test_functions,
            special_tokens=special_tokens,
            chat_template_version=chat_template_version,
        )
        rules.extend(any_string_exclude("text_without_special_tokens", special_tokens.all_special_tokens()))

        ebnf_grammar = "\n".join(rules)
        print("Generated EBNF Grammar:")
        print(ebnf_grammar)
        print()

        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled_grammar = grammar_compiler.compile_grammar(ebnf_grammar, root_rule_name="tool_call_blocks")

        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        print("Matching string:", string_to_match)
        print()

        all_accepted = True
        for c_index, c in enumerate(string_to_match):
            if not matcher.accept_string(c):
                print(f"REJECTED at char {c_index}: {string_to_match[:c_index + 1]!r}")
                all_accepted = False
                break

        if all_accepted:
            print("All characters accepted!")
            print(f"Is terminated: {matcher.is_terminated()}")
