import unittest

from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestEBNFComposer(unittest.TestCase):
    def test_boolean_enum_values_are_quoted_terminals(self):
        rule = EBNFComposer._handle_enum(
            {"type": "boolean", "enum": [True, False]}, "xml"
        )
        self.assertEqual(rule, '("true" | "false")')

    def test_string_enum_values_are_escaped_for_xml(self):
        rule = EBNFComposer._handle_enum(
            {"type": "string", "enum": ["a\\b", 'x"y', "line\nend"]}, "xml"
        )
        self.assertEqual(rule, '("a\\\\b" | "x\\"y" | "line\\nend")')

    def test_string_enum_values_are_json_quoted_and_escaped_for_json(self):
        rule = EBNFComposer._handle_enum(
            {"type": "string", "enum": ['x"y']}, "json"
        )
        self.assertEqual(rule, '"\\"x\\\\\\"y\\""')


if __name__ == "__main__":
    unittest.main()
