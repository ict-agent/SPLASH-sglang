"""
Unit tests for ebnf_utils.py
"""

import pytest
from .ebnf_utils import any_string_exclude

# Try to import xgrammar, skip tests if not available
try:
    import xgrammar as xgr
    HAS_XGRAMMAR = True
except ImportError:
    HAS_XGRAMMAR = False


class TestAnyStringExclude:
    """Tests for any_string_exclude function."""

    def test_empty_negative_strings(self):
        """Empty list should match anything."""
        result = any_string_exclude("rule", [])
        assert result == ["rule ::= [^]*"]

    def test_single_string_basic_structure(self):
        """Test basic structure for single string exclusion."""
        result = any_string_exclude("xml_text", ["ab"])

        # Should have root rule and state rules
        assert any("xml_text ::=" in line for line in result)
        # Should have at least 2 state rules (root + one intermediate)
        assert len(result) >= 2

    def test_deterministic_hash(self):
        """Hash should be deterministic for same input."""
        result1 = any_string_exclude("rule", ["abc", "def"])
        result2 = any_string_exclude("rule", ["abc", "def"])
        assert result1 == result2

        # Different order should produce same hash (sorted)
        result3 = any_string_exclude("rule", ["def", "abc"])
        assert result1 == result3

    def test_hash_differs_for_different_input(self):
        """Different inputs should produce different hashes."""
        result1 = any_string_exclude("rule", ["abc"])
        result2 = any_string_exclude("rule", ["abd"])

        # Extract hash from state names (16 hex characters)
        import re
        result1_str = "\n".join(result1)
        result2_str = "\n".join(result2)
        hashes1 = set(re.findall(r's_([a-f0-9]{16})_\d+', result1_str))
        hashes2 = set(re.findall(r's_([a-f0-9]{16})_\d+', result2_str))

        assert hashes1 != hashes2

    def test_single_char_exclusion(self):
        """Test excluding a single character string."""
        result = any_string_exclude("rule", ["x"])
        result_str = "\n".join(result)
        # Should exclude 'x' at root level
        assert "[^x]" in result_str or '[^"x"]' in result_str or '"x"' not in result[1]

    def test_special_chars_escaped_in_char_class(self):
        """Special characters should be escaped in character class."""
        result = any_string_exclude("rule", ["a]b"])
        result_str = "\n".join(result)
        # ] should be escaped as \]
        assert "\\]" in result_str

    def test_special_chars_escaped_in_string(self):
        """Special characters should be escaped in string literals."""
        result = any_string_exclude("rule", ['a"b'])
        result_str = "\n".join(result)
        # " should be escaped as \"
        assert '\\"' in result_str

    def test_newline_escaped(self):
        """Newline should be escaped."""
        result = any_string_exclude("rule", ["a\nb"])
        result_str = "\n".join(result)
        assert "\\n" in result_str

    def test_all_states_have_empty_alternative(self):
        """All non-end states should have empty alternative to allow termination.

        This is critical for:
        1. Matching strings shorter than the pattern
        2. Allowing grammar composition like: any_string_exclude(...) "abc"
        """
        result = any_string_exclude("rule", ["abc"])

        # All state rules (except root declaration) should have empty alternative
        for line in result[1:]:
            if "::=" in line:
                # Should contain empty alternative ""
                assert '""' in line, f"Missing empty alternative in: {line}"

    def test_multiple_strings_with_common_prefix(self):
        """Test multiple strings sharing a common prefix."""
        result = any_string_exclude("rule", ["abc", "abd"])

        # Should have states for: root, a, ab (shared), then diverge
        # At least 4 rules (root + 3 states)
        assert len(result) >= 4

    def test_overlapping_patterns(self):
        """Test overlapping patterns (one is prefix of another)."""
        result = any_string_exclude("rule", ["ab", "abc"])

        # Both patterns should be handled
        assert result is not None
        assert len(result) >= 2

    def test_arg_value_pattern(self):
        """Test the original use case: </arg_value>."""
        result = any_string_exclude("xml_text", ["</arg_value>"])

        # Should have 13 rules:
        # 1 root rule (xml_text ::= s_xxx_0)
        # 12 state rules (s_xxx_0 through s_xxx_11, end state s_xxx_12 has no rule)
        assert len(result) == 13

        # Root rule should reference state 0 directly (recursive structure)
        assert "::= s_" in result[0]
        assert "*" not in result[0]  # No Kleene star in root

        # State 0 should have transitions including '<' to state 1
        state_0_line = result[1]
        assert '"<"' in state_0_line

    def test_hash_format(self):
        """Hash should be 16 hex characters."""
        result = any_string_exclude("rule", ["test"])
        import re
        result_str = "\n".join(result)
        matches = re.findall(r's_([a-f0-9]+)_\d+', result_str)
        assert all(len(m) == 16 for m in matches)

    def test_duplicate_strings_handled(self):
        """Duplicate strings should be deduplicated."""
        result1 = any_string_exclude("rule", ["abc", "abc", "abc"])
        result2 = any_string_exclude("rule", ["abc"])
        assert result1 == result2

    def test_empty_string_in_list_ignored(self):
        """Empty strings in the list should be ignored."""
        result1 = any_string_exclude("rule", ["abc", "", ""])
        result2 = any_string_exclude("rule", ["abc"])
        assert result1 == result2


class TestWithXGrammar:
    """Integration tests with xgrammar library (if available)."""

    @pytest.fixture
    def grammar_compiler(self):
        """Create xgrammar compiler if available."""
        try:
            import xgrammar as xgr
            tokenizer_info = xgr.TokenizerInfo([])
            return xgr.GrammarCompiler(tokenizer_info)
        except ImportError:
            pytest.skip("xgrammar not installed")

    def test_grammar_compiles(self, grammar_compiler):
        """Generated grammar should compile without errors."""
        grammar = "\n".join(any_string_exclude("xml_text", ["</arg_value>"]))
        # Should not raise
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="xml_text")
        assert compiled is not None

    def test_grammar_rejects_excluded_string(self, grammar_compiler):
        """Grammar should reject strings containing excluded pattern."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("xml_text", ["</arg_value>"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="xml_text")
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

        # This string contains </arg_value>, should fail at some point
        test_string = "hello</arg_value>world"

        accepted = True
        for c in test_string:
            if not matcher.accept_string(c):
                accepted = False
                break

        assert not accepted, "Should reject string containing </arg_value>"

    def test_grammar_accepts_safe_string(self, grammar_compiler):
        """Grammar should accept strings not containing excluded pattern."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("xml_text", ["</arg_value>"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="xml_text")
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

        # This string does NOT contain </arg_value>
        test_string = "hello</arg_valu>world"

        for c in test_string:
            assert matcher.accept_string(c), f"Should accept char '{c}'"

    def test_grammar_accepts_partial_pattern(self, grammar_compiler):
        """Grammar should accept strings with partial pattern match."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("xml_text", ["</arg_value>"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="xml_text")
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

        # Partial patterns that don't complete
        test_strings = [
            "</arg_value",  # Missing >
            "</arg_valu",   # Missing e>
            "</arg",        # Incomplete
            "<",            # Just start
        ]

        for test_string in test_strings:
            matcher.reset()
            all_accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    all_accepted = False
                    break
            assert all_accepted, f"Should accept partial pattern: {test_string!r}"

    def test_grammar_handles_repeated_start_char(self, grammar_compiler):
        """Grammar should handle repeated pattern start characters."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("xml_text", ["</arg_value>"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="xml_text")
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

        # Multiple < characters, but no complete pattern
        test_string = "<<<</arg</arg_valu"

        for c in test_string:
            assert matcher.accept_string(c), f"Should accept char '{c}'"

    def test_grammar_rejects_pattern_after_partial(self, grammar_compiler):
        """Grammar should reject when pattern completes after partial match."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("xml_text", ["</arg_value>"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="xml_text")
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

        # </a< then </arg_value> - the second one should be rejected
        test_string = "</a</arg_value>"

        accepted = True
        for c in test_string:
            if not matcher.accept_string(c):
                accepted = False
                break

        assert not accepted, "Should reject: contains </arg_value>"

    def test_multiple_patterns(self, grammar_compiler):
        """Test grammar with multiple exclusion patterns."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("text", ["abc", "xyz"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

        # Should reject strings containing either pattern
        for test_string in ["hello abc world", "test xyz test"]:
            matcher.reset()
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert not accepted, f"Should reject: {test_string!r}"

        # Should accept strings without either pattern
        matcher.reset()
        safe_string = "hello ab xy world"
        for c in safe_string:
            assert matcher.accept_string(c)

    def test_nested_angle_brackets(self, grammar_compiler):
        """Test with nested angle brackets like <New York>."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("xml_text", ["</arg_value>"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="xml_text")
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

        # Should accept <New York> since it's not </arg_value>
        test_string = "<New York>"

        for c in test_string:
            assert matcher.accept_string(c), f"Should accept char '{c}' in '<New York>'"

    def test_common_prefix_patterns(self, grammar_compiler):
        """Test multiple patterns with common prefix."""
        import xgrammar as xgr

        # </arg_value> and </arg_key> share prefix "</arg_"
        grammar = "\n".join(any_string_exclude("text", ["</arg_value>", "</arg_key>"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Should reject - complete patterns
            ("</arg_value>", False),
            ("</arg_key>", False),
            ("hello</arg_value>world", False),
            ("hello</arg_key>world", False),
            # Should accept - partial or different
            ("</arg_", True),
            ("</arg_v", True),
            ("</arg_k", True),
            ("</arg_other>", True),
            ("</arg>", True),
            ("</argument>", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_one_pattern_is_prefix_of_another(self, grammar_compiler):
        """Test when one pattern is a prefix of another (e.g., 'ab' and 'abc')."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("text", ["ab", "abc"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Both should be rejected
            ("ab", False),
            ("abc", False),
            ("xaby", False),  # contains "ab"
            ("xabcy", False),  # contains "abc" (and "ab")
            # Should accept
            ("a", True),
            ("ac", True),
            ("ba", True),
            ("cab", False),  # contains "ab"
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_interleaved_pattern_attempts(self, grammar_compiler):
        """Test complex interleaving where partial matches overlap."""
        import xgrammar as xgr

        # Patterns: "abcd" and "abef"
        grammar = "\n".join(any_string_exclude("text", ["abcd", "abef"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Reject complete patterns
            ("abcd", False),
            ("abef", False),
            # Accept partial matches
            ("abc", True),
            ("abe", True),
            ("ab", True),
            # Tricky: "abab" - starts "ab", then "ab" again
            ("abab", True),
            # "ababcd" contains "abcd" starting at position 2
            ("ababcd", False),
            # "abcef" - starts like "abcd" but diverges, no complete match
            ("abcef", True),
            # "abecd" - starts like "abef" but diverges
            ("abecd", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_repeated_pattern_start(self, grammar_compiler):
        """Test when pattern start character repeats many times."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("text", ["aab"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            ("aab", False),
            ("aaab", False),  # "aab" at position 1
            ("aaaab", False),  # "aab" at position 2
            ("aaaaab", False),  # "aab" at position 3
            # Should accept
            ("aa", True),
            ("aaa", True),
            ("aaaa", True),
            ("ab", True),
            ("aaba", False),  # contains "aab"
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_xml_multiple_end_tags(self, grammar_compiler):
        """Test realistic XML scenario with multiple end tags to exclude."""
        import xgrammar as xgr

        # Exclude multiple XML end tags
        grammar = "\n".join(any_string_exclude("content", [
            "</arg_value>",
            "</parameter>",
            "</function>",
        ]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="content")

        test_cases = [
            # Reject all end tags
            ("</arg_value>", False),
            ("</parameter>", False),
            ("</function>", False),
            ("text</arg_value>more", False),
            ("text</parameter>more", False),
            ("text</function>more", False),
            # Accept partial matches
            ("</arg_", True),
            ("</param", True),
            ("</func", True),
            # Accept valid content with angle brackets
            ("<value>", True),
            ("<New York>", True),
            ("</other>", True),
            ("</arg>", True),
            ("</para>", True),
            # Complex valid content
            ("<city>New York</city>", True),
            ("temperature: <0°C", True),
            ("a]>b", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_overlapping_pattern_detection(self, grammar_compiler):
        """Test detection when patterns overlap in the input."""
        import xgrammar as xgr

        # Pattern "aba" can overlap with itself: "ababa" contains "aba" twice
        grammar = "\n".join(any_string_exclude("text", ["aba"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            ("aba", False),
            ("ababa", False),  # "aba" at pos 0 and pos 2
            ("abaaba", False),  # "aba" at pos 0 and pos 3
            ("abab", False),  # "aba" at pos 0 (a,b,a,b -> first 3 chars are "aba")
            # Should accept
            ("ab", True),
            ("aa", True),
            ("bab", True),
            ("abba", True),  # no "aba" substring
            ("aabb", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_three_patterns_with_shared_prefix(self, grammar_compiler):
        """Test three patterns sharing the same prefix."""
        import xgrammar as xgr

        # All start with "pre"
        grammar = "\n".join(any_string_exclude("text", ["prefix", "prepare", "prevent"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Reject all
            ("prefix", False),
            ("prepare", False),
            ("prevent", False),
            ("the prefix is", False),
            ("to prepare for", False),
            ("to prevent it", False),
            # Accept partial
            ("pre", True),
            ("pref", True),
            ("prep", True),
            ("prev", True),
            ("prefi", True),
            ("prepar", True),
            ("preven", True),
            # Accept different words
            ("present", True),
            ("pressure", True),
            ("prediction", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_patterns_with_special_characters(self, grammar_compiler):
        """Test patterns containing special regex/EBNF characters."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("text", ["a]b", "x[y", 'p"q']))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            ("a]b", False),
            ("x[y", False),
            ('p"q', False),
            # Accept without complete patterns
            ("a]", True),
            ("x[", True),
            ('p"', True),
            ("]b", True),
            ("[y", True),
            ('"q', True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_long_common_prefix_diverging_at_end(self, grammar_compiler):
        """Test patterns with very long common prefix, diverging only at the end."""
        import xgrammar as xgr

        # Only differ in last character
        grammar = "\n".join(any_string_exclude("text", ["abcdefghX", "abcdefghY", "abcdefghZ"]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Reject all three
            ("abcdefghX", False),
            ("abcdefghY", False),
            ("abcdefghZ", False),
            # Accept common prefix
            ("abcdefgh", True),
            ("abcdefg", True),
            # Accept with different ending
            ("abcdefghA", True),
            ("abcdefghW", True),
            ("abcdefgh1", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"


    def test_special_chars_with_common_prefix(self, grammar_compiler):
        """Test patterns with special characters AND common prefix."""
        import xgrammar as xgr

        # All start with "end" but have different special char endings
        grammar = "\n".join(any_string_exclude("text", [
            'end"here',   # double quote
            "end'there",  # single quote
            "end!now",    # exclamation
            "end?what",   # question mark
            "end.stop",   # period
            "end*star",   # asterisk
        ]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Reject all complete patterns
            ('end"here', False),
            ("end'there", False),
            ("end!now", False),
            ("end?what", False),
            ("end.stop", False),
            ("end*star", False),
            # Reject patterns embedded in text
            ('the end"here is bad', False),
            ("the end'there is bad", False),
            ("the end!now is bad", False),
            # Accept partial matches (common prefix)
            ("end", True),
            ("end!", True),
            ("end?", True),
            ("end.", True),
            ("end*", True),
            ('end"', True),
            ("end'", True),
            # Accept patterns that diverge before completion
            ('end"her', True),   # missing 'e'
            ("end'ther", True),  # missing 'e'
            ("end!no", True),    # missing 'w'
            ("end?wha", True),   # missing 't'
            ("end.sto", True),   # missing 'p'
            ("end*sta", True),   # missing 'r'
            # Accept different endings
            ('end"other', True),
            ("end'other", True),
            ("end!other", True),
            ("end?other", True),
            ("end.other", True),
            ("end*other", True),
            # Accept text without any pattern
            ("ending", True),
            ("endless", True),
            ("the end", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_json_like_patterns_with_escapes(self, grammar_compiler):
        """Test patterns that look like JSON with quotes and escapes."""
        import xgrammar as xgr

        # Patterns with quotes and backslashes
        grammar = "\n".join(any_string_exclude("text", [
            '{"error"}',
            "{'error'}",
            '{\\error\\}',
        ]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Reject complete patterns
            ('{"error"}', False),
            ("{'error'}", False),
            ('{\\error\\}', False),
            # Accept partial
            ('{"error"', True),
            ("{'error'", True),
            ('{\\error\\', True),
            ('{"error', True),
            ("{'error", True),
            # Accept different content
            ('{"success"}', True),
            ("{'success'}", True),
            ('{"err"}', True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_regex_metachar_patterns(self, grammar_compiler):
        """Test patterns containing regex metacharacters."""
        import xgrammar as xgr

        # Patterns with ., *, +, ?, ^, $, etc.
        grammar = "\n".join(any_string_exclude("text", [
            "a]b",
            "x[y",
            "p.q",
            "m*n",
            "r+s",
            "u?v",
            "i^j",
            "k$l",
            "(a)",
            "{b}",
        ]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Reject all patterns
            ("a]b", False),
            ("x[y", False),
            ("p.q", False),
            ("m*n", False),
            ("r+s", False),
            ("u?v", False),
            ("i^j", False),
            ("k$l", False),
            ("(a)", False),
            ("{b}", False),
            # Accept partial
            ("a]", True),
            ("x[", True),
            ("p.", True),
            ("m*", True),
            ("r+", True),
            ("u?", True),
            ("i^", True),
            ("k$", True),
            ("(a", True),
            ("{b", True),
            # Accept different content
            ("a]c", True),
            ("x[z", True),
            ("p.r", True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_xml_tags_with_attributes(self, grammar_compiler):
        """Test XML-like patterns with quotes in attributes."""
        import xgrammar as xgr

        # XML end tags and tags with attributes
        grammar = "\n".join(any_string_exclude("text", [
            '</value>',
            '</data type="string">',
            "</data type='string'>",
            '</item id="1">',
        ]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Reject complete patterns
            ('</value>', False),
            ('</data type="string">', False),
            ("</data type='string'>", False),
            ('</item id="1">', False),
            # Accept partial (common prefix </...)
            ('</val', True),
            ('</data', True),
            ('</data type="', True),
            ('</data type="string', True),
            ('</item', True),
            ('</item id="', True),
            # Accept different endings
            ('</value2>', True),
            ('</data type="int">', True),
            ('</item id="2">', True),
            # Accept other content
            ('<value>', True),
            ('<data>', True),
            ('text with </other> tag', True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"

    def test_mixed_special_chars_complex(self, grammar_compiler):
        """Complex test with multiple special chars, quotes, and common prefixes."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("text", [
            '>>>END<<<',
            '>>>STOP<<<',
            '>>>QUIT<<<',
            '"""END"""',
            "'''END'''",
            '###END###',
            '***END***',
            '???END???',
            '!!!END!!!',
        ]))
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="text")

        test_cases = [
            # Reject all patterns
            ('>>>END<<<', False),
            ('>>>STOP<<<', False),
            ('>>>QUIT<<<', False),
            ('"""END"""', False),
            ("'''END'''", False),
            ('###END###', False),
            ('***END***', False),
            ('???END???', False),
            ('!!!END!!!', False),
            # In context
            ('text >>>END<<< more', False),
            ('text """END""" more', False),
            # Accept common prefixes
            ('>>>', True),
            ('>>>E', True),
            ('>>>EN', True),
            ('>>>END', True),
            ('>>>END<', True),
            ('>>>END<<', True),
            ('"""', True),
            ('"""E', True),
            ('"""EN', True),
            ('"""END', True),
            ('"""END"', True),
            ('"""END""', True),
            # Accept divergent patterns
            ('>>>OTHER<<<', True),
            ('>>>END>>>', True),  # different ending
            ('"""OTHER"""', True),
            ("'''OTHER'''", True),
            ('###OTHER###', True),
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            assert accepted == expected, f"{test_string!r}: got {accepted}, expected {expected}"


@pytest.mark.skipif(not HAS_XGRAMMAR, reason="xgrammar not installed")
class TestPrefixTermination:
    """Tests specifically for prefix substring termination.

    This is the critical fix: strings shorter than the negative pattern
    should be able to both accept AND terminate. This enables grammar
    composition like: any_string_exclude("text", ["abc"]) "abc"
    """

    def test_prefix_can_terminate_single_char(self):
        """Single character strings should terminate."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("root", ["abc"]))
        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        # Test various single chars
        for char in ["x", "a", "b", "c", "z"]:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            assert matcher.accept_string(char), f"Should accept '{char}'"
            assert matcher.is_terminated(), f"'{char}' should be able to terminate"

    def test_prefix_can_terminate_partial_pattern(self):
        """Partial patterns (prefixes of negative string) should terminate."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("root", ["abcdef"]))
        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        # All prefixes of "abcdef" should terminate
        prefixes = ["", "a", "ab", "abc", "abcd", "abcde"]
        for prefix in prefixes:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            for c in prefix:
                assert matcher.accept_string(c), f"Should accept char in '{prefix}'"
            assert matcher.is_terminated(), f"'{prefix}' should be able to terminate"

    def test_prefix_cannot_terminate_complete_pattern(self):
        """Complete pattern should NOT be accepted (reject before termination)."""
        import xgrammar as xgr

        grammar = "\n".join(any_string_exclude("root", ["abc"]))
        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert matcher.accept_string("a")
        assert matcher.accept_string("b")
        # 'c' should be rejected because it completes "abc"
        assert not matcher.accept_string("c"), "'abc' should be rejected"

    def test_composition_text_then_literal(self):
        """Test composition: any_string_exclude(...) followed by literal.

        Pattern: text "abc" where text excludes "abc"
        Input "aabc" should match as "a" (from text) + "abc" (literal)
        """
        import xgrammar as xgr

        rules = any_string_exclude("text", ["abc"])
        grammar = "\n".join(rules) + '\nroot ::= text "abc"'

        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        # "aabc" should match: "a" from text, then "abc" literal
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        for c in "aabc":
            assert matcher.accept_string(c), f"Should accept '{c}' in 'aabc'"
        assert matcher.is_terminated(), "'aabc' should terminate"

    def test_composition_empty_text_then_literal(self):
        """Test composition with empty text portion.

        Pattern: text "abc" where text excludes "abc"
        Input "abc" should match as "" (empty text) + "abc" (literal)
        """
        import xgrammar as xgr

        rules = any_string_exclude("text", ["abc"])
        grammar = "\n".join(rules) + '\nroot ::= text "abc"'

        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        # "abc" should match: "" from text, then "abc" literal
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        for c in "abc":
            assert matcher.accept_string(c), f"Should accept '{c}' in 'abc'"
        assert matcher.is_terminated(), "'abc' should terminate"

    def test_composition_longer_text_then_literal(self):
        """Test composition with longer text portion."""
        import xgrammar as xgr

        rules = any_string_exclude("text", ["abc"])
        grammar = "\n".join(rules) + '\nroot ::= text "abc"'

        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        # "xyzabc" should match: "xyz" from text, then "abc" literal
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        for c in "xyzabc":
            assert matcher.accept_string(c), f"Should accept '{c}' in 'xyzabc'"
        assert matcher.is_terminated(), "'xyzabc' should terminate"

    def test_composition_partial_pattern_in_text(self):
        """Test composition where text contains partial pattern."""
        import xgrammar as xgr

        rules = any_string_exclude("text", ["abc"])
        grammar = "\n".join(rules) + '\nroot ::= text "abc"'

        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        # "ababc" should match: "ab" from text, then "abc" literal
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        for c in "ababc":
            assert matcher.accept_string(c), f"Should accept '{c}' in 'ababc'"
        assert matcher.is_terminated(), "'ababc' should terminate"

    def test_composition_multiple_patterns(self):
        """Test composition with multiple excluded patterns."""
        import xgrammar as xgr

        rules = any_string_exclude("text", ["abc", "def"])
        grammar = "\n".join(rules) + '\nroot ::= text "abc"'

        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        test_cases = [
            ("abc", True),      # empty text + "abc"
            ("xabc", True),     # "x" + "abc"
            ("xyabc", True),    # "xy" + "abc"
            ("deabc", True),    # "de" (partial "def") + "abc"
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            is_term = matcher.is_terminated() if accepted else False
            result = accepted and is_term
            assert result == expected, f"{test_string!r}: got accepted={accepted}, terminated={is_term}, expected={expected}"

    def test_xml_tag_composition(self):
        """Test realistic XML use case: content followed by end tag."""
        import xgrammar as xgr

        rules = any_string_exclude("content", ["</value>"])
        grammar = "\n".join(rules) + '\nroot ::= content "</value>"'

        tokenizer_info = xgr.TokenizerInfo([])
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

        test_cases = [
            ("</value>", True),                    # empty content
            ("hello</value>", True),               # simple content
            ("<nested></value>", True),            # content with angle brackets
            ("</val</value>", True),               # partial tag in content
            ("</valu</value>", True),              # almost complete tag in content
        ]

        for test_string, expected in test_cases:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = True
            for c in test_string:
                if not matcher.accept_string(c):
                    accepted = False
                    break
            is_term = matcher.is_terminated() if accepted else False
            result = accepted and is_term
            assert result == expected, f"{test_string!r}: got accepted={accepted}, terminated={is_term}, expected={expected}"

    def test_short_strings_various_patterns(self):
        """Test that various short strings can terminate for different patterns."""
        import xgrammar as xgr

        patterns_and_short_strings = [
            (["hello"], ["h", "he", "hel", "hell", "x", "xy"]),
            (["</tag>"], ["<", "</", "</t", "</ta", "</tag", "x", "<x"]),
            (["abc", "xyz"], ["a", "ab", "x", "xy", "ax", "abx"]),
        ]

        for patterns, short_strings in patterns_and_short_strings:
            rules = any_string_exclude("root", patterns)
            grammar = "\n".join(rules)

            tokenizer_info = xgr.TokenizerInfo([])
            grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
            compiled = grammar_compiler.compile_grammar(grammar, root_rule_name="root")

            for s in short_strings:
                matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
                accepted = True
                for c in s:
                    if not matcher.accept_string(c):
                        accepted = False
                        break
                assert accepted, f"patterns={patterns}, string={s!r} should be accepted"
                assert matcher.is_terminated(), f"patterns={patterns}, string={s!r} should terminate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
