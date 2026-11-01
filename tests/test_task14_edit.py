import pytest

from selfplay_opsd.mutations import parse_action, replace_function_body


def test_body_replacement_preserves_other_functions_and_signature():
    source = "def f(x):\n    return x + 1\n\ndef g(y):\n    return y * 2\n"
    result = replace_function_body(source, "f", "return x - 1")
    assert result == "def f(x):\n    return x - 1\n\ndef g(y):\n    return y * 2\n"


@pytest.mark.parametrize(
    ("source", "name", "body"),
    [
        ("def f(x):\n    return x\n", "missing", "return x"),
        ("def f(x):\n    return x\ndef f(y):\n    return y\n", "f", "return 1"),
        ("def f(x):\n    return x\n", "f", "    return x"),
        ("def f(x):\n    return x\n", "f", "return ("),
    ],
)
def test_body_replacement_rejects_invalid_actions(source, name, body):
    with pytest.raises((ValueError, SyntaxError)):
        replace_function_body(source, name, body)


def test_protocol_allows_only_selected_edit_action():
    body = '{"action":"replace_function_body","function":"f","body":"return x"}'
    diff = '{"action":"apply_patch","patch":"example"}'
    assert parse_action(body, "replace_function_body")["function"] == "f"
    with pytest.raises(ValueError):
        parse_action(body)
    with pytest.raises(ValueError):
        parse_action(diff, "replace_function_body")


def test_replacement_does_not_add_imports():
    with pytest.raises(ValueError, match="preserve imports"):
        replace_function_body("def f(x):\n    return x\n", "f", "import math\nreturn x")
