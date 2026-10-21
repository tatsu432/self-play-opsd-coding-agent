import json

import pytest

from selfplay_opsd.codec import decode, encode
from selfplay_opsd.config import PROFILES, load_config
from selfplay_opsd.core import Case
from selfplay_opsd.mutations import (
    apply_patch,
    fixed_mutation,
    make_patch,
    parse_action,
    validate_mutation,
    validate_source,
)
from selfplay_opsd.sandbox import bounded_process, compare

SOURCE = "def add(a, b):\n    return a + b\n"


def test_every_profile_and_scientific_resource_boundary():
    configs = [load_config(study_id="test", hardware_profile=p) for p in PROFILES]
    gpu = [c for c in configs if c.hardware.gpu_count]
    assert len({c.scientific_hash for c in gpu}) == 1
    assert len({c.hardware.gpu_count for c in gpu}) == 4
    with pytest.raises(ValueError):
        load_config(study_id="test", unexpected=True)
    cfg = load_config(study_id="test")
    other = cfg.model_copy(deep=True)
    other.objective.sd_lambda = 0.2
    assert cfg.scientific_hash != other.scientific_hash


def test_mutation_roundtrip_and_bounds():
    mutation = fixed_mutation(SOURCE, 2026)
    assert mutation.source != SOURCE
    assert apply_patch(mutation.source, mutation.inverse_patch) == SOURCE
    assert mutation == fixed_mutation(SOURCE, 2026)
    with pytest.raises(ValueError, match="API"):
        validate_mutation(SOURCE, make_patch(SOURCE, "def add(a):\n    return a + 1\n"))
    with pytest.raises(ValueError):
        apply_patch(
            SOURCE,
            make_patch(SOURCE, SOURCE.replace("+", "-")).replace(
                "src/solution.py", "tests/test_public.py"
            ),
        )
    with pytest.raises(ValueError):
        apply_patch(
            SOURCE,
            make_patch(SOURCE, SOURCE.replace("+", "-")).replace("return a + b", "return b + a"),
        )


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f():\n return os.environ\n",
        "def f():\n return (1).__class__\n",
        'def f():\n return eval("1")\n',
        "from math import __dict__\ndef f():\n return 0\n",
    ],
)
def test_unsafe_source_rejected(source):
    with pytest.raises(ValueError):
        validate_source(source)


@pytest.mark.parametrize(
    "raw",
    [
        '{"action":"read_file","path":"../secret"}',
        '{"action":"read_file","path":"src//solution.py"}',
        '{"action":"finish","action":"run_tests"}',
        '{"action":"finish","x":0}',
        '[{"action":"finish"}]',
    ],
)
def test_invalid_protocol(raw):
    with pytest.raises(ValueError):
        parse_action(raw)


def test_nested_or_nonfinite_json_is_an_invalid_action():
    for raw in ('{"action":[]}', '{"action":{}}', '{"action":NaN}', "[" * 2000 + "]" * 2000):
        with pytest.raises(ValueError):
            parse_action(raw)


def test_codec_and_trusted_comparison():
    value = {"x": (1, complex(2, 3), {4, 5}, None)}
    assert decode(json.loads(json.dumps(encode(value)))) == value
    with pytest.raises(ValueError):
        decode({"@": "pickle", "v": "anything"})
    case = Case(args=encode([]), expected=encode([1, 2]), comparison="set")
    assert compare(case, encode([2, 1]))
    assert not compare(case, encode([3]))


@pytest.mark.parametrize(
    "source",
    [
        "from operator import attrgetter\ndef f():\n return attrgetter('x')(())\n",
        "def f():\n return (x for x in []).gi_frame.f_globals\n",
        "from string import Formatter\ndef f():\n return Formatter()\n",
    ],
)
def test_indirect_introspection_bypasses_rejected(source):
    with pytest.raises(ValueError):
        validate_source(source)


def test_process_output_and_time_limits():
    import sys

    code, out, err, reason = bounded_process(
        [sys.executable, "-c", 'print("x"*100000)'], b"", 5, 1024
    )
    assert reason == "output_limit" and len(out) <= 1024
    code, out, err, reason = bounded_process(
        [sys.executable, "-c", "import time; time.sleep(5)"], b"", 0.1, 1024
    )
    assert reason == "timeout"
