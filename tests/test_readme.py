"""The README quick start examples run as written and give feasible predictions.

The code blocks are read from README.md, so the test always checks what users see. Only the number of epochs is
lowered to keep CI fast; feasibility comes from the projection, not from training to convergence. Blocks that do
not start with an import are fragments of an example (e.g. the ``Constraint`` block snippet) and are skipped.
"""

import re
from pathlib import Path

import numpy as np
import pytest

README = Path(__file__).parents[1] / "README.md"
EPOCHS = 200
TOL = 1e-6  # inference_tolerance in the examples


def _quick_start_examples() -> list[str]:
    text = README.read_text(encoding="utf-8")
    quick_start = text.split("## Quick start", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", quick_start, re.S)
    return [block for block in blocks if block.startswith("import")]


EXAMPLES = _quick_start_examples()


def test_readme_has_quick_start_examples():
    assert len(EXAMPLES) == 2


@pytest.mark.parametrize("code", EXAMPLES, ids=["supervised", "self_supervised"])
def test_quick_start_example_runs_and_is_feasible(code):
    assert "epochs=2000" in code
    namespace = {}
    exec(compile(code.replace("epochs=2000", f"epochs={EPOCHS}"), str(README), "exec"), namespace)

    result = namespace["result"]
    preds = namespace["preds"]
    evaluated = namespace.get("x_te_t", namespace["x_tr_t"])  # the self-supervised example evaluates on x_tr_t
    assert preds.shape == (len(evaluated), 2)
    assert np.isfinite(preds).all()
    assert result.metrics["residual_max"] < TOL
    if "g=[" in code:
        assert "ineq_g1_max_violation" in result.metrics
    for name, value in result.metrics.items():
        if name.endswith("_max_violation"):
            assert value < TOL, name
