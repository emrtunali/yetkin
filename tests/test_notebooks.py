"""Every code cell of every notebook must at least compile (shell lines starting with ! are skipped)."""

import json
from pathlib import Path

import pytest

NB = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_three_notebooks_exist():
    assert [p.stem for p in NB] == ["01_encoder", "02_qlora", "03_compare"]


@pytest.mark.parametrize("path", NB, ids=lambda p: p.stem)
def test_cells_compile(path):
    nb = json.loads(path.read_text())
    assert nb["metadata"]["accelerator"] == "GPU"
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(line for line in cell["source"] if not line.lstrip().startswith("!"))
        compile(src, f"{path.name}[{i}]", "exec")
