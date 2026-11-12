"""Check the actual report and executed notebook against submission requirements."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import nbformat
from pypdf import PdfReader

from project1_evidence import load_evidence

ROOT = Path(__file__).resolve().parents[1]
evidence = json.loads((ROOT / "results/summary.json").read_text())
checked = load_evidence(ROOT / "results")
assert checked.final.private_accuracy == 0.859
assert checked.final.public_accuracy == 0.884
assert checked.final.params == 4298970
assert checked.validation.correct == 4872 and checked.validation.rows == 5000
with tempfile.TemporaryDirectory() as temporary:
    scratch = Path(temporary) / "results"
    shutil.copytree(ROOT / "results", scratch)
    matrix_path = scratch / "validation-confusion.json"
    changed = json.loads(matrix_path.read_text())
    changed["counts"][0][0] += 1
    matrix_path.write_text(json.dumps(changed))
    try:
        load_evidence(scratch)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("Corrupt evidence checksum was accepted")
    sums = scratch / "SHA256SUMS"
    sums.write_text("".join(
        f"{hashlib.sha256(matrix_path.read_bytes()).hexdigest()}  validation-confusion.json\n"
        if line.endswith("  validation-confusion.json") else line + "\n"
        for line in sums.read_text().splitlines()
    ))
    try:
        load_evidence(scratch)
    except ValueError as error:
        assert "Confusion" in str(error)
    else:
        raise AssertionError("Inconsistent confusion counts were accepted")
pdf = PdfReader(ROOT / "report/project1.pdf")
assert 1 <= len(pdf.pages) <= 8, f"Report has {len(pdf.pages)} pages"
first_page = pdf.pages[0].extract_text()
assert evidence["repository"] in first_page, "Missing first-page repository URL"
report_text = "\n".join(page.extract_text() for page in pdf.pages)
for required in ("85.90", "88.40", "97.44", "4,298,970", "WRN-22-4"):
    assert required in report_text, f"Missing report result {required}"
notebook = nbformat.read(ROOT / "notebooks/project1_analysis.ipynb", as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
assert code_cells
assert all(cell.execution_count is not None for cell in code_cells), "Unexecuted code cell"
assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1)), "Nonsequential execution"
outputs = [output for cell in code_cells for output in cell.outputs]
assert not any(output.output_type == "error" for output in outputs), "Notebook contains an error"
text = "\n".join(output.get("text", "") for output in outputs)
for required in ("85.90", "88.40", "97.44", "4,298,970"):
    assert required in text, f"Missing printed notebook result {required}"
assert "Matched all 5,000 original validation predictions" in text, "Submitted notebook must retain the real checkpoint replay"
assert "Final full-data checkpoint loaded" in text, "Submitted notebook must retain the final checkpoint check"
assert sum("image/png" in output.get("data", {}) for output in outputs) >= 2, "Missing visible plots"
for artifact in (ROOT / "notebooks/project1_analysis.ipynb", ROOT / "report/project1.md"):
    content = artifact.read_text()
    assert not re.search(r"/Users/|/scratch/[A-Za-z0-9_]+/|/tmp/knicks", content), f"Private absolute path in {artifact.name}"
assert not re.search(r"/Users/|/tmp/knicks", report_text), "Private path in PDF"
print(f"PASS. PDF has {len(pdf.pages)} pages and the repository URL on page one.")
print(f"PASS. Notebook has {len(code_cells)} sequentially executed code cells, no errors, and visible plots.")
print("PASS. Artifacts distinguish 85.90% private test, 88.40% public test, and 97.44% validation accuracy.")
print("PASS. Confusion matrix independently totals 4,872 correct of 5,000 rows.")
print("NOT CHECKED. Repository visibility and Brightspace submission need their real services.")
