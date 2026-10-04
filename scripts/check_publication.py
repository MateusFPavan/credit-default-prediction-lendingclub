"""Publication check: the public tree must not carry working notes.

Fails (exit 1) when a tracked file:
  1. lives under _processo_interno/ or has a name starting with _interno (local-only notes,
     kept out by .gitignore; this catches a forced add);
  2. contains a machine-specific path (C:\\Users\\..., /Users/...);
  3. contains a work-tracker ID (P-123, L-123) or a reference to an _interno file;
  4. contains prose that reads as Portuguese. The repository is published in English; the
     one Portuguese file on purpose is references/one_pager.pt-br.md. Notebook code cells
     are not scanned for prose: a few plot labels there are drawn into stored images and
     can only change when those cells are re-run.

Scans `git ls-files`, so it checks exactly what is committed. Run from the repository root:
    python scripts/check_publication.py
"""
import json
import pathlib
import re
import subprocess
import sys

SELF = "scripts/check_publication.py"
SKIP_SUFFIX = (".png", ".jpg", ".jpeg", ".gif", ".pbix", ".joblib", ".parquet", ".gz", ".zip", ".ico")
PT_ON_PURPOSE = re.compile(r"\.pt-br\.")

LOCAL_PATH = re.compile(r"[A-Za-z]:(\\{1,2}|/)Users(\\{1,2}|/)|(?<![\w.])/Users/[A-Za-z]")
TRACKER_ID = re.compile(r"(?<![\w-])[PL]-\d{3}(?![\w-])")
INTERNAL_REF = re.compile(r"_interno")

# Portuguese function words with no common English or code homograph. A line is flagged
# with two distinct ones, or one plus a Portuguese-only letter (ã õ ç).
PT_WORDS = set("""
nao não tambem também entao então porque voce você sao são esta está estao estão isso isto
aqui ainda quando deve devem mesmo mesma pelo pela pelos pelas uma umas dos das apos após
até depois antes onde cada entre sobre muito muita foi sera será pode podem nosso nossa
seu sua esse essa este tudo nada sempre nunca porem porém quais linhas colunas arquivo
dados treino validacao validação lucro mediana resultado resultados etapa para que
""".split())
PT_LETTER = re.compile(r"[ãõç]")
WORD = re.compile(r"[a-zà-ÿ]+")


def looks_portuguese(line):
    low = line.lower()
    hits = {w for w in WORD.findall(low) if w in PT_WORDS}
    return len(hits) >= 2 or (len(hits) >= 1 and bool(PT_LETTER.search(low)))


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def notebook_prose(text):
    """Markdown cells and text outputs of a notebook, as (where, line) pairs."""
    nb = json.loads(text)
    for i, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        if cell.get("cell_type") == "markdown":
            for line in src.splitlines():
                yield f"cell {i} markdown", line
        for out in cell.get("outputs", []):
            t = out.get("text") or out.get("data", {}).get("text/plain", "")
            t = "".join(t) if isinstance(t, list) else t
            for line in t.splitlines():
                yield f"cell {i} output", line


def check(paths):
    problems = []
    for path in paths:
        name = pathlib.PurePosixPath(path)
        if path.startswith("_processo_interno/") or any(part.startswith("_interno") for part in name.parts):
            problems.append((path, "-", "internal file is tracked"))
            continue
        if path == SELF or path.lower().endswith(SKIP_SUFFIX):
            continue
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if LOCAL_PATH.search(line):
                problems.append((path, n, "machine-specific path"))
            if TRACKER_ID.search(line):
                problems.append((path, n, "work-tracker ID"))
            if path != ".gitignore" and INTERNAL_REF.search(line):
                problems.append((path, n, "reference to an _interno file"))
        if PT_ON_PURPOSE.search(path):
            continue
        if path.endswith(".ipynb"):
            lines = notebook_prose(text)
        else:
            lines = ((f"line {n}", line) for n, line in enumerate(text.splitlines(), 1))
        for where, line in lines:
            if looks_portuguese(line):
                problems.append((path, where, "Portuguese prose: " + line.strip()[:80]))
    return problems


if __name__ == "__main__":
    problems = check(tracked_files())
    for path, where, what in problems:
        print(f"{path}:{where}: {what}")
    print(f"publication check: {len(problems)} problem(s)" if problems else "publication check OK")
    sys.exit(1 if problems else 0)
