# Doc Consistency and Academic Compliance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the paper, README, and data manifest with the existing data and scripts, then rebuild the PDF without changing code or rerunning analyses.

**Architecture:** Documentation-only updates that correct method language, units, and metadata to match the current dataset and scripts. No pipeline reruns and no code changes.

**Tech Stack:** Markdown, LaTeX, Python (one-off stats checks), pdflatex.

---

### Task 1: Remove or correct clustered-inference claims

**Files:**
- Modify: `paper.md`
- Modify: `paper.tex`
- Modify: `README.md`

**Step 1: Write the failing test**

Run: `rg -n "cluster" paper.md paper.tex README.md`
Expected: Matches found that claim clustered SE or clustered interaction-DML.

**Step 2: Run test to verify it fails**

Run: `rg -n "cluster" paper.md paper.tex README.md`
Expected: Non-empty output (failing condition).

**Step 3: Write minimal implementation**

Update the text to reflect the actual scripts: GDP interaction uses unclustered OLS; institutional interaction uses DML with an unclustered OLS final stage; describe results as descriptive where needed.

**Step 4: Run test to verify it passes**

Run: `rg -n "cluster" paper.md paper.tex README.md`
Expected: No matches.

**Step 5: Commit**

Skip (no git repository).

---

### Task 2: Align CO2 units and descriptive stats with data

**Files:**
- Modify: `paper.md`
- Modify: `paper.tex`
- Modify: `README.md`

**Step 1: Write the failing test**

Run: `python3 - <<'PY'\nimport pandas as pd\nfrom pathlib import Path\nbase = Path('.')\ndf = pd.read_csv(base/'data/clean_data_v3_imputed.csv')\nco2 = df['CO2_per_capita'] / 100.0\nprint('co2 mean', co2.mean())\nprint('co2 std', co2.std())\nprint('co2 min', co2.min())\nprint('co2 max', co2.max())\nPY`
Expected: Values do not match the current paper table.

**Step 2: Run test to verify it fails**

Run the same command and confirm mismatch vs the table in `paper.md`/`paper.tex`.

**Step 3: Write minimal implementation**

Update the descriptive statistics table and add a short unit note that CO2 values are scaled by /100 in the analysis scripts.

**Step 4: Run test to verify it passes**

Re-run the command and confirm the table matches the printed values (rounded to two decimals).

**Step 5: Commit**

Skip (no git repository).

---

### Task 3: Fix variable counts and manifest coverage

**Files:**
- Modify: `DATA_MANIFEST.md`
- Modify: `README.md`

**Step 1: Write the failing test**

Run: `python3 - <<'PY'\nimport pandas as pd\nimport re\nfrom pathlib import Path\nbase = Path('.')\ndf = pd.read_csv(base/'data/clean_data_v3_imputed.csv')\ncols = set(df.columns)\ncols.discard('country')\ncols.discard('year')\nmanifest = (base/'DATA_MANIFEST.md').read_text()\nmanifest_vars = set(re.findall(r'`([^`]+)`', manifest))\nmissing = sorted(cols - manifest_vars)\nprint('missing', missing)\nPY`
Expected: Missing variables listed.

**Step 2: Run test to verify it fails**

Run the same command and confirm missing variables are listed.

**Step 3: Write minimal implementation**

Update `DATA_MANIFEST.md` to include the missing variables and add a short note clarifying the 60-variable count (excluding country/year). Update `README.md` to correct variable counts and file paths (CATE output in `results/`).

**Step 4: Run test to verify it passes**

Re-run the command and confirm the missing list is empty.

**Step 5: Commit**

Skip (no git repository).

---

### Task 4: Rebuild PDF

**Files:**
- Modify: `paper.pdf`

**Step 1: Write the failing test**

Run: `pdflatex -interaction=nonstopmode -halt-on-error paper.tex`
Expected: Successful build with updated PDF.

**Step 2: Run test to verify it fails**

If the build fails, capture errors and stop for clarification.

**Step 3: Write minimal implementation**

Fix LaTeX errors if any were introduced (should be none if text edits are safe).

**Step 4: Run test to verify it passes**

Re-run `pdflatex` until it succeeds.

**Step 5: Commit**

Skip (no git repository).
