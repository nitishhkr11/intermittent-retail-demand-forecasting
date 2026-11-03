# Merge and run on Windows (PowerShell)

This explains how to add the completed Phase 3 work to your existing project folder without losing your raw data,
`.venv`, generated Phase 1 / Phase 2 artifacts or Git history.

## What you downloaded

| File | What it is |
|---|---|
| `riverdemand_completion_patch.zip` | **Reduced package.** Only the files that are new or changed since the Phase 3 Step 1 (XGBoost) snapshot, with paths relative to the project root (no extra wrapper folder). It is *not* a complete repository. |
| `riverdemand_full_source.zip` | **Complete source snapshot** of the updated project (code, notebooks, tests, docs, small derived data and artifacts). Without `.git`, `.venv`, raw M5 CSV files, caches and the git-ignored resume checkpoints. Use it for a fresh checkout, or to compare. You do **not** need it if you apply the patch. |
| `apply_riverdemand_patch.ps1` | Safe copy script (preview first, then copy). Not inside either ZIP. |
| `MERGE_AND_RUN_WINDOWS.md`, `IMPLEMENTATION_MANIFEST.md`, `PHASE3_COMPLETION_REPORT.md` | These documents (also inside the patch, at the project root). |

Nothing in the patch replaces a Phase 1 or Phase 2 file. The files it replaces are listed in
`IMPLEMENTATION_MANIFEST.md` ("Modified files"): they are Phase 3 placeholders, `config.yaml` (additions only, no line
removed), `README.md`, and a docstring in `src/models/__init__.py`.

## 1. Back up first

Everywhere below, `C:\path\to\intermittent-retail-demand-forecasting` stands for **your** project folder. Replace it with the real path
(yours contains spaces, so keep the quotes in every command).

```powershell
cd "C:\path\to\intermittent-retail-demand-forecasting"
git status                      # see what is uncommitted
git add -A
git commit -m "Checkpoint before adding Phase 3 completion patch"
```

If you prefer not to commit, copy the whole folder instead (excluding `.venv` is fine):
`Copy-Item -Recurse "C:\path\to\intermittent-retail-demand-forecasting" "C:\path\to\intermittent-retail-demand-forecasting_backup"`

## 2. Preview the merge (copies nothing)

Open `apply_riverdemand_patch.ps1` in VS Code, set the project folder and the ZIP location in the lines under `edit only these lines`, then run it:

```powershell
cd "$env:USERPROFILE\Downloads"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass     # only for this window
.\apply_riverdemand_patch.ps1
```

With `$DryRun = $true` (the default) it extracts the ZIP to a temporary folder and prints one line per file:

| Tag | Meaning | What the script does |
|---|---|---|
| `NEW` | The file does not exist in your project | copies it (when `$DryRun = $false`) |
| `SAME` | Identical to yours | skips it |
| `REPLACE` | Exists, differs, and is on the list of files this patch is meant to replace | copies it |
| `CONFLICT` | Exists, differs, and is **not** on that list (you changed it, or you generated it yourself) | **skips it** and tells you |

It never deletes anything.

### Inspecting a conflict before replacing it

A conflict means "your copy and the patch's copy differ and I did not expect that". Compare them before deciding:

```powershell
# the patch is extracted to $env:TEMP\riverdemand_patch_extract while the script runs; keep it by running with $KeepTemp = $true
git diff --no-index "C:\path\to\intermittent-retail-demand-forecasting\<file>" "$env:TEMP\riverdemand_patch_extract\<file>"
```

or in VS Code: right-click the first file, **Select for Compare**, then right-click the second file, **Compare with Selected**.
Only if you want the patch's version, set `$OverwriteConflicts = $true` and run again (or copy that one file by hand).
A conflict on a generated artifact (`artifacts\...`) usually means you already re-ran that notebook yourself; keep yours.

## 3. Copy

Set `$DryRun = $false` and run the script again. It copies the `NEW` and `REPLACE` files, then compares the SHA-256 of
every copied file with the ZIP and prints `DONE` or lists the files that differ.

Afterwards, Git shows exactly what the copy changed. `git status` lists the replaced and new files; `git diff config.yaml`
should show only added lines (the Phase 3 sections) and no removed line. If it shows a removed line, that line was a
change you had made to `config.yaml` after the Step 1 snapshot; copy it back by hand. The same check applies to `README.md`,
which the patch replaces completely.

Equivalent manual route (no script): extract the ZIP to a temporary folder and copy its contents **into** the project
root so that `app\`, `src\`, `notebooks\` land next to your existing folders (do not create
`intermittent-retail-demand-forecasting\intermittent-retail-demand-forecasting`). In Explorer, choose
*Copy and Replace* only for the files in the "Modified files" list of `IMPLEMENTATION_MANIFEST.md`.

## 4. Separate removal list (nothing is deleted automatically)

Four placeholder files from Phase 1 are superseded by the real dashboard pages in `app\views\`. If you leave them, they
do no harm to the tests, but Streamlit would show them as extra pages **only** if you launched the old layout; the new
entry point `app\app.py` ignores them. Move them out of the way (reversible) after the merge:

```powershell
cd "C:\path\to\intermittent-retail-demand-forecasting"
New-Item -ItemType Directory -Force "_removed_by_phase3_patch\app\pages" | Out-Null
Move-Item "app\pages\1_Overview.py","app\pages\2_Forecast.py","app\pages\3_Model_Comparison.py","app\pages\4_Inventory_Decision.py" "_removed_by_phase3_patch\app\pages\"
```

A leftover `app\pages\` folder matters: Streamlit treats any folder named `pages` next to the entry script as its own page
list, which would bypass the shared sidebar and theme. Remove the empty folder afterwards (`Remove-Item app\pages`).
Delete `_removed_by_phase3_patch` once you are happy.

## 5. Install the new dependencies into your existing `.venv`

```powershell
cd "C:\path\to\intermittent-retail-demand-forecasting"
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-phase3.txt
```

This adds `torch` (the default Windows wheel runs on the CPU; it is a large download, over 100 MB), `prophet`, `streamlit` and `plotly` on top of `requirements.txt`. You can
install only what you need: `python -m pip install torch`, `python -m pip install prophet`,
`python -m pip install streamlit plotly`.

If PyTorch reports a missing DLL, install the *Microsoft Visual C++ Redistributable (x64)*, restart PowerShell and retry.

## 6. Validate before you commit

```powershell
python -m pytest -m "not slow"
python -m src.evaluation.final_holdout status
streamlit run app/app.py
```

- The first command runs the whole fast suite (the reference run is recorded in `reports\phase3\PHASE3_TEST_RECORD.md`).
  Tests that need `torch`, `prophet` or the saved artifacts skip themselves, with the command that creates what is missing.
- `status` must print `candidates frozen : no` and `holdout evaluated : no`.
- The dashboard opens at <http://localhost:8501>. All six pages should load; the **Dark mode** switch in the sidebar
  should change the whole page.

Then commit when you are satisfied: `git add -A` and `git commit -m "Phase 3: LSTM/GRU, Prophet, comparison, inventory, dashboard, final-holdout guard"`.

## 7. The development artifacts are already in the patch

The patch carries the small result files the notebooks wrote (`artifacts\forecasts\`, `artifacts\metrics\`,
`artifacts\inventory\`, figures in `reports\phase3\`) so the dashboard works immediately. They are git-ignored, like your
Phase 2 artifacts. To regenerate them yourself, run in this order (from the project root, `.venv` active):

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_intermittent_models.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_lstm_gru.ipynb      # ~20 min, one CPU thread; resumable
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/08_prophet.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/09_model_comparison.ipynb   # needs 04-08 first
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/10_inventory_decision_support.ipynb
```

Notebook 04 and 06 already produced your Phase 2 and XGBoost files; do not re-run them unless you want to regenerate
those (their saved outputs are not changed by this patch).

Expect small last-digit differences in the neural and Prophet numbers on your machine: they depend on library versions
and the platform. The run metadata files record what produced the numbers saved here.

### Prophet and the spaces in your path

Prophet runs a Stan executable that ships inside the Python environment. From a path with spaces
(for example `D:\My Folder\project\.venv`) it can fail to start. If notebook 08 raises `ProphetBackendError`:

```powershell
python -m venv C:\venvs\riverdemand
C:\venvs\riverdemand\Scripts\Activate.ps1
python -m pip install -r "C:\path\to\intermittent-retail-demand-forecasting\requirements-phase3.txt"
cd "C:\path\to\intermittent-retail-demand-forecasting"
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/08_prophet.ipynb
```

Nothing else needs this; the saved Prophet results and the dashboard work from your normal `.venv`.

## 8. Run the dashboard

```powershell
cd "C:\path\to\intermittent-retail-demand-forecasting"
.\.venv\Scripts\Activate.ps1
streamlit run app/app.py
```

Stop it with Ctrl+C. If port 8501 is busy: `streamlit run app/app.py --server.port 8502`.
To open straight into dark mode: <http://localhost:8501/?theme=dark>.
After you change a Python file, stop and restart the dashboard (file watching is off in `.streamlit\config.toml`).

## 9. Final holdout evaluation (do this once, last)

The last 28 days (2016-04-25 to 2016-05-22) are reserved. Nothing in the repository has read them. The final-holdout
module is the only code that can, and it refuses to start unless every step below was done.

```powershell
# 1. Make sure every development artifact is present and notebook 09 has been reviewed.
python -m src.evaluation.final_holdout status

# 2. Freeze the candidates. This uses development folds only and writes artifacts\final\frozen_candidates.json.
python -m src.evaluation.final_holdout freeze

# 3. From here on do not change any model, feature or setting. Evaluate once:
python -m src.evaluation.final_holdout run --confirm EVALUATE-FINAL-HOLDOUT-ONCE
```

What `run` writes: `artifacts\final\final_holdout_forecasts.csv`, `final_holdout_results.csv` and
`FINAL_EVALUATION_RECORD.json`. Report aggregate and item-level numbers separately; never rank them together.
It refuses when: the phrase is missing or wrong, nothing is frozen, `config.yaml` changed since the freeze, the
holdout was already evaluated, or a candidate name no longer matches. `--overwrite` re-freezes; `--override-existing`
archives the earlier evaluation (as `*.superseded-N`) and records that this is a repeat. Changing a model after seeing
the holdout result turns the holdout into a validation set; report it as such.

What is still open in this snapshot: **nothing is frozen and nothing is evaluated.** Freezing is your decision: the
candidate list is already declared in `config.yaml` (`final_holdout`), the headline model per scope is chosen by lowest
mean development-fold MAE when you run `freeze`, and the frozen file records that choice and the settings hash.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window |
| Dashboard page says a file is missing | The page prints the exact command; run it from the project root |
| `ModuleNotFoundError: streamlit` / `torch` / `prophet` | `python -m pip install -r requirements-phase3.txt` with the `.venv` active |
| Notebook 07 interrupted | Run the same command again; finished (variant, fold) pairs are restored from `artifacts\models\lstm_gru_checkpoints\` |
| Test skipped with "not found. Create it with: ..." | Run that command, or ignore it if you do not need that artifact |
| Dashboard shows old CSS or text after an edit | Restart `streamlit run app/app.py` |
