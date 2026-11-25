#!/usr/bin/env python3
"""Build clean, professional submission ZIP files for Milestone 2 and Milestone 3.
Excludes all report documents (.pdf, .docx, .doc), pycache, git metadata, and OS artifacts.
"""

import os
from pathlib import Path
import shutil
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_SUBMISSION = PROJECT_ROOT / "final_submission"
FINAL_SUBMISSION.mkdir(parents=True, exist_ok=True)
CODE_SUBMISSION = FINAL_SUBMISSION / "code"
CODE_SUBMISSION.mkdir(parents=True, exist_ok=True)

def is_excluded(path: Path) -> bool:
    parts = path.parts
    # Exclude version control, caches, environments
    if any(p in [".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".DS_Store", "scratch"] for p in parts):
        return True
    if any(p.endswith(".pyc") for p in parts):
        return True
    # Explicitly exclude report documents per user request
    if path.suffix.lower() in [".pdf", ".docx", ".doc"]:
        return True
    if "reports" in parts and path.is_file():
        return True
    return False

def make_zip(source_dir: Path, zip_dest: Path, prefix: str = ""):
    print(f"Creating zip archive: {zip_dest}")
    if zip_dest.exists():
        zip_dest.unlink()
        
    file_count = 0
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = Path(root) / file
                if is_excluded(file_path):
                    continue
                rel_path = file_path.relative_to(source_dir)
                archive_name = Path(prefix) / rel_path if prefix else rel_path
                zf.write(file_path, str(archive_name))
                file_count += 1
    size_mb = zip_dest.stat().st_size / (1024 * 1024)
    print(f"  -> Packed {file_count} files into {zip_dest.name} ({size_mb:.2f} MB)")

def build_milestone_2():
    staging = PROJECT_ROOT / "build_staging" / "Milestone_2_Code"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # 1. Notebooks (Milestone 1 EDA + Milestone 2 Classical & Multivariate)
    nb_dir = staging / "notebooks"
    nb_dir.mkdir()
    shutil.copy2(PROJECT_ROOT / "notebooks" / "01_data_acquisition_and_eda.ipynb", nb_dir)
    shutil.copy2(PROJECT_ROOT / "notebooks" / "02_classical_econometrics_and_multivariate.ipynb", nb_dir)
    shutil.copy2(PROJECT_ROOT / "notebooks" / "Milestone_2_Classical_Econometrics.ipynb", nb_dir)

    # 2. Dataset
    data_dir = staging / "data"
    data_dir.mkdir()
    shutil.copy2(PROJECT_ROOT / "data" / "disease_timeseries_master.csv", data_dir)

    # 3. Source code (src)
    shutil.copytree(PROJECT_ROOT / "src", staging / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 4. Config & top-level docs
    shutil.copytree(PROJECT_ROOT / "config", staging / "config")
    for f in ["requirements.txt", "README.md", ".gitignore"]:
        src_f = PROJECT_ROOT / f
        if src_f.exists():
            shutil.copy2(src_f, staging)

    # Create zip in both locations for convenience
    make_zip(staging, FINAL_SUBMISSION / "Milestone_2_Code.zip", prefix="Milestone_2_Code")
    shutil.copy2(FINAL_SUBMISSION / "Milestone_2_Code.zip", CODE_SUBMISSION / "Milestone_2_Code.zip")

def build_milestone_3():
    staging = PROJECT_ROOT / "build_staging" / "Milestone_3_Code"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # 1. All Notebooks (EDA, Classical, ML/Deep Learning/AutoTS)
    nb_dir = staging / "notebooks"
    nb_dir.mkdir()
    for nb in PROJECT_ROOT.glob("notebooks/*.ipynb"):
        shutil.copy2(nb, nb_dir)

    # 2. Datasets & Processed Benchmarks
    data_dir = staging / "data"
    data_dir.mkdir()
    shutil.copy2(PROJECT_ROOT / "data" / "disease_timeseries_master.csv", data_dir)
    if (PROJECT_ROOT / "data" / "processed").exists():
        shutil.copytree(PROJECT_ROOT / "data" / "processed", data_dir / "processed")

    # 3. Source Code & App
    shutil.copytree(PROJECT_ROOT / "src", staging / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (PROJECT_ROOT / "pages").exists():
        shutil.copytree(PROJECT_ROOT / "pages", staging / "pages", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (PROJECT_ROOT / "app.py").exists():
        shutil.copy2(PROJECT_ROOT / "app.py", staging)

    # 4. Scripts & Audit Results
    scripts_dir = staging / "scripts"
    scripts_dir.mkdir()
    for sc in ["run_varima_autots_audit.py", "execute_notebooks.py", "build_submission_notebooks.py"]:
        p = PROJECT_ROOT / "scripts" / sc
        if p.exists():
            shutil.copy2(p, scripts_dir)

    for res in ["results_varima_autots_audit.csv", "results_varima_autots_audit.json"]:
        p = PROJECT_ROOT / res
        if p.exists():
            shutil.copy2(p, staging)

    # 5. Config & top-level docs
    shutil.copytree(PROJECT_ROOT / "config", staging / "config")
    for f in ["requirements.txt", "README.md", ".gitignore"]:
        src_f = PROJECT_ROOT / f
        if src_f.exists():
            shutil.copy2(src_f, staging)

    # Create zip in both locations for convenience
    make_zip(staging, FINAL_SUBMISSION / "Milestone_3_Code.zip", prefix="Milestone_3_Code")
    shutil.copy2(FINAL_SUBMISSION / "Milestone_3_Code.zip", CODE_SUBMISSION / "Milestone_3_Code.zip")

    # Also build Full_Project_Source_Code.zip
    make_zip(staging, FINAL_SUBMISSION / "Full_Project_Source_Code.zip", prefix="Full_Project_Source_Code")
    shutil.copy2(FINAL_SUBMISSION / "Full_Project_Source_Code.zip", CODE_SUBMISSION / "Full_Project_Source_Code.zip")

if __name__ == "__main__":
    print("Building clean submission ZIP files without reports...")
    build_milestone_2()
    build_milestone_3()
    # Clean up staging
    shutil.rmtree(PROJECT_ROOT / "build_staging", ignore_errors=True)
    print("All submission ZIP files created successfully.")
