import zipfile
from pathlib import Path

# Replace with your registered team name
team_name = "submission_package"
zip_name = f"{team_name}.zip"

print(f"Creating final archive: {zip_name}...")

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    # 1. output/ folder
    zf.write("output/matching_results.tsv", "output/matching_results.tsv")
    zf.write("output/candidate_pairs.tsv", "output/candidate_pairs.tsv")
    
    # 2. Methodology document at root
    zf.write("Documentation_template.md", "Documentation_template.md")
    
    # 3. Source code tree
    code_root = Path("code")
    for file_path in code_root.rglob("*"):
        if file_path.is_file() and not any(skip in file_path.parts for skip in ["__pycache__", ".venv", ".git"]):
            zf.write(file_path, file_path.relative_to(Path(".")))

print(f"[SUCCESS] Packaged {zip_name} successfully.")
