"""
Rename the template project to a new name.
Simple replacement of "myproject" to specified name in .py and .toml files
"""
import re
import sys
from pathlib import Path


OLD_NAME = "myproject"

def validate_project_name(name: str):
    pattern = r"^[a-z_][a-z0-9_]*$"
    if not bool(re.match(pattern, name)):
        raise ValueError("New name must contain only lowercase letters, numbers, underscores and not start with a number")
    

if __name__ == "__main__":
    new_name = sys.argv[1]
    root = Path.cwd()
    old_dir = root / OLD_NAME
    new_dir = root / new_name

    # Checks
    validate_project_name(new_name)
    if not old_dir.is_dir():
        raise ValueError(f"Error: Directory '{OLD_NAME}' not found.")
    if new_dir.exists():
        raise ValueError(f"Error: Directory '{new_name}' already exists.")


    # directory
    old_dir.rename(new_dir)

    # all occurrences in .py and .toml files
    for pattern in ("*.py", "*.toml"):
        for filepath in root.rglob(pattern):
            content = filepath.read_text()
            if OLD_NAME in content:
                filepath.write_text(content.replace(OLD_NAME, new_name))

    print(f"Renamed {OLD_NAME} to {new_name}")

