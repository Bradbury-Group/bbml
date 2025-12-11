#!/bin/bash
# Rename the project from 'myproject' to your desired name
#
# Usage:
#   ./rename.sh your_project_name
#
# This will:
#   1. Rename the myproject/ directory
#   2. Update all imports in Python files
#   3. Update pyproject.toml
#   4. Update configs and README

set -e

if [ -z "$1" ]; then
    echo "Usage: ./rename.sh <new_project_name>"
    echo "Example: ./rename.sh my_vision_experiments"
    exit 1
fi

OLD_NAME="myproject"
NEW_NAME="$1"

# Validate new name (must be valid Python identifier)
if ! echo "$NEW_NAME" | grep -qE '^[a-z_][a-z0-9_]*$'; then
    echo "Error: Project name must be a valid Python identifier"
    echo "  - Start with lowercase letter or underscore"
    echo "  - Contain only lowercase letters, numbers, underscores"
    echo "  - Example: my_project, vision_exp_2024"
    exit 1
fi

echo "Renaming project: $OLD_NAME -> $NEW_NAME"

# Check if myproject directory exists
if [ ! -d "$OLD_NAME" ]; then
    echo "Error: Directory '$OLD_NAME' not found."
    echo "Are you running this from the project root?"
    exit 1
fi

# Check if new name already exists
if [ -d "$NEW_NAME" ]; then
    echo "Error: Directory '$NEW_NAME' already exists."
    exit 1
fi

# 1. Rename directory
echo "  Renaming directory..."
mv "$OLD_NAME" "$NEW_NAME"

# 2. Update Python imports in all .py files
echo "  Updating Python imports..."
find . -name "*.py" -type f -exec sed -i "s/from $OLD_NAME/from $NEW_NAME/g" {} \;
find . -name "*.py" -type f -exec sed -i "s/import $OLD_NAME/import $NEW_NAME/g" {} \;

# 3. Update pyproject.toml
echo "  Updating pyproject.toml..."
sed -i "s/name = \"$OLD_NAME\"/name = \"$NEW_NAME\"/g" pyproject.toml
sed -i "s/include = \[\"$OLD_NAME/include = \[\"$NEW_NAME/g" pyproject.toml
sed -i "s/--cov=$OLD_NAME/--cov=$NEW_NAME/g" pyproject.toml

# 4. Update configs
echo "  Updating configs..."
find configs -name "*.yaml" -type f -exec sed -i "s/project: $OLD_NAME/project: $NEW_NAME/g" {} \;

# 5. Update README
echo "  Updating README..."
sed -i "s/$OLD_NAME/$NEW_NAME/g" README.md

echo ""
echo "Done! Project renamed to '$NEW_NAME'"
echo ""
echo "Next steps:"
echo "  1. pip install -e '.[dev]'"
echo "  2. python scripts/list_experiments.py"
echo "  3. python scripts/run_experiment.py --name example_baseline -c configs/base.yaml"
