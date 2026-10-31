# Shortcuts for the commands in docs/baseline.md. Optional: every recipe is plain
# commands you can also type yourself. Install just from https://just.systems
# Activate your virtual environment before you run a recipe.

import 'common.just'

mod index
mod app
mod eval

# List the recipes.
default:
    @just --list --list-submodules

# Run the tests.
test:
    {{python}} -m unittest discover -s tests

# Download the QASPER dev split into <dir> and rebuild the data files there.
data dir="data":
    {{python}} tools/download_data.py --data-dir {{dir}} --splits dev
    {{python}} tools/prepare_data.py --source {{dir}}/qasper-dev-v0.3.json --output-dir {{dir}}
