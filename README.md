# dwarffi

Basic Python project scaffold (src layout) with pytest + ruff.

## Quickstart

Create a virtualenv, then install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

Run the module:

```bash
python -m dwarffi --version
```

Or via the installed script:

```bash
dwarffi --help
```
