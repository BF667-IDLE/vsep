# Contributing to vsep

Thank you for your interest in contributing to vsep! This guide covers everything you need to know to get started, from setting up your development environment to submitting a Pull Request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Making Changes](#making-changes)
- [Testing Guidelines](#testing-guidelines)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Request Workflow](#pull-request-workflow)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Documentation](#documentation)

---

## Code of Conduct

This project follows a simple principle: **be respectful and constructive**. All interactions — in issues, pull requests, discussions, and direct communication — should be professional, inclusive, and focused on improving the project. Discriminatory, harassing, or otherwise disrespectful behavior will not be tolerated.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vsep.git
   cd vsep
   ```
3. **Add the upstream remote** to keep your fork in sync:
   ```bash
   git remote add upstream https://github.com/BF667-IDLE/vsep.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or newer
- FFmpeg installed and available on your `PATH`
- Git
- (Optional) A CUDA-capable NVIDIA GPU or Apple Silicon Mac for GPU testing

### Environment

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Verify everything works
python -c "from separator import Separator; print('OK')"
pytest tests/ -v --co  # List available tests without running them
```

### Recommended IDE Setup

- **VS Code**: Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) extensions. The project includes `.vscode`-compatible settings in `pyproject.toml`.
- **PyCharm**: Set the project interpreter to your `.venv` and configure Black as the external formatter.

## Project Architecture

Understanding the codebase structure will help you make effective contributions:

```
separator/
├── separator.py          ← Main entry point. Handles CLI args, model download, and delegates to architecture-specific classes.
├── common_separator.py   ← Shared base class with common separation logic used by all architectures.
├── ensembler.py          ← Implements all 11 ensemble algorithms (avg_wave, median_fft, etc.).
├── audio_chunking.py     ← Splits long audio into fixed-length chunks for memory-efficient processing.
└── architectures/
    ├── mdx_separator.py   ← MDX-Net ONNX model inference.
    ├── vr_separator.py    ← VR Band Split PyTorch model inference.
    ├── demucs_separator.py← Demucs v4 hybrid transformer inference.
    └── mdxc_separator.py  ← MDXC / Roformer checkpoint inference.
```

**Key flows:**
1. The `Separator` class in `separator.py` is the public API. It detects hardware, downloads models, and delegates to the correct architecture class.
2. Each architecture class in `architectures/` extends `CommonSeparator` and implements `_execute_separation()`.
3. The `Ensembler` in `ensembler.py` takes multiple separation results and combines them using the specified algorithm.

## Making Changes

### Branch Naming

Use descriptive branch names that follow these conventions:

| Type | Pattern | Example |
|:-----|:--------|:--------|
| Feature | `feature/description` | `feature/add-wav-ensemble` |
| Bug fix | `fix/description` | `fix/demucs-normalization` |
| Docs | `docs/description` | `docs/api-reference` |
| Refactor | `refactor/description` | `refactor/download-parallel` |
| Test | `test/description` | `test/roformer-validation` |

### Syncing with Upstream

Before starting new work, sync your fork:

```bash
git checkout main
git pull upstream main
git push origin main
```

### Making Your Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Make your changes with clear, focused commits.
3. Add or update tests for any behavioral changes.
4. Run the test suite and linter locally before pushing.

## Testing Guidelines

### Test Structure

Tests live in the `tests/` directory (not yet fully populated — contributions here are very welcome):

```
tests/
├── unit/              # Fast, isolated tests (no GPU or model downloads)
│   ├── test_parameter_validator.py
│   └── test_configuration_normalizer.py
└── integration/       # Tests that load models and separate audio (require GPU/Internet)
    └── test_roformer_*.py
```

### Running Tests

```bash
# Run all tests with verbose output and coverage
pytest tests/ -v --cov=separator --cov-report=term-missing

# Run only unit tests (fast, no network required)
pytest tests/unit/ -v

# Run a single test file
pytest tests/unit/test_parameter_validator.py -v

# Run a specific test by name
pytest tests/unit/test_parameter_validator.py::test_validate_bs_roformer -v
```

### Writing Tests

- **Unit tests** should be self-contained — mock external dependencies (file I/O, network, GPU).
- **Integration tests** may download models and run actual separation, but should be marked with appropriate markers.
- Use descriptive test names that explain the expected behavior: `test_mdx_separator_raises_error_for_invalid_segment_size`.
- Each test should test **one thing**. If you find a test doing multiple unrelated assertions, split it up.

## Code Style

vsep uses [Black](https://black.readthedocs.io/) with a line length of **140 characters**. This is configured in `pyproject.toml`.

```bash
# Format all files
black . --line-length 140

# Check without modifying (useful for CI)
black . --line-length 140 --check
```

Additional conventions:

- **Type hints**: Use them for function signatures and return types. They are not strictly enforced yet, but new code should include them.
- **Docstrings**: All public classes and methods should have Google-style docstrings explaining the purpose, parameters, and return value.
- **Imports**: Group imports in this order: standard library, third-party packages, local modules. Separate groups with blank lines.
- **Logging**: Use the `self.logger` instance on the `Separator` class. Do not use `print()` for output.
- **Error handling**: Raise specific exceptions (`ValueError`, `FileNotFoundError`, `RuntimeError`) with clear messages. Avoid bare `except:` clauses.

## Commit Messages

Write clear, concise commit messages that explain **what** changed and **why**:

```
# Good
fix(demucs): correct normalization threshold validation

The normalization_threshold parameter was accepting values <= 0, which
caused silent failures during audio output. Added a ValueError with a
descriptive message for values outside the (0, 1] range.

# Bad
fix stuff
```

Conventional commit prefixes we use:

| Prefix | Purpose |
|:-------|:--------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code restructuring (no behavior change) |
| `test:` | Adding or updating tests |
| `chore:` | Maintenance (dependencies, config, etc.) |

## Pull Request Workflow

1. **Push** your branch to your fork:
   ```bash
   git push origin feature/my-feature
   ```
2. **Open a Pull Request** against the `main` branch of the upstream repository.
3. **Fill in the PR template** (if available) with:
   - A clear description of what the PR does and why
   - Any relevant issue numbers (`Fixes #123`, `Closes #456`)
   - Steps to test the changes
   - Screenshots or audio samples if the change affects output quality
4. **Respond** to review feedback promptly and make requested changes.
5. **Keep the PR up to date** by rebasing on `main` if conflicts arise.

### PR Checklist

Before submitting, verify:

- [ ] Code passes `black --check --line-length 140`
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New functionality is covered by tests
- [ ] Documentation is updated (README, docstrings, config reference)
- [ ] Commit messages follow the conventions above
- [ ] No hardcoded credentials or API keys

## Reporting Bugs

When filing a bug report, please include:

1. **Python version**: `python --version`
2. **OS and hardware**: e.g., "Ubuntu 22.04, RTX 3060, CUDA 12.1"
3. **vsep version**: `python utils/cli.py -v`
4. **Model used**: e.g., `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
5. **Steps to reproduce**: The exact command or code that triggers the bug
6. **Expected vs. actual behavior**: What you expected to happen vs. what happened
7. **Log output**: Paste relevant log lines (use `-d` flag for debug logging)
8. **Input file**: If possible, share the audio file or a link to it (GitHub Issues supports attachments)

## Requesting Features

Feature requests are welcome! Please open a GitHub Issue with:

1. **Use case**: Describe the real-world scenario where this feature would help.
2. **Proposed behavior**: How you expect it to work.
3. **Alternatives considered**: Any workarounds you've tried.
4. **Scope**: Is this a small tweak or a major architectural change?

## Documentation

Documentation improvements are among the most valuable contributions. This project uses:

- **README.md** — Project overview, quick start, and feature showcase.
- **INSTALL.md** — Platform-specific installation instructions.
- **config/README.md** — Configuration variable reference.
- **remote/README.md** — Remote deployment guide.
- **separator/roformer/README.md** — Roformer architecture documentation.
- **CONTRIBUTING.md** — This file.

When updating documentation:

- Keep language consistent with the rest of the project (use "vsep", not "audio-separator").
- Update all relevant docs when adding features or changing behavior.
- Include code examples that can be copy-pasted and run.
- Add cross-references between related sections and files.

Thank you for contributing to vsep! Every contribution, no matter how small, makes the project better for everyone.
