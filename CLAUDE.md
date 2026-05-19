# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nostril (Nonsense String Evaluator) is a Python 3 library that uses n-gram TF-IDF scoring to determine whether a text string is meaningful or random gibberish. Primary use case: classifying strings extracted from source code as real identifiers vs. nonsense.

## Build & Install

```bash
pip install .                # core library (zero runtime dependencies)
pip install ".[dev]"         # includes pytest, tabulate, humanize for development
pip install ".[extras]"      # includes tabulate, humanize for test utility functions
```

## Running Tests

```bash
pip install ".[dev]"
pytest -m "not slow"         # run all tests except slow ones
pytest -m slow               # run only the slow 1M-string test
pytest                       # run everything including slow tests
```

The test suite uses pytest and runs against multiple datasets (labeled cases, Ludiso oracle, OSX identifiers, dictionary words, GitHub identifiers, machine-generated random strings). Tests assert on false-positive/negative counts with tolerance margins. The `/usr/share/dict/web2` test is skipped on systems where that file doesn't exist.

## Architecture

**Core pipeline** (`nostril/nonsense_detector.py`):
1. `sanitize_string()` — strips numbers, spaces, punctuation; lowercases
2. Prefilter — fast heuristic rules catch obvious cases (length < 6 raises exception)
3. TF-IDF scoring — 4-gram analysis using precomputed weights from `ngram_data.pklz`
4. Custom scoring formula combining IDF values, repetition penalty, and length penalty

**Key exports** (`nostril/__init__.py`):
- `nonsense(s)` — returns `True` if string is likely nonsense
- `generate_nonsense_detector()` — creates a detector closure with custom tunable parameters (threshold, penalties, etc.)
- `sanitize_string(s)` — preprocessing step, also exported for direct use

**Data files**:
- `nostril/ngram_data.pklz` — pre-trained 4-gram weights (~5.9MB compressed pickle)
- `nostril/training/training_set.pklz` — training set (~33MB compressed pickle)

**Training** (`nostril/training/`):
- `training.py` generates the training set from identifier corpora + English word lists
- `optimize.py` tunes parameters using NSGA-II evolutionary algorithm (requires `platypus` library)
- Retraining is expensive (~7 hours on 4 cores)

**N-gram data structure** (`nostril/ng.py`): `NGramData` named tuple with fields: `string_frequency`, `total_frequency`, `idf`.

## CLI Usage

After `pip install .`, the `nostril` command is available directly:

```bash
nostril string1 string2 ...
nostril -f input_file.txt
nostril -t string1              # trace scoring details
```

Alternatively, run as a module without installing:

```bash
python3 -m nostril string1 string2 ...
```

## Key Constraints

- Minimum input length: 6 characters (raises exception for shorter strings)
- Trained on American English only
- All input is lowercased and stripped of non-alpha characters before evaluation
- Default parameters favor reducing false positives (real strings misclassified as nonsense) over false negatives
