#!/usr/bin/env python3

import os
import pytest

from nostril import nonsense, dataset_from_pickle
from nostril import test_labeled as run_labeled_tests
from nostril import test_unlabeled as run_unlabeled_tests

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)


# --- Basic nonsense detection ---

@pytest.mark.parametrize("s", [
    'lakdfqtajaklj',
    'AaBbCcDdEeFGgHhIiJjKkLlMmNnOoPpQqRrSsTtU',
    'AcoGQMJyIapivScpnfuXUDMtgTtvAACYdAyABnSLpoABhzZWAAVvAYAAAnqUAFTPo',
    'BCDEFGHIJKLMNOPQRSTUVWXYZ',
    'CDjjiJJTbvFWaSdEtUygGMoGl',
    'CQgHAwIDFQIDAxYCAQIeAQIXgAAKCRC',
    'CgKDQpPUkdBTklaSUHIENPTUJVFRFRQKLStLStLStLStLStLStLStLStLStLSt',
    'aoaoesuouooeueooeuoaeuoeou',
    'iuewrofahgalkfgaufpiupqrjf',
    'ieeoienkjadfakj',
    'lalalaalkjuogaajfajlfal',
])
def test_known_nonsense(s):
    assert nonsense(s), f"Expected nonsense=True for '{s}'"


# --- Labeled test cases ---

def test_labeled_cases():
    csv_path = os.path.join(TESTS_DIR, 'labeled-cases', 'real-not-real.csv')
    # Returns (fp_list, fn_list, count, skipped, elapsed_time)
    fp_list, fn_list, count, skipped, elapsed = run_labeled_tests(csv_path, nonsense)
    assert len(fp_list) <= 8, f"Too many false positives: {len(fp_list)} (expected <= 8)"
    assert len(fn_list) <= 7, f"Too many false negatives: {len(fn_list)} (expected <= 7)"


# --- Unlabeled valid string tests ---

def test_ludiso_identifiers():
    path = os.path.join(TESTS_DIR, 'unlabeled-cases', 'ludiso.txt')
    tp, tn, fp, fn, skipped, elapsed = run_unlabeled_tests(path, nonsense)
    assert fp <= 8, f"Too many false positives on Ludiso: {fp} (expected <= 8)"


def test_osx_identifiers():
    path = os.path.join(TESTS_DIR, 'unlabeled-cases', 'select-identifiers-from-osx-frameworks.txt')
    tp, tn, fp, fn, skipped, elapsed = run_unlabeled_tests(path, nonsense)
    assert fp <= 7, f"Too many false positives on OSX identifiers: {fp} (expected <= 7)"


def test_hand_written_random_strings():
    path = os.path.join(TESTS_DIR, 'unlabeled-cases', 'random-by-hand.txt')
    tp, tn, fp, fn, skipped, elapsed = run_unlabeled_tests(path, nonsense, sense='invalid')
    total = tp + tn + fp + fn
    accuracy = 100 * (tp + tn) / total if total > 0 else 0
    assert accuracy >= 75.0, f"Accuracy on hand-written random too low: {accuracy:.2f}% (expected >= 75%)"


@pytest.mark.skipif(
    not os.path.exists('/usr/share/dict/web2'),
    reason='/usr/share/dict/web2 not available on this system',
)
def test_dictionary_words():
    tp, tn, fp, fn, skipped, elapsed = run_unlabeled_tests('/usr/share/dict/web2', nonsense)
    assert fp <= 100, f"Too many false positives on dictionary: {fp} (expected <= 100)"


def test_github_identifiers():
    path = os.path.join(
        PROJECT_DIR, 'nostril', 'training', 'identifier-corpora',
        'random-identifiers-from-github.txt',
    )
    if not os.path.exists(path):
        pytest.skip('GitHub identifiers corpus not found')
    tp, tn, fp, fn, skipped, elapsed = run_unlabeled_tests(path, nonsense)
    assert fp <= 10, f"Too many false positives on GitHub identifiers: {fp} (expected <= 10)"


@pytest.mark.slow
def test_machine_generated_random_strings():
    pickle_path = os.path.join(TESTS_DIR, 'unlabeled-cases', 'random_set.pklz')
    if not os.path.exists(pickle_path):
        pytest.skip('random_set.pklz not found')
    random_strings = dataset_from_pickle(pickle_path)
    tp, tn, fp, fn, skipped, elapsed = run_unlabeled_tests(
        random_strings[:1000000], nonsense, sense='nonsense',
    )
    total = tp + tn + fp + fn
    accuracy = 100 * (tp + tn) / total if total > 0 else 0
    assert accuracy >= 90.0, f"Accuracy on random strings too low: {accuracy:.2f}% (expected >= 90%)"
