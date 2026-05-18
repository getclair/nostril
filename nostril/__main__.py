#!/usr/bin/env python3
# =============================================================================
# @file    __main__.py
# @brief   Interface to run Nostril from the command line
# @author  Michael Hucka <mhucka@caltech.edu>
# @license Please see the file named LICENSE in the project directory
# @website https://github.com/casics/nostril
# =============================================================================

import argparse
import os
import sys

import nostril
from nostril import generate_nonsense_detector, sanitize_string


# Main program.
# .............................................................................

def main():
    parser = argparse.ArgumentParser(
        prog='nostril',
        description=(
            'Nostril is the Nonsense String Evaluator. It uses heuristics and '
            'statistical methods to infer whether a given text string is likely '
            'to be meaningful text or nonsense.'
        ),
        epilog=(
            'Note that Nostril has to load a large data file when it first '
            'starts up. In normal use, within an application program, this '
            'would only happen once. However, in this interactive program, '
            'every time you run this program, the data is (re)loaded, which '
            'means that startup is slow.'
        ),
    )
    parser.add_argument(
        '-f', '--file',
        help='read input from a file',
    )
    parser.add_argument(
        '-t', '--trace',
        action='store_true',
        help='trace scoring',
    )
    parser.add_argument(
        '-V', '--version',
        action='version',
        version='{} version {} — Author: {} — URL: {}'.format(
            nostril.__title__, nostril.__version__,
            nostril.__author__, nostril.__url__,
        ),
    )
    parser.add_argument(
        'strings',
        nargs='*',
        help='text strings to test',
    )

    args = parser.parse_args()

    if not args.file and not args.strings:
        parser.error('Need a file or list of strings as input argument')

    if args.file:
        file = args.file
        if os.path.exists(file):
            with open(file) as f:
                analyze(f.readlines(), args.trace)
        elif os.path.exists(os.path.join(os.getcwd(), file)):
            with open(os.path.join(os.getcwd(), file)) as f:
                analyze(f.readlines(), args.trace)
        else:
            raise ValueError('Cannot find file "{}"'.format(file))
    else:
        analyze(args.strings, args.trace)


def analyze(string_list, trace):
    if trace:
        nonsense = generate_nonsense_detector(trace=trace)
    else:
        nonsense = generate_nonsense_detector()
    padding = max(len(s) for s in string_list)
    for s in [string.rstrip() for string in string_list]:
        if len(sanitize_string(s)) >= 6:
            print('{}  [{}]'.format(s.ljust(padding),
                                      'nonsense' if nonsense(s) else 'real'))
        else:
            print('{}  [too short to test]'.format(s.ljust(padding)))


# Main entry point.
# .............................................................................

if __name__ == '__main__':
    main()
