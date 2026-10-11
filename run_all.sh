#!/usr/bin/env bash
# Usage: bash run_all.sh path/to/export.csv
set -e
DATA="${1:-data/doctors_MPV_Jan26.csv}"
cd "$(dirname "$0")/src"
python 01_themes.py "../$DATA" ../results 10
python 02_keyword_coding.py "../$DATA" ../results
python 03_sources.py "../$DATA" ../results
