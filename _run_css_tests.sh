#!/bin/bash
cd "/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim"
export PYTHONPATH=src
export WISE_INPROCESS_LIMIT=999999999
my_venv/bin/python -m pytest tests/test_css_surgery_pipeline.py -x -q -k 'not TestDistanceSweep' 2>&1 | tee _css_test_output.txt
