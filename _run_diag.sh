#!/bin/bash
export WISE_INPROCESS_LIMIT=999999999
export WISE_SAT_WORKERS=1
export PYTHONUNBUFFERED=1
cd "/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim"
exec my_venv/bin/python -u _trace_phases.py > _diag_compile_output.txt 2>&1
