#!/bin/bash
export WISE_INPROCESS_LIMIT=999999999
cd "/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim"
exec my_venv/bin/python _quick_css_test.py > _quick_test_output.txt 2>&1
