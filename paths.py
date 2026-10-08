"""paths.py — path configuration shared by every script.

The defaults are data/ and outputs/ inside this repository. Override them with
environment variables to use other locations.

    export ASD_DATA=/path/to/data
    export ASD_OUT=/path/to/output

ASD_XLSX names the team's internal summary workbook, which only excel_recompute.py
reads and which is optional. The original file was named in Korean; if yours still
carries that name, either rename it to the default below or point ASD_XLSX at it.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('ASD_DATA', ROOT / 'data'))
RAW  = Path(os.environ.get('ASD_RAW',  DATA / 'raw'))
OUT  = Path(os.environ.get('ASD_OUT',  ROOT / 'outputs'))
XLSX = Path(os.environ.get('ASD_XLSX', DATA / 'ASD_study_data.xlsx'))

OUT.mkdir(parents=True, exist_ok=True)
