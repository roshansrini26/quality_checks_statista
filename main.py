import pandas as pd
import numpy as np
import re
import os
import json
import requests
from datetime import datetime

#load data
df = pd.read_excel("/Users/roshans/Documents/Projects/quality_checks_statista/dataset/CaseStudy_Quality_sample25.xlsx")
df['providerkey'] = df['providerkey'].astype(str)

#to check correctness of fiscalperiodend 
def check_datetime(val):
    return isinstance(val, datetime) or (not pd.isnull(val) and hasattr(val, 'year'))

df['is_datetime'] = df['fiscalperiodend'].apply(check_datetime)

#Quality check 1: Completeness
imp_cols = ['companynameofficial', 'REVENUE', 'unit_REVENUE', 'fiscalperiodend', 'geonameen', 'industrycode']

df['flag_completeness'] = df[imp_cols].isnull().any(axis=1)
df['flag_completeness_info'] = df.apply(
    lambda row: 'Missing: ' + ', '.join([i for i in imp_cols if pd.isnull(row[i])])
    if row['flag_completeness'] else '',
    axis=1
)

#Quality check 2: Consistency
df['flag_consistency_fiscalperiodend'] = df['is_datetime']
df['flag_consistency_revenue_unit'] = (
    (df['REVENUE'].notna() & df['unit_REVENUE'].isnull()) |
    (df['REVENUE'].isnull() & df['unit_REVENUE'].notna())
)
df['flag_consistency'] = df['flag_consistency_fiscalperiodend'] | df['flag_consistency_revenue_unit']
df['flag_consistency_info'] = df.apply(
    lambda r: '; '.join(filter(None, [
        f"fiscal period format wrong: {r['fiscalperiodend']}" if r['flag_consistency_fiscalperiodend'] else '',
        'revenue unit mismatch'                               if r['flag_consistency_revenue_unit']   else ''
    ])), axis=1
)
df.drop(columns=['flag_consistency_fiscalperiodend', 'flag_consistency_revenue_unit'], inplace=True)

#Quality check 3: Uniqueness
df['flag_uniqueness'] = df.duplicated(subset=['providerkey', 'timevalue'], keep=False)
df['flag_uniqueness_info'] = df.apply(
    lambda r: f"duplicate entry: providerkey={r['providerkey']} timevalue={r['timevalue']}"
    if r['flag_uniqueness'] else '',
    axis=1
)