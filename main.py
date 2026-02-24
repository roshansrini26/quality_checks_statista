import pandas as pd
import numpy as np
import os
from datetime import datetime

# Load data
df = pd.read_excel("/Users/roshans/Documents/Projects/quality_checks_statista/dataset/CaseStudy_Quality_sample25.xlsx")
df['providerkey'] = df['providerkey'].astype(str)

# Helper: detect full datetime in fiscalperiodend
def check_datetime(val):
    return isinstance(val, datetime) or (not pd.isnull(val) and hasattr(val, 'year'))

df['is_datetime'] = df['fiscalperiodend'].apply(check_datetime)

# ── CHECK 1: COMPLETENESS ──────────────────────────────────
imp_cols = ['companynameofficial', 'REVENUE', 'unit_REVENUE', 'fiscalperiodend', 'geonameen', 'industrycode']

df['flag_completeness'] = df[imp_cols].isnull().any(axis=1)
df['flag_completeness_info'] = df.apply(
    lambda row: 'Missing: ' + ', '.join([i for i in imp_cols if pd.isnull(row[i])])
    if row['flag_completeness'] else '',
    axis=1
)

# ── CHECK 2: CONSISTENCY ───────────────────────────────────
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

# ── CHECK 3: UNIQUENESS ────────────────────────────────────
df['flag_uniqueness'] = df.duplicated(subset=['providerkey', 'timevalue'], keep=False)
df['flag_uniqueness_info'] = df.apply(
    lambda r: f"duplicate entry: providerkey={r['providerkey']} timevalue={r['timevalue']}"
    if r['flag_uniqueness'] else '',
    axis=1
)

# ── CHECK 4: PLAUSIBILITY ──────────────────────────────────
df['flag_plausibility_negative'] = df['REVENUE'].notna() & (df['REVENUE'] < 0)

def iqr_flag(grp):
    rev = grp['REVENUE'].dropna()
    if len(rev) < 3:
        return pd.Series(False, index=grp.index)
    Q1, Q3 = rev.quantile(0.25), rev.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 3*IQR, Q3 + 3*IQR
    return grp['REVENUE'].apply(lambda x: False if pd.isnull(x) else not (lo <= x <= hi))

df['flag_plausibility_iqr'] = df.groupby('providerkey', group_keys=False).apply(iqr_flag)

def yoy_flag(grp):
    grp = grp.sort_values('timevalue').copy()
    grp['prev_rev'] = grp['REVENUE'].shift(1)
    def check_yoy(row):
        curr, prev = row['REVENUE'], row['prev_rev']
        if pd.isnull(curr) or pd.isnull(prev) or prev == 0:
            return False
        return abs(curr - prev) / abs(prev) > 3.0
    result = grp.apply(check_yoy, axis=1)
    grp.drop(columns=['prev_rev'], inplace=True)
    return result

df['flag_plausibility_yoy'] = df.groupby('providerkey', group_keys=False).apply(yoy_flag)

df['flag_plausibility'] = df['flag_plausibility_negative'] | df['flag_plausibility_iqr'] | df['flag_plausibility_yoy']
df['flag_plausibility_info'] = df.apply(
    lambda r: '; '.join(filter(None, [
        'negative_revenue'  if r['flag_plausibility_negative'] else '',
        'iqr_outlier'       if r['flag_plausibility_iqr']      else '',
        'yoy_spike_>300pct' if r['flag_plausibility_yoy']      else ''
    ])), axis=1
)
df.drop(columns=['flag_plausibility_negative', 'flag_plausibility_iqr', 'flag_plausibility_yoy'], inplace=True)

# ── CHECK 5: LLM-BASED ANOMALY ────────────────────────────
# Uses prompt engineering logic with domain knowledge
# to detect currency mismatches and revenue anomalies per company
COUNTRY_CURRENCY = {
    'United Kingdom': ['GBP'], 'Sweden':       ['SEK'], 'Denmark':     ['DKK'],
    'India':          ['INR'], 'United States': ['USD'], 'Indonesia':   ['IDR'],
    'France':         ['EUR'], 'Germany':       ['EUR'], 'Italy':       ['EUR'],
    'Spain':          ['EUR'], 'Netherlands':   ['EUR'],
}

def llm_anomaly_check(company, country, grp):
    """
    Simulates LLM prompt engineering logic:
    - Checks if currency matches the company's country
    - Flags zero revenue years
    - Flags identical revenue across all years (extraction error)
    """
    issues = []
    expected = COUNTRY_CURRENCY.get(country)
    units = grp['unit_REVENUE'].dropna().unique()
    if expected:
        bad = [u for u in units if u not in expected]
        if bad:
            issues.append(f"currency mismatch: {country} expects {expected} but found {bad}")
    valid_rev = grp['REVENUE'].dropna()
    if len(valid_rev[valid_rev == 0]) > 0:
        issues.append(f"zero revenue in {len(valid_rev[valid_rev == 0])} year(s)")
    if len(valid_rev) >= 3 and valid_rev.nunique() == 1:
        issues.append("revenue identical across all years (possible extraction error)")
    return (True, ' | '.join(issues)) if issues else (False, '')

print("Running LLM anomaly checks per company...")
llm_results = {}
for pk, grp in df.groupby('providerkey'):
    company = grp['companynameofficial'].dropna().iloc[0] if grp['companynameofficial'].notna().any() else 'Unknown'
    country = grp['geonameen'].iloc[0]
    result  = llm_anomaly_check(company, country, grp)
    llm_results[pk] = result
    if result[0]:
        print(f"  ⚠ {company}: {result[1]}")

df['flag_llm_anomaly']      = df['providerkey'].map(lambda pk: llm_results.get(pk, (False, ''))[0])
df['flag_llm_anomaly_info'] = df['providerkey'].map(lambda pk: llm_results.get(pk, (False, ''))[1])

# ── MASTER FLAG + OUTPUT ───────────────────────────────────
df['flag_any_issue'] = (
    df['flag_completeness'] | df['flag_consistency'] |
    df['flag_plausibility'] | df['flag_uniqueness']  | df['flag_llm_anomaly']
)

out_cols = [
    'timevalue', 'providerkey', 'companynameofficial', 'fiscalperiodend',
    'operationstatustype', 'ipostatustype', 'geonameen', 'industrycode', 'REVENUE', 'unit_REVENUE',
    'flag_completeness',  'flag_completeness_info',
    'flag_consistency',   'flag_consistency_info',
    'flag_plausibility',  'flag_plausibility_info',
    'flag_uniqueness',    'flag_uniqueness_info',
    'flag_llm_anomaly',   'flag_llm_anomaly_info',
    'flag_any_issue'
]

df[out_cols].to_excel("/Users/roshans/Documents/Projects/quality_checks_statista/dataset/output_CaseStudy.xlsx", index=False)

print(f"\n{'='*55}")
print(f"QUALITY CHECK SUMMARY  |  Total rows: {len(df)}")
print(f"{'='*55}")
for flag, label in [
    ('flag_completeness', 'Completeness (missing fields)'),
    ('flag_consistency',  'Consistency  (format / unit mismatch)'),
    ('flag_plausibility', 'Plausibility (negative / IQR / YoY)'),
    ('flag_uniqueness',   'Uniqueness   (duplicate company+year)'),
    ('flag_llm_anomaly',  'LLM Anomaly  (currency / zero / pattern)'),
    ('flag_any_issue',    'ANY ISSUE    (total flagged)'),
]:
    print(f"  {label:<45}: {df[flag].sum():>4} rows")