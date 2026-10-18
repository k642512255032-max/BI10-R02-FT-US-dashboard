# -*- coding: utf-8 -*-
"""
make_sample_for_cloud.py – stratified 50,000-row sample of customer_scores_nba.csv for GitHub / Streamlit Community Cloud.

Takes up to 5,000 random customers from each of the 10 SUB_SEGMENTs (= 50,000 rows), reading the 1M-row CSV in chunks so it
runs on a laptop or in Colab, and writes customer_scores_sample.parquet (zstd, ~3 MB). The dashboard re-weights each
sub-segment back to its official full-base size, so portfolio estimates stay correct.

Usage:  python make_sample_for_cloud.py [path/to/customer_scores_nba.csv] [rows_per_sub_segment]
"""
import os
import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else 'customer_scores_nba.csv'
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000
OUT = 'customer_scores_sample.parquet'
rng = np.random.default_rng(2025)

kept = None
total = 0
for chunk in pd.read_csv(SRC, chunksize=200_000, low_memory=False, keep_default_na=False, na_values=['']):
    total += len(chunk)
    chunk['_R'] = rng.random(len(chunk))          # random key → keeping the PER smallest keys per stratum = a uniform random sample
    kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
    kept = kept.sort_values('_R').groupby('SUB_SEGMENT', group_keys=False, sort=False).head(PER)
    print(f'read {total:,} rows · kept {len(kept):,}', end='\r')

sample = kept.drop(columns='_R').sort_values(['SUB_SEGMENT', 'SKP_CLIENT']).reset_index(drop=True)
sample['NBA_TRIGGER'] = sample['NBA_TRIGGER'].replace({'': 'None'})
for c in sample.columns:                          # low-cardinality text → category: much smaller parquet
    if sample[c].dtype == object and sample[c].nunique() < 300:
        sample[c] = sample[c].astype('category')
sample.to_parquet(OUT, compression='zstd', index=False)
print(f'\nwrote {OUT}: {len(sample):,} rows from {total:,} · {os.path.getsize(OUT) / 1e6:.1f} MB')
print(sample.SUB_SEGMENT.value_counts().to_string())
