# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
"""Concatenate work/p_<class>.jsonl (written by run_pipeline.py) into results/features10.csv."""
import glob, os, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
order = ['B-1', 'B-52', 'C-130', 'C-135', 'B-2', 'C-17', 'C-5', 'P-3', 'KC-10', 'E-3']
fs = [os.path.join(ROOT, 'work', 'p_%s.jsonl' % c) for c in order]
d = pd.concat([pd.read_json(f, lines=True) for f in fs if os.path.exists(f)])
d.to_csv(os.path.join(ROOT, 'results', 'features10.csv'), index=False); print(d.shape)
