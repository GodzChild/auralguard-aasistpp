
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
from tqdm import tqdm
from src.infer import load_model, run_auralguard

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate prediction stability across audio durations.')
    p.add_argument('--csv', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--aasist-root', default='external/aasist')
    p.add_argument('--aasist-config', default='external/aasist/config/AASIST.conf')
    p.add_argument('--sample-rate', type=int, default=16000)
    p.add_argument('--durations', nargs='+', type=float, default=[4,8,12,20])
    p.add_argument('--feature-dim', type=int, default=160)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--out-csv', required=True)
    p.add_argument('--summary-csv', required=True)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    if args.limit and args.limit > 0: df = df.head(args.limit)
    device = torch.device(args.device)
    model = load_model(SimpleNamespace(**vars(args)), device)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Length robustness'):
        for dur in args.durations:
            report = run_auralguard(row['file_path'], model, sample_rate=args.sample_rate, duration_sec=dur, device=device)
            rows.append({'file_path': row['file_path'], 'dataset': row.get('dataset',''), 'binary_label': int(row['binary_label']), 'attack_type': row.get('attack_type',''), 'duration_sec': dur, 'fake_probability': float(report.get('fake_probability',0.0)), 'decision': report.get('decision',''), 'pred_attack_type': report.get('attack_type','')})
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary = out.groupby(['dataset','binary_label','duration_sec']).agg(n=('fake_probability','size'), mean_fake_probability=('fake_probability','mean'), median_fake_probability=('fake_probability','median'), predicted_fake_rate_065=('fake_probability', lambda x: (x>=0.65).mean()*100), predicted_fake_rate_085=('fake_probability', lambda x: (x>=0.85).mean()*100)).reset_index()
    summary.to_csv(args.summary_csv, index=False)
    print(summary)
    print('Saved:', args.out_csv, args.summary_csv)
if __name__ == '__main__': main()
