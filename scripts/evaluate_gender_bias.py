
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
from tqdm import tqdm
from src.infer import load_model, run_auralguard

def normalize_gender(x):
    x = str(x).strip().lower()
    if x in {'m', 'male', 'man'}: return 'male'
    if x in {'f', 'female', 'woman'}: return 'female'
    if x in {'nan', 'none', ''}: return 'unknown'
    return x

def parse_args():
    p = argparse.ArgumentParser(description='Gender bias diagnosis on real/bonafide speech.')
    p.add_argument('--csv', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--aasist-root', default='external/aasist')
    p.add_argument('--aasist-config', default='external/aasist/config/AASIST.conf')
    p.add_argument('--sample-rate', type=int, default=16000)
    p.add_argument('--duration-sec', type=float, default=4.0)
    p.add_argument('--feature-dim', type=int, default=160)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--thresholds', nargs='+', type=float, default=[0.65, 0.85])
    p.add_argument('--target-false-alarm', type=float, default=0.05)
    p.add_argument('--out-csv', required=True)
    p.add_argument('--summary-csv', required=True)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    if 'gender' not in df.columns:
        raise ValueError("CSV has no gender column. Run scripts/make_edacc_metadata_with_gender.py first.")
    df = df[df['binary_label'].astype(int) == 0].reset_index(drop=True)
    df['gender'] = df['gender'].map(normalize_gender)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    device = torch.device(args.device)
    model = load_model(SimpleNamespace(**vars(args)), device)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Gender bias eval'):
        report = run_auralguard(row['file_path'], model, sample_rate=args.sample_rate, duration_sec=args.duration_sec, device=device)
        fake_prob = float(report.get('fake_probability', 0.0))
        out = row.to_dict()
        out.update({'fake_probability': fake_prob, 'decision': report.get('decision',''), 'pred_attack_type': report.get('attack_type','')})
        for t in args.thresholds:
            out[f'false_alarm_at_{t}'] = int(fake_prob >= t)
        rows.append(out)
    pred = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_csv, index=False)
    summary_rows = []
    for gender, g in pred.groupby('gender'):
        item = {'gender': gender, 'n': len(g), 'mean_fake_probability': g['fake_probability'].mean(), 'median_fake_probability': g['fake_probability'].median(), 'p95_fake_probability': g['fake_probability'].quantile(0.95), 'calibrated_threshold_for_target_fa': g['fake_probability'].quantile(1 - args.target_false_alarm)}
        for t in args.thresholds:
            item[f'false_alarm_rate_at_{t}_percent'] = g[f'false_alarm_at_{t}'].mean() * 100
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    gap = {'gender': 'GAP_max_minus_min', 'n': ''}
    for t in args.thresholds:
        col = f'false_alarm_rate_at_{t}_percent'
        if col in summary and len(summary): gap[col] = summary[col].max() - summary[col].min()
    if len(summary): summary = pd.concat([summary, pd.DataFrame([gap])], ignore_index=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    print(summary)
    print('Saved:', args.out_csv, args.summary_csv)

if __name__ == '__main__':
    main()
