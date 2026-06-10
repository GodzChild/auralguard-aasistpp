
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.infer import load_model, run_auralguard

def expected_calibration_error(y_true, probs, n_bins=10):
    y_true = np.asarray(y_true).astype(int); probs = np.asarray(probs).astype(float)
    bins = np.linspace(0,1,n_bins+1); ece=0.0; rows=[]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins-1 else probs < hi)
        if mask.sum()==0:
            rows.append({'bin':i,'low':lo,'high':hi,'n':0,'mean_confidence':None,'empirical_fake_rate':None,'abs_gap':None}); continue
        conf = probs[mask].mean(); rate = y_true[mask].mean(); gap=abs(conf-rate); ece += (mask.sum()/len(probs))*gap
        rows.append({'bin':i,'low':lo,'high':hi,'n':int(mask.sum()),'mean_confidence':conf,'empirical_fake_rate':rate,'abs_gap':gap})
    return ece, pd.DataFrame(rows)

def parse_args():
    p=argparse.ArgumentParser(description='Calibration / confidence reliability evaluation.')
    p.add_argument('--csv', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--aasist-root', default='external/aasist'); p.add_argument('--aasist-config', default='external/aasist/config/AASIST.conf')
    p.add_argument('--sample-rate', type=int, default=16000); p.add_argument('--duration-sec', type=float, default=4.0); p.add_argument('--feature-dim', type=int, default=160)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--limit', type=int, default=0); p.add_argument('--bins', type=int, default=10)
    p.add_argument('--out-pred-csv', required=True); p.add_argument('--out-calibration-csv', required=True); p.add_argument('--out-summary-csv', required=True)
    return p.parse_args()

def main():
    args=parse_args(); df=pd.read_csv(args.csv, low_memory=False)
    if args.limit and args.limit>0: df=df.head(args.limit)
    device=torch.device(args.device); model=load_model(SimpleNamespace(**vars(args)), device)
    rows=[]
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Calibration eval'):
        report=run_auralguard(row['file_path'], model, sample_rate=args.sample_rate, duration_sec=args.duration_sec, device=device)
        rows.append({'file_path':row['file_path'],'dataset':row.get('dataset',''),'binary_label':int(row['binary_label']),'attack_type':row.get('attack_type',''),'fake_probability':float(report.get('fake_probability',0.0)),'decision':report.get('decision',''),'pred_attack_type':report.get('attack_type','')})
    pred=pd.DataFrame(rows); Path(args.out_pred_csv).parent.mkdir(parents=True, exist_ok=True); pred.to_csv(args.out_pred_csv,index=False)
    ece, cal = expected_calibration_error(pred['binary_label'], pred['fake_probability'], args.bins); cal.to_csv(args.out_calibration_csv,index=False)
    brier=np.mean((pred['fake_probability'].values - pred['binary_label'].values)**2)
    summary=pd.DataFrame([{'rows':len(pred),'ece':ece,'brier_score':brier,'mean_fake_probability':pred['fake_probability'].mean(),'true_fake_rate':pred['binary_label'].mean()}])
    summary.to_csv(args.out_summary_csv,index=False); print(summary); print('Saved:', args.out_pred_csv, args.out_calibration_csv, args.out_summary_csv)
if __name__=='__main__': main()
