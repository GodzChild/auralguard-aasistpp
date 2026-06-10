
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
from tqdm import tqdm
from src.infer import load_model, run_auralguard

def iou(a_start,a_end,b_start,b_end):
    inter=max(0.0, min(a_end,b_end)-max(a_start,b_start)); union=max(a_end,b_end)-min(a_start,b_start)
    return inter/union if union>0 else 0.0

def parse_args():
    p=argparse.ArgumentParser(description='Evaluate sliding-window localization on partial fake metadata.')
    p.add_argument('--csv', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--aasist-root', default='external/aasist'); p.add_argument('--aasist-config', default='external/aasist/config/AASIST.conf')
    p.add_argument('--sample-rate', type=int, default=16000); p.add_argument('--duration-sec', type=float, default=8.0); p.add_argument('--feature-dim', type=int, default=160); p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int, default=0); p.add_argument('--out-csv', required=True); p.add_argument('--summary-csv', required=True)
    return p.parse_args()

def main():
    args=parse_args(); df=pd.read_csv(args.csv, low_memory=False)
    if args.limit>0: df=df.head(args.limit)
    device=torch.device(args.device); model=load_model(SimpleNamespace(**vars(args)), device); rows=[]
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Localization eval'):
        gt_start=float(row['start_fake']); gt_end=float(row['end_fake'])
        report=run_auralguard(row['file_path'], model, sample_rate=args.sample_rate, duration_sec=args.duration_sec, device=device)
        segs=report.get('suspicious_segments', []); best_iou=0.0; best_seg=None
        for s in segs:
            score=iou(gt_start, gt_end, float(s.get('start',0.0)), float(s.get('end',0.0)))
            if score>best_iou: best_iou=score; best_seg=s
        center_error=None
        if best_seg:
            gt_c=(gt_start+gt_end)/2; pr_c=(float(best_seg.get('start',0.0))+float(best_seg.get('end',0.0)))/2; center_error=abs(gt_c-pr_c)
        rows.append({'file_path':row['file_path'],'gt_start':gt_start,'gt_end':gt_end,'num_predicted_segments':len(segs),'best_iou':best_iou,'hit_iou_0_1':int(best_iou>=0.1),'hit_iou_0_3':int(best_iou>=0.3),'center_error_sec':center_error,'fake_probability':report.get('fake_probability',None),'attack_type':report.get('attack_type',None)})
    out=pd.DataFrame(rows); Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True); out.to_csv(args.out_csv,index=False)
    summary=pd.DataFrame([{'rows':len(out),'mean_best_iou':out['best_iou'].mean(),'hit_rate_iou_0_1':out['hit_iou_0_1'].mean(),'hit_rate_iou_0_3':out['hit_iou_0_3'].mean(),'mean_center_error_sec':out['center_error_sec'].dropna().mean()}])
    summary.to_csv(args.summary_csv,index=False); print(summary); print('Saved:', args.out_csv, args.summary_csv)
if __name__=='__main__': main()
