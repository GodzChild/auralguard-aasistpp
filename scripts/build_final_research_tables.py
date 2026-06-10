
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def read_optional(path):
    if not path: return None
    p=Path(path)
    if not p.exists():
        print(f'Warning: missing file {path}'); return None
    return pd.read_csv(p)

def main():
    p=argparse.ArgumentParser(description='Build final research summary tables.')
    p.add_argument('--gender-summary', default=''); p.add_argument('--length-summary', default=''); p.add_argument('--calibration-summary', default=''); p.add_argument('--partial-summary', default='')
    p.add_argument('--out-md', required=True); p.add_argument('--out-csv-prefix', required=True)
    args=p.parse_args(); sections=[]
    for title, path, suffix in [('Gender Bias Diagnosis',args.gender_summary,'gender'),('Audio Length Robustness',args.length_summary,'length'),('Calibration / Confidence Reliability',args.calibration_summary,'calibration'),('Partial Fake Localization',args.partial_summary,'partial')]:
        df=read_optional(path)
        if df is not None:
            df.to_csv(args.out_csv_prefix+'_'+suffix+'.csv', index=False)
            sections.append(f'## {title}\n\n'+df.to_markdown(index=False)+'\n')
    text='# AuralGuard-AASIST++ Robustness Evaluation Summary\n\n'+'\n'.join(sections)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True); Path(args.out_md).write_text(text, encoding='utf-8')
    print(text); print('Saved:', args.out_md)
if __name__=='__main__': main()
