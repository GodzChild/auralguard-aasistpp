
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

def load_fixed(path, sr=16000, seconds=8.0):
    wav, old_sr = torchaudio.load(path); wav = wav.mean(dim=0)
    if old_sr != sr: wav = torchaudio.functional.resample(wav, old_sr, sr)
    n=int(sr*seconds)
    if wav.numel()<n: wav=torch.nn.functional.pad(wav,(0,n-wav.numel()))
    else: wav=wav[:n]
    return wav.numpy().astype(np.float32), sr

def main():
    p=argparse.ArgumentParser(description='Create partial fake test audio by inserting fake speech into real speech.')
    p.add_argument('--real-csv', required=True); p.add_argument('--fake-csv', required=True); p.add_argument('--out-audio-dir', required=True); p.add_argument('--out-csv', required=True)
    p.add_argument('--num-samples', type=int, default=300); p.add_argument('--sample-rate', type=int, default=16000); p.add_argument('--duration-sec', type=float, default=8.0); p.add_argument('--insert-sec', type=float, default=2.0); p.add_argument('--seed', type=int, default=42)
    args=p.parse_args(); rng=np.random.default_rng(args.seed)
    real_df=pd.read_csv(args.real_csv, low_memory=False); fake_df=pd.read_csv(args.fake_csv, low_memory=False)
    real_df=real_df[real_df['binary_label'].astype(int)==0].reset_index(drop=True); fake_df=fake_df[fake_df['binary_label'].astype(int)==1].reset_index(drop=True)
    out_dir=Path(args.out_audio_dir); out_dir.mkdir(parents=True, exist_ok=True); rows=[]
    for i in tqdm(range(args.num_samples), desc='Creating partial fakes'):
        r=real_df.sample(n=1, random_state=int(rng.integers(0,1000000))).iloc[0]; f=fake_df.sample(n=1, random_state=int(rng.integers(0,1000000))).iloc[0]
        real,sr=load_fixed(r['file_path'], args.sample_rate, args.duration_sec); fake,_=load_fixed(f['file_path'], args.sample_rate, args.duration_sec)
        insert_len=int(args.insert_sec*sr); max_start=max(1,len(real)-insert_len); start=int(rng.integers(0,max_start)); end=start+insert_len
        mixed=real.copy(); mixed[start:end]=fake[start:end]
        out_path=out_dir/f'partial_fake_{i:06d}.wav'; sf.write(out_path,mixed,sr)
        rows.append({'file_path':out_path.resolve().as_posix(),'binary_label':1,'attack_type':'partial','start_fake':start/sr,'end_fake':end/sr,'dataset':'PartialSim','split':'test'})
    out_df=pd.DataFrame(rows); Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True); out_df.to_csv(args.out_csv,index=False); print('Saved:', args.out_csv, 'Rows:', len(out_df))
if __name__=='__main__': main()
