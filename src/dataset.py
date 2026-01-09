
import torch
import torchaudio
import os
import numpy as np
from torch.utils.data import Dataset
from src.dsp import dyn_encoding_log

class AudioDataset(Dataset):
    def __init__(self, root_dir='./data', subset='test-clean', seg_len=16000, download=True):
        self.seg_len = seg_len
        self.data = []
        
        print(f"Caricamento {subset}...")
        # scaricare dataset se non c'è
        os.makedirs(root_dir, exist_ok=True)
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root_dir, url=subset, download=download)
        
        # processamento e cache in RAM
        print("Preprocessing segmenti...")
        for i in range(len(self.dataset)):
           # si prende solo il primo elemento (l'audio) e si ignora il resto
            waveform = self.dataset[i][0]
            # taglio a segmenti di 1 secondo (seg_len)
            num_segments = waveform.shape[1] // seg_len
            for j in range(num_segments):
                seg = waveform[0, j*seg_len : (j+1)*seg_len]
                
                # filtro silenzio (se max amplitude < 0.02 scarta)
                if torch.max(torch.abs(seg)) > 0.02:
                    self.data.append(seg)
                    
        print(f"Dataset Caricato: {len(self.data)} segmenti validi.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # pre-calcolo encoding 
        m, g_exp, _ = dyn_encoding_log(x)

        return m, g_exp, x # Input1, Input2, Target
