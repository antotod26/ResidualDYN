import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import AudioDataset
from src.model import ResidualDYN
from src.utils import mulaw_baseline
import os

# NOTA: Abbiamo rimosso PESQ e STOI perché su Windows richiedono compilatori C++ complessi.
# Ci basiamo sull'MSE (Mean Squared Error) e sull'analisi visiva.

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- VALUTAZIONE MODELLO (Solo MSE) ---")
    
    # Caricamento Modello
    model = ResidualDYN().to(device)
    ckpt_path = os.path.join('checkpoints', 'residual_dyn_final.pth')
    
    if os.path.exists(ckpt_path):
        # weights_only=True è per sicurezza nelle nuove versioni di Torch
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        except:
             model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Checkpoint caricato con successo.")
    else:
        print(f"ERRORE: File {ckpt_path} non trovato. Esegui prima train.py!")
        return

    model.eval()
    
    # Dataset
    ds = AudioDataset(root_dir='./data', subset='test-clean', download=False)
    
    # Prendi un sample specifico (es. indice 10) per visualizzazione
    idx = 10
    if len(ds) <= idx: idx = 0 # Safety check se il dataset è piccolo
    
    m, g, target = ds[idx]
    target_np = target.numpy()
    
    with torch.no_grad():
        m_in = m.unsqueeze(0).to(device)
        g_in = g.unsqueeze(0).to(device)
        
        # Predizione rete
        pred_tensor = model(m_in, g_in).squeeze()
        pred = pred_tensor.cpu().numpy()

    # Baseline Mu-Law per confronto
    bl = mulaw_baseline(target_np)
    
    # Calcolo MSE
    mse_model = np.mean((target_np - pred)**2)
    mse_bl = np.mean((target_np - bl)**2)
    
    print("-" * 30)
    print(f"MSE Baseline (Mu-Law): {mse_bl:.2e}")
    print(f"MSE ResidualDYN:       {mse_model:.2e}")
    print("-" * 30)
    
    # --- PLOTTING ---
    print("Generazione grafico...")
    plt.figure(figsize=(10, 8))
    
    # 1. Waveform Zoom
    plt.subplot(2, 1, 1)
    # Cerchiamo di centrare lo zoom su una parte interessante
    mid = len(target_np) // 2
    zoom_slice = slice(mid, mid+200) 
    
    plt.plot(target_np[zoom_slice], label='Original', color='black', linewidth=1.5)
    plt.plot(pred[zoom_slice], label='ResidualDYN', linestyle='--', color='orange', linewidth=1.5)
    plt.title("Waveform Zoom (Dettaglio Ricostruzione)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Errore Residuo
    plt.subplot(2, 1, 2)
    plt.plot(target_np - bl, label='Err Baseline (Mu-Law)', alpha=0.5, color='blue')
    plt.plot(target_np - pred, label='Err ResidualDYN', alpha=0.8, color='orange')
    plt.title("Confronto Errore Residuo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print("Grafico salvato come 'evaluation_results.png'")

if __name__ == "__main__":
    run_evaluation()