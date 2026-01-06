
import torch
import numpy as np

def dyn_encoding_log(x, block_size=16, min_db=-60.0):
    """
    Applica encoding DYN: divide in blocchi, calcola gain, normalizza mantissa.
    """
    # Se arriva un numpy array, convertilo in tensore
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    
    # Pad se necessario (per avere multipli di 16)
    pad_len = (block_size - (x.shape[-1] % block_size)) % block_size
    if pad_len > 0:
        x = torch.nn.functional.pad(x, (0, pad_len))
    
    # Reshape in blocchi [N_blocchi, 16]
    x_blocked = x.view(-1, block_size)
    
    # 1. Calcolo del Picco per blocco
    peak = torch.max(torch.abs(x_blocked), dim=1)[0]
    
    # 2. Conversione in dB
    peak_db = 20 * torch.log10(peak + 1e-9)
    peak_db = torch.clamp(peak_db, min=min_db, max=0.0)
    
    # 3. Quantizzazione Gain (6 bit -> 64 livelli)
    # IMPORTANTE: Usa CEIL invece di FLOOR
    # Se usi floor, stimi un guadagno < picco reale -> clipping nella mantissa -> rumore
    # Usando ceil, guadagno >= picco reale -> mantissa sempre in [-1, 1] -> stabile
    step = -min_db / 63  # 64 livelli tra min_db e 0
    g_idx = torch.ceil((peak_db - min_db) / step)
    g_q_db = g_idx * step + min_db
    g_lin = 10 ** (g_q_db / 20)
    
    # Espandi gain per matchare dimensioni originali
    g_expanded = g_lin.unsqueeze(1).repeat(1, block_size).view(-1)
    
    # 4. Calcolo Mantissa
    m = x / (g_expanded + 1e-9)
    m = torch.clamp(m, -1.0, 1.0) # Safety check
    
    # 5. Quantizzazione Mantissa (8 bit)
    # Range [-1, 1] mappato su 256 livelli (int8 o uint8 simu)
    m_int = torch.round(m * 127)
    m_q = m_int / 127.0
    
    return m_q, g_expanded, g_lin

def dyn_deterministic_recon(x, block_size=16):
    """Funzione helper per ricostruzione rapida (solo DSP)"""
    m, g, _ = dyn_encoding_log(x, block_size)
    return (m * g).numpy() if isinstance(m, torch.Tensor) else (m * g)