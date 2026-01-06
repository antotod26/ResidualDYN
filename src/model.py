import torch
import torch.nn as nn

class ResidualDYN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Feature Encoder (K=7 per contesto locale) ---
        # Input: 2 canali (Mantissa, Gain espanso)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(64)
        )
        
        # --- Residual Predictor (Impara l'errore) ---
        self.body = nn.Sequential(
            # Depthwise separabile per efficienza
            nn.Conv1d(64, 64, kernel_size=7, padding=3, groups=64), 
            nn.Conv1d(64, 128, kernel_size=1), 
            nn.GELU(),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(64)
        )
        
        # Output: 1 canale (l'errore epsilon)
        self.head = nn.Conv1d(64, 1, kernel_size=3, padding=1)

    def forward(self, m, g):
        # Gestione dimensioni: m, g [Batch, Time] -> [Batch, 1, Time]
        if m.dim() == 2:
            m = m.unsqueeze(1)
            g = g.unsqueeze(1)
            
        x_in = torch.cat([m, g], dim=1) # [Batch, 2, Time]
        
        # Forward pass
        feat = self.encoder(x_in)
        feat = self.body(feat) + feat # Skip connection interna
        residual = self.head(feat)
        
        # Ricostruzione DYN standard (Baseline algebrica)
        x_base = m * g
        
        # Somma: Baseline + Residuo Predetto
        return x_base + residual