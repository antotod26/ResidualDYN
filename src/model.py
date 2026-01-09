import torch
import torch.nn as nn

class ResidualDYN(nn.Module):
    def __init__(self):
        super().__init__()
        
        #feature encoder (K=7 per contesto locale)
        #input: 2 canali (mantissa, gain espanso)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(64)
        )
        
        #residual predictor (impara errore)
        self.body = nn.Sequential(
            #depthwise separabile per efficienza
            nn.Conv1d(64, 64, kernel_size=7, padding=3, groups=64), 
            nn.Conv1d(64, 128, kernel_size=1), 
            nn.GELU(),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(64)
        )
        
        #output: 1 canale (errore epsilon)
        self.head = nn.Conv1d(64, 1, kernel_size=3, padding=1)

    def forward(self, m, g):
        #gestione dimensioni: m, g [Batch, Time] -> [Batch, 1, Time]
        if m.dim() == 2:
            m = m.unsqueeze(1)
            g = g.unsqueeze(1)
            
        x_in = torch.cat([m, g], dim=1) # [Batch, 2, Time]
        
        #forward pass
        feat = self.encoder(x_in)
        feat = self.body(feat) + feat # Skip connection interna
        residual = self.head(feat)
        
        #ricostruzione DYN standard (baseline algebrica)
        x_base = m * g
        
        #somma baseline + residuo predetto

        return x_base + residual
