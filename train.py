
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import AudioDataset
from src.model import ResidualDYN
import os

def train_model(args):
    print("INIZIO TRAINING RESIDUAL DYN")
    
    # Configurazione Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {device}")

    #dataset e dataLoader
    #test-clean per velocità (vedi notebooks)
    train_ds = AudioDataset(root_dir=args.data_dir, subset='test-clean', download=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    #modello
    model = ResidualDYN().to(device)
    print(f"Modello creato con {sum(p.numel() for p in model.parameters())} parametri.")
    
    #loss e ottimizzatore
    criterion = nn.MSELoss() # Loss standard per regressione
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    #Scheduler OneCycleLR (fondamentale per convergenza rapida)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                              steps_per_epoch=len(loader), epochs=args.epochs)
    
    #ciclo training
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for m, g, target in loader:
            m, g, target = m.to(device), g.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            #forward
            output = model(m, g)
            
            #loss calcolata sul target reale
            loss = criterion(output.squeeze(), target)
            
            #backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.2e}")
        
    #salvataggio
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'residual_dyn_final.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training ResidualDYN")
    parser.add_argument("--epochs", type=int, default=30, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=16, help="Dimensione batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--data_dir", type=str, default="./data", help="Cartella dati")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Cartella output")
    
    args = parser.parse_args()

    train_model(args)
