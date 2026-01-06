# Residual DYN
Official implementation of the project "Residual DYN: Block-Floating-Point Audio tramite Residual Learning Ibrido".
**Student ID:** 2151006

##Project Structure
```text
.
├── src/                # moduli codice sorgente
│   ├── model.py        # CNN (rete neurale)
│   ├── dsp.py          # DYN quantizzazioe (funzioni)
│   └── dataset.py      # LibriSpeech caricamento dati 
├── train.py            # addestramento
├── evaluate.py         # grafici e metriche
├── requirements.txt    
├── checkpoints/        
└── README.md           

pip install -r requirements.txt

python train.py --epochs 30 --lr 0.001 --batch_size 16

python evaluate.py