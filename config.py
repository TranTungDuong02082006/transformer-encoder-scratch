import os
import torch

class Config:
    TRAIN_PATH = "data/processed/train_wiki.pkl"
    VAL_PATH = "data/processed/val_wiki.pkl"
    TEST_PATH = "data/processed/test_wiki.pkl"
    VOCAB_PATH = "data/processed/vocab_wiki.json"
    CHECKPOINT_DIR = "checkpoints"
    REPORT_PATH = "reports/eval_metrics.json"

    load_model_path = None 

    d_model = 512
    num_layers = 8
    num_heads = 8
    d_ff = 2048
    max_len = 256
    dropout = 0.1
    vocab_size = 30000
    
    batch_size = 128      
    accumulation_steps = 2 
    
    epochs = 15
    
    lr = 2.5e-4
    weight_decay = 0.01
    
    warmup_steps = 2000   

    use_amp = True
    clip_grad = 1.0

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    num_workers = 8 if os.name != 'nt' else 0 
    
    @classmethod
    def get_last_checkpoint_path(cls):
        return os.path.join(cls.CHECKPOINT_DIR, "bert_last.pt")

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v)}