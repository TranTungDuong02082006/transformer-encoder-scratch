import os
import csv
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import platform
import math
import time

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from data import load_processed, BERTDataset 
from tokenizer_wrapper import TokenizerWrapper 
from utils import set_seed, save_checkpoint 
from model.encoder import TransformerEncoder
from model.bert import BERTLanguageModel
from config import Config
    

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = "gloo" if platform.system() == 'Windows' else "nccl"
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(model, dataloader, optimizer, criterion, scaler, scheduler, device, epoch_idx, rank, accumulation_steps, log_path):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    num_batches = len(dataloader)
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=Config.use_amp):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * accumulation_steps
        total_loss += current_loss
        
        if rank == 0 and i % 50 == 0:
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            global_step = (epoch_idx - 1) * len(dataloader) + i
            print(f"[Epoch {epoch_idx}][Step {i}/{num_batches}] Loss: {current_loss:.4f} | LR: {lr:.2e} ...")
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_step, epoch_idx, current_loss, lr])

            start_time = time.time()
            
    return total_loss / num_batches


def validate_step(model, dataloader, criterion, device, rank):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=Config.use_amp):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            
    avg_loss = total_loss / num_batches
    if rank == 0:
        print(f">>> Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main_process(rank: int, world_size: int):
    use_ddp = world_size > 1
    if use_ddp:
        ddp_setup(rank, world_size)
        device = rank
    else:
        device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    set_seed(42 + rank)
    
    if rank == 0: print(f"Processing on rank {rank}...")
    tokenizer = TokenizerWrapper(Config.VOCAB_PATH)
    
    train_texts = load_processed(Config.TRAIN_PATH)
    val_texts = load_processed("data/processed/val_wiki.pkl")
    
    train_dataset = BERTDataset(train_texts, tokenizer, max_len=Config.max_len)
    val_dataset = BERTDataset(val_texts, tokenizer, max_len=Config.max_len)
    
    train_sampler = DistributedSampler(train_dataset) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=4 if platform.system() != 'Windows' else 0,
        pin_memory=True, persistent_workers=(True if platform.system() != 'Windows' else False)
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=2 if platform.system() != 'Windows' else 0,
        pin_memory=True
    )

    encoder = TransformerEncoder(
        vocab_size=len(tokenizer), d_model=Config.d_model, num_layers=Config.num_layers,
        num_heads=Config.num_heads, d_ff=Config.d_ff, max_len=Config.max_len, 
        dropout=Config.dropout, use_checkpointing=True 
    )
    model = BERTLanguageModel(encoder, d_model=Config.d_model, vocab_size=len(tokenizer)).to(device)
    model.tie_weights()

    if hasattr(torch, "compile"):
        print(f"[Rank {rank}] Compiling model...")
        model = torch.compile(model)

    if use_ddp:
        model = DDP(model, device_ids=[rank])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, betas=(0.9, 0.999), weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.use_amp)
    
    total_steps = (len(train_loader) // Config.accumulation_steps) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, Config.accumulation_steps, total_steps)

    train_log_path = os.path.join(Config.CHECKPOINT_DIR, "train_logs.csv")
    val_log_path = os.path.join(Config.CHECKPOINT_DIR, "val_logs.csv")

    if rank == 0:
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        print(f"--- Starting Training | Steps: {total_steps} ---")
        
        with open(train_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['global_step', 'epoch', 'train_loss', 'learning_rate'])
            
        with open(val_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'val_loss', 'perplexity'])
    

    best_val_loss = float('inf')

    global_step_counter = 0

    for epoch in range(1, Config.epochs + 1):
        if use_ddp: train_sampler.set_epoch(epoch)
        
        train_loss = train_step(
            model, train_loader, optimizer, criterion, scaler, scheduler, 
            device, epoch, rank, Config.accumulation_steps, train_log_path
        )
        
        val_loss = validate_step(model, val_loader, criterion, device, rank)
        
        if rank == 0:
            ppl = math.exp(val_loss) if val_loss < 100 else float('inf')
            print(f"Epoch {epoch} Done. Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            with open(val_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, val_loss, ppl])
            
            save_checkpoint(model_without_ddp, optimizer, epoch, train_loss, 
                          os.path.join(Config.CHECKPOINT_DIR, f"bert_last.pt"))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Found new best model! Saving...")
                save_checkpoint(model_without_ddp, optimizer, epoch, val_loss, 
                              os.path.join(Config.CHECKPOINT_DIR, f"bert_best.pt"))
    
    if use_ddp: destroy_process_group()

if __name__ == "__main__":
    try:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size > 1:
            mp.spawn(main_process, args=(world_size,), nprocs=world_size)
        else:
            main_process(0, 1)
    except Exception as e:
        import traceback
        traceback.print_exc()