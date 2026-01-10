import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import json
import math
from datetime import datetime

from config import Config
from model.encoder import TransformerEncoder
from model.bert import BERTLanguageModel
from tokenizer_wrapper import TokenizerWrapper 
from data import load_processed, BERTDataset

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(checkpoint_path: str, device: torch.device):
    """
    Loads model architecture and weights using the global Config.
    """
    print(f"Loading tokenizer and model from {checkpoint_path}...")
    
    tokenizer = TokenizerWrapper(Config.VOCAB_PATH)
    vocab_size = len(tokenizer)
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        num_layers=Config.num_layers,
        num_heads=Config.num_heads,
        d_ff=Config.d_ff,
        max_len=Config.max_len,
        dropout=Config.dropout
    )
    
    model = BERTLanguageModel(encoder, d_model=Config.d_model, vocab_size=vocab_size)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

def save_eval_report(metrics: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": Config.to_dict(),
        "metrics": metrics
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\nReport saved to: {save_path}")

def evaluate_metrics(model, tokenizer, device, save_path: str = None):
    """
    Calculates Perplexity (PPL) and Masked Accuracy.
    """
    print("\n--- Starting Metric Evaluation ---")
    
    try:
        val_texts = load_processed(Config.VAL_PATH)
    except FileNotFoundError:
        print(f"Error: Validation data not found at {Config.VAL_PATH}")
        return

    val_dataset = BERTDataset(val_texts, tokenizer, max_len=Config.max_len)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    use_amp = Config.use_amp and torch.cuda.is_available()
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                
                flat_outputs = outputs.view(-1, outputs.size(-1))
                flat_labels = labels.view(-1)
                
                loss = criterion(flat_outputs, flat_labels)
            
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=-1)
            mask_positions = (labels != -100)
            correct_predictions = (predictions == labels) & mask_positions
            
            total_correct += correct_predictions.sum().item()
            total_masked += mask_positions.sum().item()

    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_masked if total_masked > 0 else 0.0
    
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity (PPL): {perplexity:.2f}")
    print(f"Masked Accuracy: {accuracy:.2%}")
    
    if save_path:
        metrics = {
            "validation_loss": avg_loss,
            "perplexity": perplexity,
            "masked_accuracy": accuracy
        }
        save_eval_report(metrics, save_path)

def predict_mask(model, tokenizer, text: str, device: torch.device, top_k: int = 5):
    """
    Demo function: Uses the actual tokenizer to process input.
    """
    model.eval()
    
    mask_token = "<mask >"
    clean_text = text.replace("<mask >", mask_token).replace("<mask>", mask_token).replace("<mask", mask_token)
    
    if mask_token not in clean_text:
        clean_text += f" {mask_token}"
        print(f"Warning: Appended mask token. Input: {clean_text}")

    input_ids_list = tokenizer.encode(clean_text)
    
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    try:
        mask_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    except:
        print(f"Error: Tokenizer did not produce MASK_ID ({tokenizer.mask_token_id}).")
        return

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        
    print(f"\nInput: {text}")
    
    for mask_pos in mask_indices:
        logits = outputs[0, mask_pos, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=top_k)
        
        print(f"--- Prediction at pos {mask_pos.item()} ---")
        for prob, idx in zip(top_probs, top_ids):
            token = tokenizer.decode([idx.item()])
            print(f"  {token:<15} ({prob.item():.2%})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate BERT Model")
    parser.add_argument("--mode", type=str, choices=["metrics", "demo"], default="demo", help="Mode: 'metrics' or 'demo'")
    parser.add_argument("--text", type=str, default="hanoi is the capital of <mask > .", help="Input text for demo")
    
    default_ckpt = os.path.join(Config.CHECKPOINT_DIR, "bert_best.pt")
    if not os.path.exists(default_ckpt):
        default_ckpt = Config.get_last_checkpoint_path()
        
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to .pt file")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    try:
        model, tokenizer = load_trained_model(args.checkpoint, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.mode == "metrics":
        evaluate_metrics(model, tokenizer, device, save_path=Config.REPORT_PATH)
    elif args.mode == "demo":
        predict_mask(model, tokenizer, args.text, device)

if __name__ == "__main__":
    main()