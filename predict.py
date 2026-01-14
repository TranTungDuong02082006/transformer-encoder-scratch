import torch
import torch.nn.functional as F
import os
import sys

# Import modules
try:
    from config import Config
    from model.encoder import TransformerEncoder
    from model.bert import BERTLanguageModel
    from tokenizer_wrapper import TokenizerWrapper
except ImportError:
    print("Error: Please run this file from the project root directory.")
    sys.exit(1)

def load_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device: {device} ---")

    if not os.path.exists(Config.VOCAB_PATH):
        print(f"Error: Vocab file not found at {Config.VOCAB_PATH}")
        sys.exit(1)
    
    tokenizer = TokenizerWrapper(Config.VOCAB_PATH)
    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded! Vocab size: {vocab_size}")

    try:
        MASK_TOKEN = "<mask>"
        MASK_ID = tokenizer.token_to_id(MASK_TOKEN)
        
        if MASK_ID is None:
            MASK_ID = 4
            token_at_4 = tokenizer.id_to_token(MASK_ID)
            if token_at_4:
                MASK_TOKEN = token_at_4
                print(f"Warning: '<mask>' not found by name. Using token at ID 4: '{MASK_TOKEN}'")
            else:
                print(f"Error: Could not identify the mask token in the vocabulary.")
                sys.exit(1)
        else:
            print(f"Mask Token confirmed: '{MASK_TOKEN}' (ID: {MASK_ID})")
            
    except Exception as e:
        print(f"Error detecting Mask Token: {e}")
        sys.exit(1)
        
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

    # 3. Load Checkpoint
    checkpoint_path = "checkpoints/bert_best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/bert_last.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: Checkpoint file not found!")
        sys.exit(1)

    model.to(device)
    model.eval()
    
    return model, tokenizer, device, MASK_TOKEN, MASK_ID

def predict(text, model, tokenizer, device, MASK_TOKEN, MASK_ID):
    text = text.lower().strip()
    
    if MASK_TOKEN not in text:
        print(f"Input Error: Sentence must contain '{MASK_TOKEN}'")
        return

    # Encode
    encoding = tokenizer.encode(text)
    
    if hasattr(encoding, 'ids'):
        token_ids = encoding.ids
    else:
        token_ids = encoding

    input_tensor = torch.tensor([token_ids], device=device)
    mask_indices = (input_tensor == MASK_ID).nonzero(as_tuple=True)[1]
            
    if len(mask_indices) == 0:
        print(f"Tokenizer did not find ID {MASK_ID} in the sentence!")
        print(f" -> Actual IDs: {token_ids}")
        return
            
    mask_pos = mask_indices[0].item()

    with torch.no_grad():
        output = model(input_tensor, torch.ones_like(input_tensor))
    
    mask_logits = output[0, mask_pos, :]
    probs = F.softmax(mask_logits, dim=-1)
    top_k = torch.topk(probs, 5)
    
    print(f"\nInput: {text}")
    print(f"Model prediction (at '{MASK_TOKEN}'):")
    print("-" * 35)
    for value, index in zip(top_k.values, top_k.indices):
        word = tokenizer.id_to_token(index.item())
        if word in [MASK_TOKEN, "<pad>", "<unk>", "<cls>", "<sep>"]: continue
        print(f"{word:<20} : {value.item():.2%}")
    print("-" * 35)

if __name__ == "__main__":
    model, tokenizer, device, MASK_TOKEN, MASK_ID = load_system()
    
    while True:
        try:
            text = input("\n>> Enter sentence (q to quit): ")
            if text.lower() in ['q', 'exit']: break
            if not text.strip(): continue
            predict(text, model, tokenizer, device, MASK_TOKEN, MASK_ID)
        except Exception as e:
            print(f"Error: {e}")