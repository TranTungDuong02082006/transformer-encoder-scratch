import os
import json
import re
import pickle
from collections import Counter
from typing import List, Dict, Tuple, Optional

# --- 1. SPECIAL TOKENS AND DEFAULTS ---
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "<cls>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3
MASK_ID = 4

DEFAULT_VOCAB_SIZE = 30000
DEFAULT_SAVE_PATH = "data/processed/vocab_wiki.json"

# --- 2. NORMALIZE AND TOKENIZER ---
def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

# --- 3. VOCABULARY UTILITIES ---

def build_vocab_from_train(
    train_pkl_path: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    save_path: str = DEFAULT_SAVE_PATH,
    min_freq: int = 1,
    verbose: bool = True,
) -> Dict[str, int]:
    if not os.path.exists(train_pkl_path):
        raise FileNotFoundError(f"Train pickle file not found: {train_pkl_path}")
    
    with open(train_pkl_path, "rb") as f:
        train_data = pickle.load(f)
        
    # count token frequencies
    counter = Counter()
    for text in train_data: 
        tokens = tokenize(text)
        counter.update(tokens)
        
    # Filter tokens below min_freq
    if min_freq > 1:
        counter = Counter({tok: count for tok, count in counter.items() if count >= min_freq})
                
    # Determine how many tokens can keep beyond specials
    n_specials = 5 # PAD, UNK, CLS, SEP, MASK
    n_keep = max(0, vocab_size - n_specials)
    
    # Select top-n_keep tokens by frequency
    most_common = counter.most_common(n_keep)
    kept_tokens = [tok for tok, _ in most_common]
    
    # Build token -> id mapping, reserving ids for special tokens first
    token_to_id: Dict[str, int] = {}
    token_to_id[PAD_TOKEN] = PAD_ID
    token_to_id[UNK_TOKEN] = UNK_ID
    token_to_id[CLS_TOKEN] = CLS_ID
    token_to_id[SEP_TOKEN] = SEP_ID
    token_to_id[MASK_TOKEN] = MASK_ID
    
    next_id = n_specials
    for tok in kept_tokens:
        token_to_id[tok] = next_id
        next_id += 1
        
    # Save vocab to JSON
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    save_vocab(token_to_id, save_path, verbose=verbose)
    
    if verbose:
        print(f"Built vocab size (including specials): {len(token_to_id)}")
        if len(token_to_id) < vocab_size:
            print(f"Warning: requested vocab_size={vocab_size} but only {len(token_to_id)} tokens available.")
    return token_to_id


def save_vocab(token_to_id: Dict[str, int], save_path: str = DEFAULT_SAVE_PATH, verbose: bool = True):
    id_to_token = {str(v): k for k, v in token_to_id.items()}
    payload = {"token_to_id": token_to_id, "id_to_token": id_to_token}
    with open(save_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
    if verbose:
        print(f"Saved vocab to: {save_path}")
        
        
def load_vocab(vocab_path: str = DEFAULT_SAVE_PATH) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    with open(vocab_path, "r", encoding='utf-8') as fin:
        payload = json.load(fin)
    token_to_id = payload.get("token_to_id", {})
    id_to_token_raw = payload.get("id_to_token", {})
    
    # Convert id keys back to int
    id_to_token: Dict[int, str] = {int(k): v for k, v in id_to_token_raw.items()}
    return token_to_id, id_to_token


def encode(
    text: str,
    token_to_id: Dict[str, int],
    max_len: Optional[int] = None,
    add_cls: bool = True,
    add_sep: bool = True,
    pad_to_max_len: bool = True,
) -> Tuple[List[int], List[int]]:
        
    tokens = tokenize(text)
    ids = [token_to_id.get(tok, UNK_ID) for tok in tokens]
        
    special_tokens_count = 0
    if add_cls: special_tokens_count += 1
    if add_sep: special_tokens_count += 1
    
    if max_len:
        allowed_text_len = max_len - special_tokens_count
        ids = ids[:allowed_text_len]
    
    input_ids = []
    if add_cls:
        input_ids.append(CLS_ID)
    
    input_ids.extend(ids)
    
    if add_sep:
        input_ids.append(SEP_ID)
        
    seq_len = len(input_ids)
    
    if pad_to_max_len and max_len is not None:
        if seq_len < max_len:
            pad_length = max_len - seq_len
            input_ids = input_ids + [PAD_ID] * pad_length
            attention_mask = [1] * seq_len + [0] * pad_length
        else:
            input_ids = input_ids[:max_len]
            attention_mask = [1] * max_len
    else:
        attention_mask = [1] * seq_len
    
    return input_ids, attention_mask


def decode(
    ids: List[int],
    id_to_token: Optional[Dict[int, str]] = None,
    skip_specials_tokens: bool =True,
    join_with: str = " ",
) -> str:
    if id_to_token is None:
        toks = [f"<{i}>" for i in ids]
        return join_with.join(toks)
    
    tokens = []
    for i in ids:
        tok = id_to_token.get(i, UNK_TOKEN)
        if skip_specials_tokens and tok in {PAD_TOKEN, CLS_TOKEN}: 
            continue
        if skip_specials_tokens and tok == UNK_TOKEN:
            continue
            
        tokens.append(tok)
    return join_with.join(tokens)


def build_vocab_entrypoint(
    train_pkl_path: str = "data/processed/train_wiki.pkl",
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    save_path: str = DEFAULT_SAVE_PATH
):
    token_to_id = build_vocab_from_train(
        train_pkl_path=train_pkl_path,
        vocab_size=vocab_size,
        save_path=save_path,
        min_freq=1,
        verbose=True,
    )
    return token_to_id
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build/load tokenizer vocab")
    parser.add_argument("--train_pkl", type=str, default="data/processed/train_wiki.pkl", help="Path to train_wiki.pkl")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Target vocab size (including specials)")
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save vocab_wiki.json")
    parser.add_argument("--test_encode", action="store_true", help="Run a sample encode/decode test after building vocab")
    args = parser.parse_args()

    # 1. Build Vocab
    print("--- Building Vocabulary ---")
    tok2id = build_vocab_entrypoint(train_pkl_path=args.train_pkl, vocab_size=args.vocab_size, save_path=args.save_path)

    # 2. Load and Test
    print("\n--- Testing Encode/Decode ---")
    try:
        token_to_id, id_to_token = load_vocab(args.save_path)
    except FileNotFoundError:
        print(f"Error: Vocab file not found at {args.save_path}. Cannot test encode/decode.")
        exit()

    if args.test_encode:
        max_len_test = 32
        sample = "The new Apple iPhone 16 Pro Max has 8GB RAM, 1TB SSD, and 5G connectivity."
        
        ids, mask = encode(sample, token_to_id, max_len=max_len_test, add_cls=True, pad_to_max_len=True)
        print(f"Sample: \"{sample}\"")
        print(f"Max Len: {max_len_test}")
        print("Tokens (after tokenization):", tokenize(sample))
        print("Input IDs:", ids)
        print("Attention Mask:", mask)
        decoded = decode(ids, id_to_token, skip_specials_tokens=True)
        print("Decoded (skip specials):", decoded)