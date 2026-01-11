%%writefile tokenizer.py
import os
import re
import json
import pickle
from collections import Counter, defaultdict

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"

class WordPieceTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab else {}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.max_token_len = max(len(t) for t in self.vocab.keys()) if self.vocab else 0

    def load_vocab(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "token_to_id" in data:
                self.vocab = data["token_to_id"]
            else:
                self.vocab = data
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.max_token_len = max(len(t) for t in self.vocab.keys()) if self.vocab else 0

    def basic_tokenize(self, text):
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def encode_word(self, word):
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            if end - start > self.max_token_len:
                end = start + self.max_token_len

            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                return [UNK_TOKEN]
            
            tokens.append(cur_substr)
            start = end 
            
        return tokens

    def encode(self, text):
        basic_tokens = self.basic_tokenize(text)
        subword_tokens = []
        
        for token in basic_tokens:
            if token in self.vocab:
                subword_tokens.append(token)
            else:
                subword_tokens.extend(self.encode_word(token))
                
        ids = [self.vocab.get(t, self.vocab.get(UNK_TOKEN)) for t in subword_tokens]
        return subword_tokens, ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, UNK_TOKEN) for i in ids]
        text = ""
        for t in tokens:
            if t in [PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]:
                continue
            if t.startswith("##"):
                text += t[2:]
            else:
                text += " " + t
        return text.strip()


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    first, second = pair
    if second.startswith("##"):
        second_stripped = second[2:]
    else:
        second_stripped = second
        
    combined_token = first + second_stripped
    
    for word in v_in:
        w_out = p.sub(combined_token, word)
        v_out[w_out] = v_in[word]
        
    return v_out, combined_token

def train_tokenizer_bpe(train_pkl_path, vocab_size, save_path):
    print(f"--- Training Tokenizer (Pure Python) | Format: <unk>, ##subword ---")
    
    if not os.path.exists(train_pkl_path):
        print(f"Error: Not found {train_pkl_path}")
        return

    with open(train_pkl_path, "rb") as f:
        dataset = pickle.load(f)

    print("1. Counting word frequencies...")
    word_freqs = Counter()
    tokenizer_dummy = WordPieceTokenizer()
    
    LIMIT_SENTENCES = 50000 
    for i, text in enumerate(dataset):
        if i >= LIMIT_SENTENCES: break
        words = tokenizer_dummy.basic_tokenize(text)
        for w in words:
            word_freqs[w] += 1
            
    vocab_train = {}
    for word, freq in word_freqs.items():
        if len(word) == 1:
            vocab_train[word] = freq
        else:
            chars = [word[0]] + ["##" + c for c in word[1:]]
            vocab_train[" ".join(chars)] = freq

    base_vocab = {PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN}
    for word in vocab_train:
        for char in word.split():
            base_vocab.add(char)
            
    num_merges = vocab_size - len(base_vocab)
    print(f"Base vocab size: {len(base_vocab)}. Merging {num_merges} times (Running BPE)...")
    
    for i in range(num_merges):
        pairs = get_stats(vocab_train)
        if not pairs:
            print("No more pairs to merge.")
            break
            
        best = max(pairs, key=pairs.get)
        
        vocab_train, new_token = merge_vocab(best, vocab_train)
        
        base_vocab.add(new_token)
        
        if (i + 1) % 100 == 0:
            print(f"Merge {i+1}/{num_merges}: {best} -> {new_token}")

    print("Assigning IDs (Deterministic Sort)...")
    
    specials = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]
    final_vocab = {k: i for i, k in enumerate(specials)}
    idx = len(specials)
    
    sorted_tokens = sorted(list(base_vocab))
    
    for token in sorted_tokens:
        if token not in final_vocab:
            final_vocab[token] = idx
            idx += 1
            
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_vocab, f, ensure_ascii=False, indent=2)
        
    print(f"--- Done! Vocab saved to {save_path} (Size: {len(final_vocab)}) ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pkl", type=str, default="train_wiki.pkl")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--save_path", type=str, default="vocab_wiki.json")
    args = parser.parse_args()

    train_tokenizer_bpe(args.train_pkl, args.vocab_size, args.save_path)