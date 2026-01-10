import os
import re
import random
import pickle
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: Please install the 'datasets' library using 'pip install datasets'")
    load_dataset = None 


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = "".join([c for c in text if c.isprintable()])
    return text.strip()

def load_raw_wikipedia_dataset(
    language: str = "en", 
    date: str = "20220301",
    num_samples: int = None
) -> List[str]:
    if load_dataset is None:
        raise ImportError("The 'datasets' library is not installed.")
        
    print(f"Loading 'wikitext-103-raw-v1' from Hugging Face...")
    try:
        dataset_obj = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')
    except Exception as e:
        print(f"Error loading Wikitext dataset.")
        raise e

    dataset: List[str] = []
    print("Extracting text...")
    count = 0
    for item in dataset_obj:
        text = item['text']
        if len(text.strip()) > 50: 
            dataset.append(text)
            count += 1
            if num_samples is not None and count >= num_samples:
                break
        
    print(f"Loaded {len(dataset)} valid articles/paragraphs from Wikitext.")
    return dataset

def split_train_val_test_text_only(
    dataset: List[str],
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    if not 0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01:
         print("Warning: Sum of split ratios is not 1.0")
         
    random.seed(seed)
    random.shuffle(dataset)
    total = len(dataset)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = dataset[:train_end]
    val_set = dataset[train_end:val_end]
    test_set = dataset[val_end:]
    
    print(f"Split completed: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_set, val_set, test_set

def save_processed(data: Any, path: str):
    """Saves data to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved processed data to {path}")

def load_processed(path: str) -> Any:
    """Loads data from a pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found at {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded processed data from {path}")
    return data

def build_and_save_wikipedia_dataset(
    save_dir: str,
    wiki_lang: str = "en",
    wiki_date: str = "20231101",
    num_samples: int = None
):
    print("--- Starting Wikipedia Data Processing Pipeline ---")
    
    raw_data = load_raw_wikipedia_dataset(language=wiki_lang, date=wiki_date, num_samples=num_samples)
    
    print("Cleaning text data...")
    cleaned_data = [clean_text(text) for text in raw_data]
    cleaned_data = [text for text in cleaned_data if text]
    
    print(f"Total valid articles after cleaning: {len(cleaned_data)}")
    
    print("Splitting dataset...")
    train_set, val_set, test_set = split_train_val_test_text_only(cleaned_data)
    
    os.makedirs(save_dir, exist_ok=True)
    save_processed(train_set, os.path.join(save_dir, "train_wiki.pkl"))
    save_processed(val_set, os.path.join(save_dir, "val_wiki.pkl"))
    save_processed(test_set, os.path.join(save_dir, "test_wiki.pkl"))
    
    print("--- Done building and saving Wikipedia dataset. ---")
    return train_set, val_set, test_set


class BERTDataset(Dataset):
    def __init__(self, data: List[str], tokenizer, max_len: int = 512, mask_prob: float = 0.15):
        self.data = data
        self.tokenizer = tokenizer 
        self.max_len = max_len
        self.mask_prob = mask_prob
        
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        
        token_ids = self.tokenizer.encode(text) 
        
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids = token_ids + [self.pad_token_id] * (self.max_len - len(token_ids))
            
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        labels = input_ids.clone()
        
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        special_tokens_mask = (input_ids == self.pad_token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100 

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

if __name__ == "__main__":
    print("--- WARNING: Loading full English Wikipedia takes time ---")
    
    build_and_save_wikipedia_dataset(
        save_dir="data/processed", 
        num_samples=None 
    )