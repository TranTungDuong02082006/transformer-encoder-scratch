import tokenizer as tk_lib

class TokenizerWrapper:
    """
    Wrapper class to make functional tokenizer compatible with BERTDataset.
    """
    def __init__(self, vocab_path="data/processed/vocab_wiki.json"):
        self.token_to_id, self.id_to_token = tk_lib.load_vocab(vocab_path)
        self.vocab_size = len(self.token_to_id)

        self.pad_token_id = tk_lib.PAD_ID
        self.mask_token_id = tk_lib.MASK_ID
        self.cls_token_id = tk_lib.CLS_ID
        self.sep_token_id = tk_lib.SEP_ID
        self.unk_token_id = tk_lib.UNK_ID

    def encode(self, text):
        ids, _ = tk_lib.encode(
            text, 
            self.token_to_id, 
            max_len=None,
            add_cls=True, 
            add_sep=True, 
            pad_to_max_len=False
        )
        return ids
    
    def decode(self, ids):
        return tk_lib.decode(ids, self.id_to_token)

    def __len__(self):
        return self.vocab_size