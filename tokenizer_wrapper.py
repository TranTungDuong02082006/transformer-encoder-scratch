import tokenizer as tk_lib

class TokenizerWrapper:
    def __init__(self, vocab_path="vocab_wiki.json"):
        self.processor = tk_lib.WordPieceTokenizer()
        self.processor.load_vocab(vocab_path)
        self.vocab = self.processor.vocab 
        self.vocab_size = len(self.processor.vocab)
        self.pad_token_id = self.vocab.get(tk_lib.PAD_TOKEN, 0)
        self.unk_token_id = self.vocab.get(tk_lib.UNK_TOKEN, 1)
        self.cls_token_id = self.vocab.get(tk_lib.CLS_TOKEN, 2)
        self.sep_token_id = self.vocab.get(tk_lib.SEP_TOKEN, 3)
        self.mask_token_id = self.vocab.get(tk_lib.MASK_TOKEN, 4)

    def token_to_id(self, token):
        """Chuyển đổi 1 token (string) sang ID (int)"""
        return self.vocab.get(token, self.unk_token_id)

    def id_to_token(self, idx):
        """Chuyển đổi 1 ID (int) sang token (string)"""
        return self.processor.id_to_token.get(idx, tk_lib.UNK_TOKEN)

    def encode(self, text):
        _, ids = self.processor.encode(text)
        class Encoding:
            def __init__(self, i): 
                self.ids = i
            def __len__(self):
                return len(self.ids)
                
        return Encoding(ids)
    
    def decode(self, ids):
        return self.processor.decode(ids)
        
    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return self.vocab_size