import tokenizer as tk_lib

class TokenizerWrapper:
    def __init__(self, vocab_path="vocab_wiki.json"):
        self.processor = tk_lib.WordPieceTokenizer()
        self.processor.load_vocab(vocab_path)
        self.vocab_size = len(self.processor.vocab)
        self.pad_token_id = self.processor.vocab.get(tk_lib.PAD_TOKEN, 0)
        self.mask_token_id = self.processor.vocab.get(tk_lib.MASK_TOKEN, 4)
        self.cls_token_id = self.processor.vocab.get(tk_lib.CLS_TOKEN, 2)
        self.sep_token_id = self.processor.vocab.get(tk_lib.SEP_TOKEN, 3)
        self.unk_token_id = self.processor.vocab.get(tk_lib.UNK_TOKEN, 1)

    def encode(self, text):
        _, ids = self.processor.encode(text)
        ids = [self.cls_token_id] + ids + [self.sep_token_id]
        
        class Encoding:
            def __init__(self, i): 
                self.ids = i
            def __len__(self):
                return len(self.ids)
                
        return Encoding(ids)
    
    def decode(self, ids):
        return self.processor.decode(ids)

    def __len__(self):
        return self.vocab_size