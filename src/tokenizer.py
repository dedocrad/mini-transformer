class CharTokenizer:
    """
    Посимвольный токенизатор с добавлением специальных токенов <pad>, <bos>, <eos>.

    Args:
        chars (str): строка всех допустимых символов.
    """
    def __init__(self, chars: str):
        self.chars = ['<pad>', '<bos>', '<eos>'] + list(chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
    
    def encode(self, text: str):
        if not isinstance(text, str):
            raise ValueError(f"Expected str, got {type(text)}: {repr(text)}")
        return [self.bos_id] + [self.char_to_idx[c] for c in text] + [self.eos_id]

    def decode(self, token_ids: list):
        # Убираем <bos>, <eos>, <pad>
        tokens = []
        for t in token_ids:
            if t == self.eos_id or t == self.pad_id:
                break
            if t != self.bos_id:
                tokens.append(self.idx_to_char[t])
        return ''.join(tokens)