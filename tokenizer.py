import jieba
from transformers import BertTokenizer

class T5TokenizerFast(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def pre_tokenizer(self, x):
        return jieba.cut(x, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            split_tokens.append(text if text in self.vocab else super()._tokenize(text))
        return split_tokens
    

