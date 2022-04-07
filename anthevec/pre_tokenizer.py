import spacy

# https://github.com/explosion/spaCy/issues/5399#issuecomment-623593208
class PreTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words):
        return spacy.tokens.Doc(self.vocab, words=words)