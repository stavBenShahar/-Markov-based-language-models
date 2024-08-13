
from collections import defaultdict
import numpy as np


class UnigramModel:
    def __init__(self, dataset, nlp):
        self.unigram_probs = defaultdict(float)
        self.total_count = 0
        self.nlp = nlp
        self.dataset = dataset

    def train(self):
        for line in self.dataset:
            doc = self.nlp(line['text'])
            for token in doc:
                if token.is_alpha:
                    lemma = token.lemma_.lower()
                    self.unigram_probs[lemma] += 1
                    self.total_count += 1

        # Convert counts to log probabilities
        for word in self.unigram_probs:
            self.unigram_probs[word] = np.log(self.unigram_probs[word] / self.total_count)

    def predict_next_word(self):
        if not self.unigram_probs:
            return None
        return max(self.unigram_probs, key=self.unigram_probs.get)
