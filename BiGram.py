from collections import defaultdict
import numpy as np


def default_dict_float():
    return defaultdict(float)


class BigramModel:
    def __init__(self, dataset, nlp):
        self.bigram_probs = defaultdict(default_dict_float)
        self.total_counts = defaultdict(int)
        self.nlp = nlp
        self.dataset = dataset

    def train(self):
        for line in self.dataset:
            doc = self.nlp(line['text'])
            prev_token = "START"
            for token in doc:
                if token.is_alpha:
                    lemma = token.lemma_.lower()
                    self.bigram_probs[prev_token][lemma] += 1
                    self.total_counts[prev_token] += 1
                    prev_token = lemma

        for prev_word, counts in self.bigram_probs.items():
            total_count = self.total_counts[prev_word]
            for word in counts:
                counts[word] = np.log(counts[word] / total_count)

    def predict_next_word(self, sentence):
        tokens = [token.lemma_.lower() for token in self.nlp(sentence) if token.is_alpha]
        last_word = tokens[-1] if tokens else "START"
        next_word_probs = self.bigram_probs[last_word]
        if not next_word_probs:
            return None
        next_word = max(next_word_probs, key=next_word_probs.get)
        return next_word

    def sentence_probability(self, sentence):
        doc = self.nlp(sentence)
        prev_token = "START"
        sentence_log_prob = 0.0
        for token in doc:
            if token.is_alpha:
                lemma = token.lemma_.lower()
                bigram_log_prob = self.bigram_probs[prev_token].get(lemma, float('-inf'))
                sentence_log_prob += bigram_log_prob
                prev_token = lemma
        return sentence_log_prob

    def calculate_perplexity(self, sentences):
        total_log_prob = 0.0
        total_words = 0

        for sentence in sentences:
            doc = self.nlp(sentence)
            prev_token = "START"
            for token in doc:
                if token.is_alpha:
                    lemma = token.lemma_.lower()
                    bigram_log_prob = self.bigram_probs[prev_token].get(lemma, float('-inf'))
                    total_log_prob += bigram_log_prob
                    prev_token = lemma
                    total_words += 1

        average_log_prob = total_log_prob / total_words
        perplexity = np.exp(-average_log_prob)
        return perplexity
