from UniGram import UnigramModel
from BiGram import BigramModel
import spacy
from datasets import load_dataset
import numpy as np
import pickle


def interpolate_probability(unigram, bigram, sentence, lambda_bigram, lambda_unigram):
    doc = bigram.nlp(sentence)
    prev_token = "START"
    sentence_log_prob = 0.0
    for token in doc:
        if token.is_alpha:
            lemma = token.lemma_.lower()
            bigram_log_prob = bigram.bigram_probs[prev_token].get(lemma, float('-inf'))
            unigram_log_prob = unigram.unigram_probs.get(lemma, float('-inf'))
            interpolated_log_prob = np.logaddexp(np.log(lambda_bigram) + bigram_log_prob,
                                                 np.log(lambda_unigram) + unigram_log_prob)
            sentence_log_prob += interpolated_log_prob
            prev_token = lemma
    return sentence_log_prob


def calculate_interpolated_perplexity(unigram, bigram, sentences, lambda_bigram,
                                      lambda_unigram):
    total_log_prob = 0.0
    total_words = 0

    for sentence in sentences:
        prob = interpolate_probability(unigram, bigram, sentence, lambda_bigram, lambda_unigram)
        total_log_prob += prob
        total_words += len([token for token in bigram.nlp(sentence) if token.is_alpha])

    average_log_prob = total_log_prob / total_words
    perplexity = np.exp(-average_log_prob)
    return perplexity


def task1():
    print("--Test Question 1# --")
    nlp = spacy.load("en_core_web_sm")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    unigram = UnigramModel(dataset, nlp)
    unigram.train()
    print("Trained UniGram successfully")
    bigram = BigramModel(dataset, nlp)
    bigram.train()
    print("Trained BiGram successfully")

    with open('unigram_model.pkl', 'wb') as f:
        pickle.dump(unigram, f)
    with open('bigram_model.pkl', 'wb') as f:
        pickle.dump(bigram, f)


def task2(bigram):
    print("--Test Question 2# --")
    sentence_0 = "I have a house in"
    next_word = bigram.predict_next_word(sentence_0)
    print(f"Next word is: '{next_word}', so the complete sentence now  is '{sentence_0} {next_word}'.")


def task3(bigram):
    print("--Test Question 3# --")
    sentence_1 = "Brad Pitt was born in Oklahoma"
    sentence_2 = "The actor was born in USA"

    probability1 = bigram.sentence_probability(sentence_1)
    print(f"The probability of the sentence: '{sentence_1}' is {probability1:.3f}")

    probability2 = bigram.sentence_probability(sentence_2)
    print(f"The probability of the sentence: '{sentence_2}' is {probability2:.3f}")

    perplexity0 = bigram.calculate_perplexity([sentence_1, sentence_2])
    print(f"Perplexity of the sentences: {perplexity0:.3f}")


def task4(unigram, bigram):
    print("--Test Question 4# --")
    sentence_1 = "Brad Pitt was born in Oklahoma"
    sentence_2 = "The actor was born in USA"
    probability_of_sentence_1 = interpolate_probability(unigram, bigram, sentence_1, 2 / 3, 1 / 3)
    print(f"The probability of the sentence: '{sentence_1}' is {probability_of_sentence_1:.3f}")

    probability_of_sentence_2 = interpolate_probability(unigram, bigram, sentence_2, 2 / 3, 1 / 3)
    print(f"The probability of the sentence: '{sentence_2}' is {probability_of_sentence_2:.3f}")

    perplexity1 = calculate_interpolated_perplexity(unigram, bigram, [sentence_1, sentence_2], 2 / 3, 1 / 3)
    print(f"The interpolated perplexity of the sentences is: {perplexity1:.3f}")


if __name__ == '__main__':
    # Task 1 - Training the model and create pickle files for them
    task1()

    # Loads the models for further use
    with open('unigram_model.pkl', 'rb') as file:
        unigram_model = pickle.load(file)

    with open('bigram_model.pkl', 'rb') as file:
        bigram_model = pickle.load(file)

    # Predict the next word using the Bigram Model
    task2(bigram_model)

    # Compute the probability of two given sentences using the Bigram model and compute the perplexity
    task3(bigram_model)

    # Create a linear Interpolation smoothing using both model with given lambdas and compute the probability and
    # perplexity of two given sentences
    task4(unigram_model, bigram_model)
