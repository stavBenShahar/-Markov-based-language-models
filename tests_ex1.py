from UniGram import UnigramModel
from BiGram import BigramModel
import pickle


def test1():
    import spacy
    from collections import Counter

    # Sample dataset
    dataset = [
        {"text": "Natural language processing (NLP) is a field of computer science."},
        {"text": "Artificial intelligence and linguistics are important in NLP."},
        {"text": "NLP involves interactions between computers and human languages."}
    ]

    # Initialize spaCy
    nlp = spacy.load("en_core_web_sm")

    # Create and train the UnigramModel
    unigram_model = UnigramModel(dataset, nlp)
    unigram_model.train()

    # Calculate expected unigram probabilities manually
    word_counts = Counter()
    total_count = 0
    for line in dataset:
        doc = nlp(line['text'])
        for token in doc:
            if token.is_alpha:
                lemma = token.lemma_.lower()
                word_counts[lemma] += 1
                total_count += 1

    expected_probs = {word: count / total_count for word, count in word_counts.items()}

    # Compare with UnigramModel probabilities
    for word, expected_prob in expected_probs.items():
        model_prob = unigram_model.unigram_probs[word]
        assert abs(
            model_prob - expected_prob) < 0.01, f"Probability mismatch for '{word}': Expected {expected_prob}, got {model_prob}"

    print("Test1 passed")

def test2():
    import spacy
    from collections import defaultdict, Counter

    # Larger sample dataset
    dataset = [
        {"text": "Natural language processing is fascinating."},
        {"text": "Language models are useful in NLP."},
        {"text": "Processing language data requires models."},
        {"text": "Artificial intelligence is a key component of NLP."},
        {"text": "Understanding natural language is challenging."},
        {"text": "NLP techniques are evolving rapidly."},
        {"text": "Machine learning algorithms play a crucial role in language processing."},
        {"text": "Text data can be analyzed using NLP."},
        {"text": "Sentiment analysis is a common application of NLP."},
        {"text": "Linguistic features are important in natural language understanding."}
    ]

    # Initialize spaCy
    nlp = spacy.load("en_core_web_sm")

    # Create and train the BigramModel
    bigram_model = BigramModel(dataset, nlp)
    bigram_model.train()

    # Calculate expected bigram probabilities manually
    bigram_counts = defaultdict(Counter)
    word_counts = Counter()
    for line in dataset:
        doc = nlp(line['text'])
        prev_token = "START"
        for token in doc:
            if token.is_alpha:
                lemma = token.lemma_.lower()
                bigram_counts[prev_token][lemma] += 1
                word_counts[prev_token] += 1
                prev_token = lemma

    expected_probs = {prev_word: {word: count / word_counts[prev_word] for word, count in counts.items()} for
                      prev_word, counts in bigram_counts.items()}

    # Compare with BigramModel probabilities
    for prev_word, counts in expected_probs.items():
        for word, expected_prob in counts.items():
            model_prob = bigram_model.bigram_probs[prev_word].get(word, 0)
            assert abs(
                model_prob - expected_prob) < 0.01, f"Probability mismatch for ('{prev_word}', '{word}'): Expected {expected_prob}, got {model_prob}"

    print("Test2 passed!")

if __name__ == '__main__':
    with open('unigram_model.pkl', 'rb') as file:
        unigram = pickle.load(file)

    with open('bigram_model.pkl', 'rb') as file:
        bigram = pickle.load(file)

    test1()
    test2()