import logging
import numpy as np
import pandas as pd
from gensim.models import word2vec
from utilities import nlp_text_preprocessing_utility

# from bs4 import BeautifulSoup #TODO: delete
# import re #TODO: delete

# from nltk.corpus import stopwords #TODO: delete
import nltk.data

#TODO: delete
# def review_to_wordlist(raw, remove_stopwords=False):
#     ''' Given a paragraph, returns a list of words. '''

#     review_text = BeautifulSoup(raw).get_text()
#     letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
#     words = letters_only.lower().split()
#     if remove_stopwords:
#         stops = set(stopwords.words("english"))
#         words = [w for w in words if not w in stops]
#     return words


#TODO: delete
# def review_to_sentences(raw, tokenizer, remove_stopwords=False):
#     ''' Given a paragraph, returns a list of sentences of words (list of list of words). '''

#     raw_sentences = tokenizer.tokenize(raw.decode('utf-8').strip())
#     sentences = []
#     for raw_sentence in raw_sentences:
#         if len(raw_sentence) > 0:
#             sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
#     return sentences

if __name__ == '__main__':
    ''' Generate word2vec model to be used for word vector feature generation. '''

    # This will help create nice output messages.
    logging.basicConfig(format='%(asctime)s: %(levelname)s : %(message)s', level=logging.INFO)

    # SOP of getting the data set.
    train = pd.read_csv('../data/labeledTrainData.tsv', header=0, quoting=3, delimiter='\t')
    test = pd.read_csv('../data/testData.tsv', header=0, quoting=3, delimiter='\t')
    unlabeled_train = pd.read_csv('../data/unlabeledTrainData.tsv', header=0, quoting=3, delimiter='\t')

    # Convert all paragraphs into a list of sentences of words (list of lists of words).
    sentences = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += nlp_text_preprocessing_utility.review_to_sentences(review, tokenizer)

    print "Parsing sentences from unlabeled training set"
    for review in unlabeled_train["review"]:
        sentences += nlp_text_preprocessing_utility.review_to_sentences(review, tokenizer)

    # Create the word2vec model and save it.
    print "Training Word2Vec model..."
    model = word2vec.Word2Vec(sentences, \
                            workers=4, \
                            size=1000, \
                            min_count=40, \
                            window=10, \
                            sample=1e-3)
    model.init_sims(replace=True)
    model.save('../models/1000features_40minwords_10context')
