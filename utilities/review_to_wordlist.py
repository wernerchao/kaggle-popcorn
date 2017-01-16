import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

class nlp_text_preprocessing_utility(object):
    ''' A utility class for text preprocessing raw HTML into
    a list of words for further NLP methods. '''

    @staticmethod
    def review_to_wordlist(raw, remove_stopwords=False, join_with_space=False):
        ''' Clean one review with optional removing stop words, & join the output with space. '''

        example = BeautifulSoup(raw, 'html5lib').get_text()
        letters_only = re.sub('[^a-zA-Z]', ' ', example)
        words = letters_only.lower().split()
        # Make a set of stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            meaningful_words = [w for w in words if not w in stop_words]
        if join_with_space:
            return ' '.join(meaningful_words)
        else:
            return words


    @staticmethod
    def get_all_cleaned(raw, remove_stopwords=False, join_with_space=False):
        ''' Loop through all reviews, and preprocess all reviews. '''

        num_reviews = raw.size
        clean_reviews = []
        print "Cleaning and parsing the data set movie reviews...\n"
        for i in range(num_reviews):
            if i%5000 == 0:
                print 'Review %d of %d\n' %(i, num_reviews)
            clean_reviews.append(nlp_text_preprocessing_utility.review_to_wordlist(raw[i], \
                                                                                   remove_stopwords, \
                                                                                   join_with_space))
        return clean_reviews
