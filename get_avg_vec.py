import numpy as np
import pandas as pd
import time

from bs4 import BeautifulSoup
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from utilities import nlp_text_preprocessing_utility


def makeAvgFeatureVec(words, model, num_features):
    ''' Get one average of feature vector from one review. '''

    featureVec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            n_words += 1.
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, n_words)
    return featureVec

def getAllAvgFeatureVec(reviews, model, num_features):
    ''' Get all average of feature vector from all reviews in the data set. '''

    allFeatureVec = np.zeros((len(reviews), num_features), dtype='float32')
    for i, item in enumerate(reviews):
        allFeatureVec[i] = makeAvgFeatureVec(item, model, num_features)
        if i%5000 == 0:
            print 'Making avg features for %d reviews of %d: ' % (i, len(reviews))
    return allFeatureVec

if __name__ == '__main__':

    train = pd.read_csv('data/labeledTrainData.tsv', header=0, quoting=3, delimiter='\t')
    test = pd.read_csv('data/testData.tsv', header=0, quoting=3, delimiter='\t')
    unlabeled_train = pd.read_csv('data/unlabeledTrainData.tsv', header=0, quoting=3, delimiter='\t')
    model = Word2Vec.load("models/300features_40minwords_10context")

    clean_train_reviews = nlp_text_preprocessing_utility.get_all_cleaned(train['review'], True, False)
    clean_test_reviews = nlp_text_preprocessing_utility.get_all_cleaned(test['review'], True, False)

    print 'Averaging feature vectors for train set...'
    train_data_vec = getAllAvgFeatureVec(clean_train_reviews, model, 300)

    print 'Averaging feature vectors for test set...'
    test_data_vec = getAllAvgFeatureVec(clean_test_reviews, model, 300)

    pred_df = pd.DataFrame(data=train_data_vec, index=train['id'])
    pred_df.to_csv("Word2Vec_Train_Features.csv", index=False, quoting=3)

    pred_df = pd.DataFrame(data=test_data_vec, index=train['id'])
    pred_df.to_csv("Word2Vec_Test_Features.csv", index=False, quoting=3)
