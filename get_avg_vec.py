import numpy as np
import pandas as pd
import time

from bs4 import BeautifulSoup
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def review_to_wordlist(raw, remove_stopwords=False):
    ''' Given a paragraph, returns a list of words. '''
    
    review_text = BeautifulSoup(raw).get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

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

    train = pd.read_csv('labeledTrainData.tsv', header=0, quoting=3, delimiter='\t')
    test = pd.read_csv('testData.tsv', header=0, quoting=3, delimiter='\t')
    unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', header=0, quoting=3, delimiter='\t')
    model = Word2Vec.load("300features_40minwords_10context")

    clean_train_reviews = []
    for i, review in enumerate(train['review']):
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
        process = threading.currentThread()
        if i%1000==0 or (i-1)%1000==0:
            print "Review %d of %d. Worker: %s" % (i, len(train['review']), process)
            print time.ctime(time.time())

    clean_test_review = []
    for i, review in enumerate(test['review']):
        clean_test_review.append(review_to_wordlist(review, remove_stopwords=True))
        if i%2500 == 0:
            print "Review %d of %d" % (i, len(test['review']))
            print time.ctime(time.time())

    print 'Averaging feature vectors for train set...'
    train_data_vec = getAllAvgFeatureVec(clean_train_reviews, model, 300)

    print 'Averaging feature vectors for test set...'
    test_data_vec = getAllAvgFeatureVec(clean_test_review, model, 300)

    # Predict the avg feature vec using random forest
    print "Fitting a random forest to labeled training data..."
    forest = RandomForestClassifier( n_estimators = 100 )
    forest.fit(train_data_vec, train["sentiment"])
    rf_pred = forest.predict(test_data_vec)

    # Write the test results 
    output = pd.DataFrame(data={"id":test["id"], "sentiment":rf_pred})
    output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
