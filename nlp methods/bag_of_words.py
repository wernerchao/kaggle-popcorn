import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import re

import nltk
from nltk.corpus import stopwords

def text_preprocessing(raw):
    example = BeautifulSoup(raw).get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', example)
    words = letters_only.lower().split()
    # Make a set of stopwords
    stop_words = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stop_words]
    
    return (' '.join(meaningful_words))

def get_all_cleaned(raw):
    num_reviews=raw.size
    clean_reviews = []
    print "Cleaning and parsing the data set movie reviews...\n"
    for i in range(num_reviews):
        if i%5000 == 0:
            print 'Review %d of %d\n' %(i, num_reviews)
        clean_reviews.append(text_preprocessing(raw[i]))
    return clean_reviews

def tf_train_predict(model_func, name):
    ''' Train and predict with the specified model, and output a submission file with the specified name. '''
    dense_train_features = tf_train_data_features.toarray()
    dense_test_features = tf_test_data_features.toarray()
    
    model = model_func
    print "Training %s..." % (name)
    model.fit(dense_train_features, train['sentiment'])
    print "Predicting with %s..." % (name)
    pred = model.predict(dense_test_features)
    pred_df = pd.DataFrame(data={'id':test['id'], 'sentiment':lr_pred})
    pred_df.to_csv("Bag_of_Words_Submission_tf_{}.csv".format(name), index=False, quoting=3)
    return pred_df

if __name__ == '__main__':
    train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    test = pd.read_csv('testData.tsv', header=0, delimiter='\t', quoting=3)
    unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

    clean_train_reviews = get_all_cleaned(train['review'])
    unlabeled_clean_train_reviews = get_all_cleaned(unlabeled_train['review'])
    clean_test_reviews = get_all_cleaned(test['review'])

    print 'Tfidf vectorizing...'
    tf_vectorizer = TfidfVectorizer(min_df=2, \
                                max_df=0.95, \
                                max_features=200000, \
                                ngram_range=(1, 4), \
                                sublinear_tf=True)
    tf_vectorizer.fit(clean_train_reviews)
    tf_train_data_features = tf_vectorizer.transform(clean_train_reviews)
    tf_test_data_features = tf_vectorizer.transform(clean_test_reviews)

    # Logistic Regression
    lr = LogisticRegression(class_weight="auto")
    pred_lr = tf_train_predict(lr, 'LogisticRegression')

    # SVC
    lrsvc = LinearSVC()
    pred_df = tf_train_predict(lrsvc, 'LinearSVC')

    # SGDClassifier
    sgdc = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
    pred_sgdc = tf_train_predict(sgdc, 'SGDClassifier')

    # Naive Bayes
    nb = GaussianNB()
    pred_nb = tf_train_predict(nb, 'GaussianNB')

    # Multinomia Naive Bayes
    mnb = MultinomiaNB(alpha=0.0005)
    pred_mnb = tf_train_predict(mnb, 'MultinomiaNB')
