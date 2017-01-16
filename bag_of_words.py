import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from utilities import nlp_text_preprocessing_utility
from utilities import train_predict_submit


if __name__ == '__main__':
    train = pd.read_csv('data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    test = pd.read_csv('data/testData.tsv', header=0, delimiter='\t', quoting=3)
    unlabeled_train = pd.read_csv('data/unlabeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

    clean_train_reviews = nlp_text_preprocessing_utility.get_all_cleaned(train['review'], True, True)
    unlabeled_clean_train_reviews = nlp_text_preprocessing_utility.get_all_cleaned(unlabeled_train['review'], True, True)
    clean_test_reviews = nlp_text_preprocessing_utility.get_all_cleaned(test['review'], True, True)

    print "Creating the bag of words for train data...\n"
    vectorizer = CountVectorizer(analyzer='word', \
                                tokenizer=None, \
                                preprocessor=None, \
                                stop_words=None, \
                                max_features=100)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    print "Transforming test data according to bag of words...\n"
    test_data_features = vectorizer.transform(clean_test_reviews)

    # Logistic Regression
    lr = LogisticRegression(class_weight="auto")
    pred_lr = train_predict_submit.train_predict_submit(lr, \
                                                        'LogisticRegression', \
                                                        train_data_features, \
                                                        test_data_features, \
                                                        train['sentiment'], \
                                                        test['id'])

    # Linear Support Vector Classifier
    lrsvc = LinearSVC()
    pred_df = train_predict_submit.train_predict_submit(lrsvc, \
                                                 'LinearSVC', \
                                                 train_data_features, \
                                                 test_data_features, \
                                                 train['sentiment'], \
                                                 test['id'])

    # SGDClassifier
    sgdc = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
    pred_sgdc = train_predict_submit.train_predict_submit(sgdc, \
                                                          'SGDClassifier', \
                                                          train_data_features, \
                                                          test_data_features, \
                                                          train['sentiment'], \
                                                          test['id'])

    # Naive Bayes
    nb = GaussianNB()
    pred_nb = train_predict_submit.train_predict_submit(nb, \
                                                        'GaussianNB', \
                                                        train_data_features, \
                                                        test_data_features, \
                                                        train['sentiment'], \
                                                        test['id'])

    # Multinomia Naive Bayes
    mnb = MultinomialNB(alpha=0.0005)
    pred_mnb = train_predict_submit,train_predict_submit(mnb, \
                                                         'MultinomialNB', \
                                                         train_data_features, \
                                                         test_data_features, \
                                                         train['sentiment'], \
                                                         test['id'])
