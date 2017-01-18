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
    clean_test_reviews = nlp_text_preprocessing_utility.get_all_cleaned(test['review'], True, True)

    print "Creating the bag of words for train data...\n"
    vectorizer = CountVectorizer(analyzer='word', \
                                tokenizer=None, \
                                preprocessor=None, \
                                stop_words=None, \
                                max_features=5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    print "Transforming test data according to bag of words...\n"
    test_data_features = vectorizer.transform(clean_test_reviews)

    pred_df = pd.DataFrame(data=train_data_features.toarray(), index=train['id'])
    pred_df.to_csv("Bag_of_Words_Train_Features.csv", index=False, quoting=3)

    pred_df = pd.DataFrame(data=test_data_features.toarray(), index=train['id'])
    pred_df.to_csv("Bag_of_Words_Test_Features.csv", index=False, quoting=3)
