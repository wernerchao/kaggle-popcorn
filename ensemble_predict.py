import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from utilities import train_predict_submit

if __name__ == '__main__':
    train = pd.read_csv('data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    test = pd.read_csv('data/testData.tsv', header=0, delimiter='\t', quoting=3)

    train_bow = pd.read_csv('nlp_output/Bag_of_Words_Train_Features.csv')
    test_bow = pd.read_csv('nlp_output/Bag_of_Words_Test_Features.csv')

    train_w2v = pd.read_csv('nlp_output/Word2Vec_Train_Features.csv')
    test_w2v = pd.read_csv('nlp_output/Word2Vec_Test_Features.csv')

    train_bow_w2v = np.hstack((train_bow, train_w2v))
    test_bow_w2v = np.hstack((test_bow, test_w2v))

    # Logistic Regression
    lr = LogisticRegression(class_weight="auto")
    pred_lr = train_predict_submit.train_predict_submit(lr, \
                                                        'LogisticRegression', \
                                                        train_bow_w2v, \
                                                        test_bow_w2v, \
                                                        train['sentiment'], \
                                                        test['id'])

    # Linear Support Vector Classifier
    lrsvc = LinearSVC()
    pred_df = train_predict_submit.train_predict_submit(lrsvc, \
                                                    'LinearSVC', \
                                                    train_bow_w2v, \
                                                    test_bow_w2v, \
                                                    train['sentiment'], \
                                                    test['id'])

    # SGDClassifier
    sgdc = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
    pred_sgdc = train_predict_submit.train_predict_submit(sgdc, \
                                                            'SGDClassifier', \
                                                            train_bow_w2v, \
                                                            test_bow_w2v, \
                                                            train['sentiment'], \
                                                            test['id'])

    # Naive Bayes
    nb = GaussianNB()
    pred_nb = train_predict_submit.train_predict_submit(nb, \
                                                        'GaussianNB', \
                                                        train_bow_w2v, \
                                                        test_bow_w2v, \
                                                        train['sentiment'], \
                                                        test['id'])

    print "Writing results..."
    pred_ensemble = (pred_nb.iloc[:,1]*0.2) + (pred_sgdc.iloc[:,1]*1.0)
    pred_df = pd.DataFrame(data={'id':test['id'], 'sentiment':pred_ensemble})
    pred_df.to_csv("BoW_W2V_Submission_{}.csv".format('ensemble'), index=False, quoting=3)

    # Multinomia Naive Bayes
    mnb = MultinomialNB(alpha=0.0005)
    pred_mnb = train_predict_submit.train_predict_submit(mnb, \
                                                        'MultinomialNB', \
                                                        train_bow_w2v, \
                                                        test_bow_w2v, \
                                                        train['sentiment'], \
                                                        test['id'])
