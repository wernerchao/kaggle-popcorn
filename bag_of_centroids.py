import numpy as np
import pandas as pd
import time

from bs4 import BeautifulSoup
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def create_bag_of_centroids( wordlist, word_centroid_map ):
    num_centroids = max( word_centroid_map.values() ) + 1
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

if __name__ == '__main__':

    train = pd.read_csv('labeledTrainData.tsv', header=0, quoting=3, delimiter='\t')
    test = pd.read_csv('testData.tsv', header=0, quoting=3, delimiter='\t')
    unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', header=0, quoting=3, delimiter='\t')
    model = Word2Vec.load("300features_40minwords_10context")

    # Make bag of centroids as features, and predict with random forest
    start = time.time() # Start time

    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / 50

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # End time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."

    # Zip the words to the vectors
    word_centroid_map = dict(zip( model.index2word, idx ))

    # Making features for train/test set using bag of centroids.
    train_centroids = np.zeros((train['review'].size, num_clusters), dtype='float32')
    test_centroids = np.zeros((test['review'].size, num_clusters), dtype='float32')
    for i, review in enumerate(clean_train_reviews):
        train_centroids[i] = create_bag_of_centroids(review, word_centroid_map)
    for i, review in enumerate(clean_test_review):
        test_centroids[i] = create_bag_of_centroids(review, word_centroid_map)

    forest = RandomForestClassifier(n_estimators = 100)
    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(train_centroids, train["sentiment"])
    rf_cent_pred = forest.predict(test_centroids)
    output = pd.DataFrame(data={"id":test["id"], "sentiment":rf_cent_pred})
    output.to_csv( "BagOfCentroids_rf.csv", index=False, quoting=3 )
