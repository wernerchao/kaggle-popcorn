# kaggle-popcorn
Kaggle popcorn contest

Steps:
1) Run nlp_methods/bag_of_words.py file - Generate train/test sets of bag of words features for model training/predicting.
2) Run nlp_methods/model_word_to_vec.py file - Generate a word2vec model to be further processed into features vectors.
3) Run nlp_methods/get_avg_vec.py file - Generate features of word vectors from word2vec model. 
                                         The generated features will be used for model training/predicting.
4) Run predict/ensemble_predict.py file - Make prediction based on (bag_of_words + word2vec) features.
                                          Ensemble Naive Bayes & SGD Classifier prediction.
# TODO: bag_of_centroids.py
# TODO: ensemble_predict.py