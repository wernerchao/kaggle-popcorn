import numpy as np
import pandas as pd

class train_predict_submit(object):
    ''' Train and predict with the specified model, vectorized data, data label, test set row id
        and output a submission file with the specified name. '''

    @staticmethod
    def train_predict_submit(model_func, name, vec_train_data, vec_test_data, target, test_id):
        ''' Train and predict with the specified model, vectorized data, data label, test set row id
        and output a submission file with the specified name. '''

        # dense_train_features = vec_train_data.toarray()
        # dense_test_features = vec_test_data.toarray()
        dense_train_features = vec_train_data
        dense_test_features = vec_test_data

        model = model_func
        print "Training %s..." % (name)
        model.fit(dense_train_features, target)
        print "Predicting with %s..." % (name)
        pred = model.predict(dense_test_features)
        pred_df = pd.DataFrame(data={'id':test_id, 'sentiment':pred})
        pred_df.to_csv("Ensemble_Submission_{}.csv".format(name), index=False, quoting=3)
        return pred_df
