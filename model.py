import pickle as pkl
import numpy as np
import time


class Model:
    def __init__(self):
        with open('dummy_lr_model.pkl', 'rb') as f:
            self.model = pkl.load(f)

    def run(self, input_data):
        # preprocess
        # assuming input data has the following format
        # {
        #   sepal_length: val,
        #   sepal_width: val,
        #   petal_length: val,
        #   petal_width: val,
        # }
        processed_data = np.array([[input_data['sepal_length'],
                                    input_data['sepal_width'],
                                    input_data['petal_length'],
                                    input_data['petal_width']]])

        # predict
        label = self.model.predict(processed_data)[0]
        proba = self.model.predict_proba(processed_data)
        score = proba[0, label]

        # add some random fluctuation to some random samples
        if np.random.rand() < 0.2:
            score += np.random.uniform(-0.2, 0.2)

        # post-process
        response = {
            'decision': {
                'class': int(label),
                'score': float(score)
            }
        }
        time.sleep(2) # lets assume this model takes time to process
        return response
