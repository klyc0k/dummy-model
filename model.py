import pickle as pkl
import numpy as np


class Model:
    def __init__(self):
        self.model = pkl.load('binary/dummy_lr_model.pkl')

    def run(self, input_data):
        # preprocess
        # assuming input data has the following format
        # {
        #   sepal_length: val,
        #   sepal_width: val,
        #   petal_length: val,
        #   petal_width: val,
        # }
        processed_data = np.array([input_data['sepal_length'],
                                   input_data['sepal_width'],
                                   input_data['petal_length'],
                                   input_data['petal_width']])

        # predict
        result = self.model.predict(processed_data)
        score = self.model.score(processed_data)

        # postprocess
        response = {
            'decision': {
                'class': result,
                'score': score
            }
        }
        return response
