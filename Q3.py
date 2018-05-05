from keras_rcnn import preprocessing
from keras_rcnn import datasets

if __name__ == '__main__':
    training_dictionary, test_dictionary = datasets.load_data('a')
    categories = {'background': 0, 'object': 1}
    #generator = preprocessing.ObjectDetectionGenerator()

    print('All Done :)')
