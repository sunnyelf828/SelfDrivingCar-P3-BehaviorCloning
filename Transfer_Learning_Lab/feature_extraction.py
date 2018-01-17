import pickle
import tensorflow as tf
# TODO: import Keras layers you need here

import numpy as np
from keras.layers import Input,Flatten,Dense
from keras.models import Model 

flags = tf.app.flags
FLAGS = flags.FLAGS


# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")

flags.DEFINE_string('epochs',50,'The number of epochs')
flags.DEFINE_string('batch_size',128,'The batch size.')

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    path = '/Users/chenm/Documents/Udacity_SelfDrivingCar/CarND-Transfer-Learning-Lab/'
    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    nb_class = len(np.unique(y_train))

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    #define the model
    input_shape = X_train.shape[1:]
    inp = Input(shape = input_shape)
    x = Flatten()(inp)
    x = Dense(nb_class,activation='softmax')(x)
    model = Model(inp,x)
    model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


    # TODO: train your model here
    # Attention: The CarND_Term1-Starter-Kit install keras version 1.2.1
    # model.fit (nb_epoch=) while in version 2.3.1 model.fit(epochs = )
    model.fit(X_train,y_train,batch_size=FLAGS.batch_size,nb_epoch=FLAGS.epochs,validation_data=(X_val,y_val),shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
