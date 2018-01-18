import csv
import cv2

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader: 
		lines.append(line)

images = [] # X_train
measurements = [] # y_train
for line in lines:
	# Only use center camera
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = 'data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
    
	'''
	# Use all three cameras
	correction = 0.2
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/'[-1])
		current_path = 'data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		if i == 0: # Center no correction
			measurements.append(measurement)
		elif i == 1: # left camera
			measurements.append(measurement + correction)
		else:
			measurements.append(measurement - correction)
	'''



# augment data
import numpy as np
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images,measurements):
	augmented_image = np.fliplr(image)
	augmented_measurement = - measurement

	augmented_images.append(image)
	augmented_images.append(augmented_image)
	augmented_measurements.append(measurement)
	augmented_measurements.append(augmented_measurement)

images = augmented_images
measurements = augmented_measurements



X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential,Model 
from keras.layers import Input,Flatten,Dense


model = Sequential()
'''
# model 1:
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))


# model 2:
# Normalized the image with zero-mean center
from keras.layers.core import Lambda
model.add(Lambda x: x / 255.0 - 0.5, input_shape = (160,320,3))
model.add(Flatten())


# model 3:
# LeNet model
from keras.layers.core import Lambda
import keras.backend as ktf
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
model.add(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3))
model.add(Convolution2D(6,5,5,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

'''
# model 4:
# Add cropping with Nividia model -- 5 Conv Layers with 3 Fully-connected Layers
from keras.layers.core import Lambda
import keras.backend as ktf
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Cropping2D

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((50,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64,5,5,activation = 'relu'))
model.add(Convolution2D(64,3,3,activation = 'relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


print('model is ready for training')


model.compile(loss='mse',optimizer='adam')
print('start training')
model.fit(X_train, y_train, validation_split = 0.2, batch_size = 128, shuffle = True, nb_epoch =7)

'''
# Recording/Visualizing loss
import matplotlib.pyplot as pyplot
history_object = model.fit_generator(train_generator, samples_per_ephoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples),nb_epoch = 5, verbose = 1)
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history_object['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squred error loss')
plt.xlabel('epoch')
plt.legend(['training_set','validation_set'])
plt.show()
'''

model.save('model.h5')
print('model saved')
exit()
