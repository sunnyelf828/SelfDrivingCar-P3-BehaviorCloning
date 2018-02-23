# SelfDrivingCar-P3-Behavior Cloning

In this project, I use deep neural networks and convolutional neural networks to clone driving behavior. The image data and steering angle data are collected from a unity simulator, where good driving manner will be recorded through three front-facing cameras located on the left/center/right of the car. The model is trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle which drive the car run  autonomously around the track. 

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior including information about the steering measurement, throttle, brake and speed of the vehicle
- Data Preprocessing: 
  - **Data augmentation**: (1) since there are not so many left turn cases collected during simulator, data are augmented through **flipping images** and taking the opposite sign of the steering measurement. (2) Use all the **three cameras with some calibration/correction** (3) constantly wander off to the side of the road and then steer back to the middle to teach the car how to drive from the side of the road back toward the center line.
  - **Data normalization**: zero-mean, lambda layer
  - **Image Crop**: Since the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car, I crop the top portion as they have nothing to do with the model training and will cost extra computation.
- Build, a convolution neural network in Keras that predicts steering angles from images:
  - I start with LeNet-5
  - Then I found **Nivida model network -- 5 Conv Layers with 3 Fully-connected Layers**, which is more complicate but perform better, I modified my model based on it.
- Train and validate the model with a training and validation set: The model is evaluate by the error metric of **mean squared error**. When the mean squared error is high on both a training and validation set, the model is underfitting. When the mean squared error is low on a training set but high on a validation set, the model is overfitting. Collecting more data help improve a model when the model is overfitting.
- Test that the model successfully drives around track one without leaving the road

### My project includes the following files:

- clone.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- method4 result/model4.h5 containing a trained convolution neural network
- method4 result/output_video.mp4 is the final output video
- README.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```
python drive.py model.h5
```





**original images (left/center/right) taken from simulator**

![left_2018_01_17_21_58_35_473](/Users/chenm/Documents/Udacity_SelfDrivingCar/GitHub_projects/SelfDrivingCar-P3-BehaviorCloning/training_data/left_2018_01_17_21_58_35_473.jpg)

![center_2018_01_17_21_58_35_473](/Users/chenm/Documents/Udacity_SelfDrivingCar/GitHub_projects/SelfDrivingCar-P3-BehaviorCloning/training_data/center_2018_01_17_21_58_35_473.jpg)

![right_2018_01_17_21_58_35_473](/Users/chenm/Documents/Udacity_SelfDrivingCar/GitHub_projects/SelfDrivingCar-P3-BehaviorCloning/training_data/right_2018_01_17_21_58_35_473.jpg)



**Nvidia 9 layer Covolution Neural Network**



![image](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)







### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of four convolutional neural network with 5x5 filter size, one convolutional neural network with 3x3 filter size and four fully connected layers.

The model includes RELU function to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains maxpooling layers in order to reduce overfitting as well as reduce computational cost by reducing the numbers of parameters to train.

The model was trained and validated on different data sets / epoch number to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.



### Model Architecture and Training Strategy

#### The final model is revised on the basis of Nividia model

```
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
```

To train the model, the epoch number is chosen as 8 with batch size 128. Also I shuffle the data set so as to eliminate the coheret effect of continous data.