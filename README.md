# SelfDrivingCar-P3-Behavior Cloning

In this project, I use deep neural networks and convolutional neural networks to clone driving behavior. The image data and steering angle data are collected from a unity simulator, where good driving manner will be recorded through three front-facing cameras located on the left/center/right of the car. The model is trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle which drive the car run  autonomously around the track. 

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior including information about the steering measurement, throttle, brake and speed of the vehicle
- Data Preprocessing: 
  - Data augmentation: (1) since there are not so many left turn cases collected during simulator, data are augmented through flipping images and taking the opposite sign of the steering measurement. (2) Use all the three cameras with some calibration/correction (3) constantly wander off to the side of the road and then steer back to the middle to teach the car how to drive from the side of the road back toward the center line.
  - Data normalization: zero-mean
  - Image Crop: Since the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car, I crop the top portion as they have nothing to do with the model training and will cost extra computation.
- Build, a convolution neural network in Keras that predicts steering angles from images:
  - I start with LeNet-5
  - Then I found Nivida 9-layer network, which is more complicate but perform better
- Train and validate the model with a training and validation set: The model is evaluate by the error metric of mean squared error. When the mean squared error is high on both a training and validation set, the model is underfitting. When the mean squared error is low on a training set but high on a validation set, the model is overfitting. Collecting more data help improve a model when the model is overfitting.
- Test that the model successfully drives around track one without leaving the road



Nvidia 9 layer Covolution Neural Network



![image](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)







### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.