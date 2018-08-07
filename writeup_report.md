
# **Behavioral Cloning** 

## Writeup

### Intro
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/data_hist.png "Steering Angle Histogram"
[image2]: ./examples/batch_hist_01.png "Batch Histogram 1"
[image3]: ./examples/batch_hist_02.png "Batch Histogram 2"
[image4]: ./examples/batch_hist_03.png "Batch Histogram 3"
[image5]: ./examples/model.png "Model Architecture"
[image6]: ./examples/center1.jpg "Center Driving"
[image7]: ./examples/recovery1.jpg "Recovery Driving 1"
[image8]: ./examples/recovery2.jpg "Recovery Driving 2"
[image9]: ./examples/recovery3.jpg "Recovery Driving 3"
[image10]: ./examples/flip1.jpg "Not-Flipped Image"
[image11]: ./examples/flip2.jpg "Flipped Image"
[image12]: ./examples/translate1.jpg "Translated Image 1"
[image13]: ./examples/translate2.jpg "Translated Image 2"
[image14]: ./examples/jungle1.jpg "Jungle training 1"
[image15]: ./examples/jungle2.jpg "Jungle training 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### 1. Files Submitted & Code Quality

#### 1.1 Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 1.2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 1.3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### 2. Model Architecture and Training Strategy

#### 2.1. An appropriate model architecture has been employed
My model consists of the NVIDIA architecture presented in the lecture videos. The input is cropped in a network layer (line 134) to make use of the gpu during training. The five convolutional layers are built with three 5×5 sized and two 3×3 sized filters and depths between 24 and 64 (model.py lines 133-147). The model includes 'relu' functions as activations to introduce nonlinearity (line 136-145), and the data is normalized in the model using a Keras lambda layer (line 135).

#### 2.2. Attempts to reduce overfitting in the model
The model contains a dropout layer in order to reduce overfitting (model.py line 141). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 72). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 2.3. Model parameter tuning
The model used an adam optimizer and mean square error, with the learning rate lowered to 1e-4 (line 160). 

#### 2.4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I have created an auxiliary function to be able to store various measurements, and then combine a customized set of them as needed. I recorded recovery routes, extra data with turns or dirt edges, and laps with backward direction, all in order for the model to generalize better. I also made use of the side cameras to simplify collecting recovery type data.

### 3. Model Architecture and Training Strategy

#### 3.1. Solution Design Approach
The overall strategy for deriving a model architecture was to start from simple and see how far it takes me. A LeNet type architecture trained very fast and performed well, but could not handle some exceptionary locations on the track. The NVIDIA architecture was already optimized for such a task, so it seemed perfect for the next step as a more sophisticated model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. At first I collected my custom data and then I added more and more until a point where memory was a problem. So I implemented the training with generators. I also realized that quality control over the added data is important, and manually sorting out the images was a tedious work. Therefore I tended to use the provided dataset, knowing that it is a high amount of data that can generally be trusted.

The main problem with the data was distribution. On the histogram of the steering angles, it can be seen that the training data consists mostly of straight driving, and this biases the learning process. Every time the model is uncertain, the "safe bet" is the straight driving, which results in leaving the track.

![alt text][image1]

To augment the data, I tried using various image processing techniques, such as conversion to other image spaces, brightness shifting, shearing, translating, resizing and cropping. The most useful of them proved to be cropping and horizontal translating, as it mimics the side camera offsets and steering angle offsets. With the translating, I was able to broaden the spectrum of the dataset in the generated batches that were fed to the neural network:

![alt text][image2]
![alt text][image3]
![alt text][image4]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 3.2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer (type)                | Output Shape             | Param #  |
| --------------------------- |:------------------------:| --------:|
| cropping2d_1 (Cropping2D)   | (None, 80, 320, 3)       | 0        |
| lambda_1 (Lambda)           | (None, 80, 320, 3)       | 0        |
| conv2d_1 (Conv2D)           | (None, 38, 158, 24)      | 1824     |
| conv2d_2 (Conv2D)           | (None, 17, 77, 36)       | 21636    |
| conv2d_3 (Conv2D)           | (None, 7, 37, 48)        | 43248    |
| conv2d_4 (Conv2D)           | (None, 5, 35, 64)        | 27712    |
| conv2d_5 (Conv2D)           | (None, 3, 33, 64)        | 36928    |
| dropout_1 (Dropout)         | (None, 3, 33, 64)        | 0        |
| flatten_1 (Flatten)         | (None, 6336)             | 0        |
| dense_1 (Dense)             | (None, 100)              | 633700   |
| dense_2 (Dense)             | (None, 50)               | 5050     |
| dense_3 (Dense)             | (None, 10)               | 510      |
| dense_4 (Dense)             | (None, 1)                | 11       |

#### 3.3. Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover once it inevitably starts drifting to the sides of the track. These images show what a recovery looks like:

![alt text][image7]
![alt text][image8]
![alt text][image9]

To augment the data sat, I also flipped images and angles thinking that this would eliminate the left turn bias, arising from the directionality of the track. For example, here is an image that has then been flipped:

![alt text][image10]
![alt text][image11]

I used the translated images and corrected steering angles described before to eliminate the bias towards straight driving. These images look as follows: 

![alt text][image12]
![alt text][image13]

After the collection process, I had 48,216 number of data points. The preprocessing of the data includes the cropping, and normalizing of the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs turned out to be 2 with 15,000 steps as more epochs did not improve the validation accuracy.

#### 3.4 Jungle Update
In the meantime, I trained the model some more to handle the jungle track with 2 laps of training data in that track, then after succesful testing, I also added video2.mp4 to the recordings. The data augmentation techniques helped to handle the sharp turns and changes in pitch.

![alt text][image14]
![alt text][image15]
