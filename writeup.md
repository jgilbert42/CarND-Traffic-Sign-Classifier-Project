#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image4]: ./test-images/children-crossing.jpg "Children Crossing"
[image5]: ./test-images/direction-down-right.jpg "Direction Down Right"
[image6]: ./test-images/no-stop.jpg "No Stopping"
[image7]: ./test-images/quayside-or-riverbank.jpg "Quayside or Riverbank"
[image8]: ./test-images/speed-limit-60.jpg "Speed Limit 60kph"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jgilbert42/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the jupyter notebook.  

I used python and the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

I also printed the counts for each sign class.  There are clearly some classes
with ~10x more training images.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the jupyter notebook.  

I displayed a few random images and read in the sign labels from signnames.csv.
Some of these images I could not tell what they were.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the jupyter notebook.

I normalized the images using the equation from the lecture video, (x -
128)/128.  Centering the data seems to be referenced in many places as
important, but is done in many different ways.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The images were already separated into training, validation, and test sets.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

The first model is LeNet defined in LeNet() in block 7

My second and final model, defined in NewNet(), based on example code from the
TensorFlow tutorials on tensorflow.org.  It is also similar to the model
described in a paper referenced in the Udacity course material that did well on
the German Sign Test.

| Layer         	    |     Description	        		            | 
|:---------------------:|:---------------------------------------------:| 
| Input         	    | 32x32x3 RGB image   				            | 
| Convolution 5x5  	    | 1x1 stride, same padding, outputs 32x32x16 	|
| RELU			        |						                        |
| Max pooling	      	| 2x2 stride, outputs 16x16x32			        |
| Convolution 5x5     	| 2x2 stride, same padding, outputs 16x16x64 	|
| RELU			        |						                        |
| Max pooling	      	| 2x2 stride, outputs 8x8x64 			        |
| Flatten	     	    | outputs 4096                                  |
| Fully connected	    | outputs 1024 					                |
| Dropout		        | outputs 1024					                |
| Fully connected	    | outputs 43 					                |

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Functions for training, evaluating, and testing are in code block 10.  The
following cell blocks train and test the models.  To train the models, I used
an Adam Optimizer.

The LeNet training used a batch size of 128 for 150 epochs with a learning rate
of 0.001 for an approximately 94% validation accuracy and 93% test accuracy.

The NewNet training used a batch size of 32 for 500 epochs with a learning rate
of 0.001 for an approximately 97.5% validation accuracy and 95.1% test accuracy.

The model checks validation accuracy each epoch iteration and saves the model
if it's a higher accuracy.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the jupyter notebook.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.5%
* test set accuracy of 95.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

As previously mentioned, I started with the LeNet model and some basic
preprocessing based on the recommendation in the Udacity course material.

* What were some problems with the initial architecture?

After many different runs with different hyperparameters, I never got a
validation accuracy above ~94%.  So, I figured switching models is probably
better than more hyperparameter exploration.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I initially had some problems with high training accuracy, but ~80% validation.
It turned out to be due to forgetting to preprocess the validation and test
sets rather than the model.  The second model was tried since it was similar to
the one in the paper referenced in the Udacity course that reached fairly good
accuracy of ~98%. I was never able to get this high, but they were converting
their images to YUV.  I tried converting to HSV, but that appeared to make
things worse.

* Which parameters were tuned? How were they adjusted and why?

I primarily tuned the number of epochs, batch size, and learning rates.  I
found [How to choose a neural network's
hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters)
to be a useful explanation as well as [Setting up the Data
Model](http://cs231n.github.io/neural-networks-2/) from Stanford CS231n.  Batch size seemed to have
a significant affect on speed of epoch execution and required appropriate
learning rate adjustments to get good accuracy.  However, lower batch sizes
than 128 seemed to have better accuracy results.

* What are some of the important design choices and why were they chosen?

The models were chosen based on existing research.  In general, convolutational
neural networks seem to work well for image classification problems and require
less computation that fully connected layers.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

[image4]: ./test-images/children-crossing.jpg "Children Crossing"
[image5]: ./test-images/direction-down-right.jpg "Direction Down Right"
[image6]: ./test-images/no-stop.jpg "No Stopping"
[image7]: ./test-images/quayside-or-riverbank.jpg "Quayside or Riverbank"
[image8]: ./test-images/speed-limit-60.jpg "Speed Limit 60kph"

Images 6 and 7 are not in the training set as there are more than 43 German
traffic signs.  I included them anyway to see what kind of output they would
give even though it would be wrong.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the jupyter notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        | 
|:---------------------:|:-------------------------:|
| Children Crossing    	| Children Crossing		    | 
| Direction Down Right	| Keep Right				|
| No Stopping		    | Stop                      |
| Quayside or Riverbank	| Bicycles Crossing    		|
| 60 km/h	      	    | 80km/h    				|

The model was able to correctly predict 2 of the 5 traffic signs, which gives
an accuracy of 40%. This compares unfavorably to the accuracy on the test set,
but 2 of the images weren't in the training set.  Of the 3 images in the
training set, 2 of 3 were recognized which is 66% and the speed limit was close
(write sign, wrong limit).

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell
of the Ipython notebook.

The top predictions always being 1, even when wrong,  with all others 0 seem to
be too certain.  Only the 4th image had a non-zero second probability and it
was tiny.  Maybe this indicates more trainable parameters are needed or better
training data.

Probability | #  | Sign Label

1.0         | 38 | Keep Right
0           | 0  | Speed limit (20km/h)
0           | 1  | Speed limit (30km/h)
0           | 2  | Speed limit (50km/h)
0           | 3  | Speed limit (60km/h)

1.0         | 28 | Children Crossing
0           | 0  | Speed limit (20km/h)
0           | 1  | Speed limit (30km/h)
0           | 2  | Speed limit (50km/h)
0           | 3  | Speed limit (60km/h)

1.0         | 29 | Bicycles crossing
0           | 0  | Speed limit (20km/h)
0           | 1  | Speed limit (30km/h)
0           | 2  | Speed limit (50km/h)
0           | 3  | Speed limit (60km/h)

1.0         | 14 | Stop
2.15e-19    | 17 | No Entry
0           | 0  | Speed limit (20km/h)
0           | 1  | Speed limit (30km/h)
0           | 2  | Speed limit (50km/h)

1.0         | 5  | Speed limit (80km/h)
0           | 0  | Speed limit (20km/h)
0           | 1  | Speed limit (30km/h)
0           | 2  | Speed limit (50km/h)
0           | 3  | Speed limit (60km/h)

