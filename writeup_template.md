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

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/preprocess.png "PreProcessing"
[image3]: ./web_data/1.png "Int Traffic Sign 1"
[image4]: ./web_data/13.png "Int Traffic Sign 2"
[image5]: ./web_data/14.png "Int Traffic Sign 3"
[image6]: ./web_data/14_1.png "Int Traffic Sign 4"
[image7]: ./web_data/17.png "Int Traffic Sign 5"
[image8]: ./web_data/36.png "Int Traffic Sign 6"
[image9]: ./examples/SoftmaxProbs1.png "SoftMax Probs 1-3"
[image10]: ./examples/SoftmaxProbs2.png "SoftMax Probs 4-6"


## Rubric Points

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/pssrijith/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the cells 2 & 3 of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is : 34,799
* The size of test set is : 12630
* The shape of a traffic sign image is : (32, 32, 3)
* The number of unique classes/labels in the data set is : 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in cell 4 of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the histogram of each class distribution in the trainign and validation data set. The validation data set is a good representation of the training data set which shows that the dataset was sampled with class stratification

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to gra
yscale, normalization, etc.

The code for this step is contained in the cells 5, 6 and 7 of the IPython notebook.

As shown in cell 5, I picked an image (20 kmph speed limit sign) and tried out the follwing 2 conversions 
1. a YUV histogram equalized image : 
   convert the RGB image to YUV and do a histogram equalize on the y channel
2. a grayscale histogram equalized image)
   convert the RGB image to grayscale and do a histogram equalize on the grayscale image

Here is a screen shot of cell 5 output that shows the 3 images.

![alt text][image2]

I prefered the YUV hisotogram equalized method as it still keeps the colors. Signs have different colors and grayscaling will lose the ability to use that feature. However, I decided to try both mehtods YUV(in the main ipyton notebool) and Grayscale(code in a different notebook -  (https://github.com/pssrijith/CarND-Traffic-Sign-Classifier-Project/blob/master/GRAY_SC_HIST_EQ_Traffic_Sign_Classifier.ipynb)

In cell 6, I plotted 1 image for each of the 43 classes after pre processing them with the YUV histogram equalization.

Cell 7 contains the actual code where the preprocess method (YUV histogram equalization) for the data set is defined.  I also normalized the image data to bring down the scale within the range -0.5 to +0.5. This makes for easier computation and does not give undue weight to colors with higher values

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 
The training, validation and test data sets were already split. Here are the sizes

Training dataset size 34799 <br/>
Validation dataset size 4410<br/>
Testing dataset size 12630<br/>


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the  cell 10 of the ipython notebook. 

I implemented the LeNet-5 neural network architecture. I used the multiscale architecture approach (connecting outputs from both convolutional layer 1 and layer 2 outputs as input to the fully connected layer 3, as mentioned in the paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks  By Pierre Sermanet and Yann LeCun ](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)


My final model consisted of the following layers:


| Layer         		|             Description      											| 
|:---------------------:|:---------------------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   													| 
| Convolution 1: 5x5   	| 1x1 stride, VALID padding, outputs 28x28x32 							|
| RELU					|																		|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 										|
| Convolution 2: 5x5 	| 1x1 stride, VALID padding, outputs 10x10x64							|
| RELU					|																		|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 											|
| Fully connected 1		| input from o/p of conv 1 and conv 2 (flattened and appended) 7872x512	|
| RELU					|																		|
| Regularization	   	| Dropout (keep prob =0.5)  											|
| Fully connected 2		| 512x256     															|
| RELU					|																		|
| Regularization	   	| Dropout (keep prob =0.5)  											|
| Output Layer			| 256 inputs . 43 outputs 												|
| Softmax				| Apply softmax to output layer											|

 
The model architecture is detailed in the markup above cell 10 in the ipython notebook.


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the cells 11 to 15 of the ipython notebook. 

To train the model, I used an AdamOptimizer which uses momentum (moving average of parameters) method to perform a smoothed gradient descent. This removes some of the noise and oscillations that gradient descent has. [Practical Recommendations for Gradient-Based Training of Deep Architectures](  https://arxiv.org/pdf/1206.5533.pdf)

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixteenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.975
* test set accuracy of 0.9584

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I started with the base LeNet model that we used for the LeNet lab problem. With no preprocessing , the base model gave a validation accuracty of 89%.  

* What were some problems with the initial architecture?
The data wasn't pre processed.  A lot of the raw images had low lighting and without pre-processing the model was not able to learn from them

* How was the architecture adjusted and why was it adjusted? 
<p>I updated the model to use input data with normalization and YUV histogram equalization. That improved the accuracty to 92%. Then I updated the model to be a multi scale architecture by connecting both layer 1 and layer 2 convolutons to layer3 and also update the filter depth to higher values so the convolutions can learn more features . This increased the validation to 97%.</p>

* Which parameters were tuned? How were they adjusted and why?
<p>Adjusted the depth of the convolutions filters so the model can learn more features. To improve the validation accuracy to get closer to the training accuracy, set the dropout regularization parameters to 0.5 in the fully connected layers
</p>

If a well known architecture was chosen:
* What architecture was chosen?
  <p>Multi-scale  convolutional LeNet archictecture</p>
  
* Why did you believe it would be relevant to the traffic sign application?
  <p>Traffic signs have edges and shapes that can easily distinguish each other. A colvolutional neural net like LeNet works well to learn susch patterns in images. Also by connecting convolutions from both layers to the Fully connected layer, we allow  the model to learn from both convolution scales.</p>
  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
<p>We achieved a training accuracy of 100% and validation accuracy of 97.5%. This is the best score from all the possible parameter tuning, pre-processing (Grayscale was slightly lower in validation accuracy) that we could achieve. We could possibly improve the validation accuracy by 1 or 2 percentage points using augmented images (scale, position, rotation etc.). For now 97.5% validation accuracy is the best we could achieve without augmented data. The test score was 95.84 which shows that the model has learnt a fairly good deal on the input dataset
</p>

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The Yield Sign (2nd image) is taken along with a a lot of noise in the background. The 4th image is a spanish STOP sign. The color and the shape of the sign is similar to a German STOP sign. Wonder if the model can clasify that correctly. Image 5 is a squashed image of a No-Entry Sign and might be difficult to classify. Sign 6 is go straight or right turn. I wanted this image to see if the model can differentiate with teh right turn traffic sign which has more examples.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 17th and 18th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30 km./h 	| Speed Limit 30 km./h  						| 
| Yield     			| Yield 										|
| Stop					| Stop											|
| Stop  	      		| Stop      					 				|
| No Entry  			| Priority Road      							|
| Go Straight or Right 	| Go Straight or Right      					|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 95.8%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for computing softmax probabilities for the internet image is located in the 19th cell of the Ipython notebook. BElow is as screenshot with bar charts showing the top 5 softmax probabilites predicted by the model for each of the 6 images.

![alt text][image9] 
![alt text][image10]

most of the images were classified correctly and the model seems to be very sure (probability 1) for the predcitions. But for the no-entry it failed badly and classified as Prioirty Road 1 with Probability 1. In fact No Entry is the 5th bestprobab amon the top 5 soft max probabilities. This is probably because the model is not able to learn clearly when the image is squashed. Learnign with augmented images will likely resolve this problem