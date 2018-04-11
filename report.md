# **Traffic Sign Classifier Project**

The goals of this project are the following:
* Build a model to classify traffic signs images;
* Training the model in German Traffic Sign Database and evalute its performance;
* Reflect about the pros and cons of using Deep Learning strategy in Autonomous Car solutions.

### The German Traffic Sign Dataset

Udacity has provided a curated subset of [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) that can be found in [curated Dataset](http://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This zip file contains 3 separated files:
1. train.p a python pick file with 34799 images and labels to be used as train dataset;
2. valid.p a python pick file with 4410 images and labels to be used as validation dataset;
3. test.p a python pick file with 12630 images and labels to be used as test dataset;
4. All images are 32 pixels x 32 pixels size with 3 RGB channels;
5. For each dataset there is a numpy array with labels. A label is a integer between 0 and 33 and indicates sign class. All classes description can be found in [signames.cvs](signames.cvs);

### Exploring
Let's visualize the pipeline showing the result for each step. Consider the original image:

![](./report_images/original.png)

In **step 1** we get two images, one selecting the yellow channel:

![](./report_images/yellowchannel.png)

Another image with white channel:

![](./report_images/whitechannel.png)

In **step 2**, we join yellow and white images:

![](./report_images/joinwhiteandyellowchannel.png)

In **step 3**, we convert to grayscale:

![](./report_images/gray.png)

Now, in **step 4**, we apply Canny filter to detect edges:

![](./report_images/gray.png)

In **step 5**, we isolate only the area of interest:

![](./report_images/onlyregionofinterest.png)

Following, **steps 6 through 10**, we do:
- execute Hough algorithm to detect lines in image;
- select lines with *correct* slope (absolute value between 0.5 and 10.0) [*draw_lines function*];
- using slope direction, classify them in two groups: right lines and left lines [*draw_lines function*];
- find the *mean* line for each group [*draw_lines function*];
- extrapolate these lines to cross the horizontal line at the bottom (x axis) and horizontal line in the middle [*draw_lines function*];

Finally, we obtain:

![](./report_images/houghandrawlines.png)

Now, we can join original image and lane lines image:

![](./report_images/orignialpluslinesdetect.png)


### Shortcomings

The main drawback of this method is it detects only straight lines, but real lanes could be curved too. The movie *challenge.mp4* shows a car driving in a highway turning right, it is ease to see that our pipeline fails to detect the *correct* lane. Another shortcoming is not to work well when there is not contrast between lane lines and the pavement.

### Possible Improvements

I believe that the most effective and cheap improvement would be calculate the lanes getting the current image but considering previous frames too. The reason is that lanes do not change abrupt in short period of time.
Other obvious improvement is detect curved lanes (there is an Hough procedure to detect curved lines).
