# Lab 3: Thresholding and Contours

## 1 Introduction
In this lab we learn about thresholding and contours. The aim is to isolate each of the dices on the given image and count the total sum of the dots on them.

## 2 The Program
The program *contours* is provided, and I've made extensive notes in a coopy of it that says what all the various functions do and what their parameters mean. So no need to write in length about it here, just check *contours_remake*.

## 3 Thresholding
What thresholding does is changing a picture based on it's values. We mainly use it to create binary images that we can then extract contours from. See *contours_remake* for a comprehensive explenation of the different ways you can threshold an image. There is also several pictures in this folder that shows the result of different thresholding on the dice image.

## 4 Processing Contours
Contours are outlines of shapes in an image. For more info see *contours_remake* and the lab script.

## 5 Counting the Dots
To count the dots we simply find internal contours and counting them. Again, see *contours_remake* for a detailed explenation.

## 6 Concluding Remarks
Thresholding and contouring is an interesting aspect of computer vision. From a bit of testing, it seems the best way to threshold most images is using Otsu's method.
