# Lab 4 Notes

## 1 Introduction
In this lab the goal is to figure out whether pictures of biscuits are broken or not.
This is done using the program *bikky*, which does the following things as a baseline:
1. Reads the image as a grey-scale image
2. Thresholds using a fixed value
3. Then it is tidied up using morphological operations
4. Contours are found around each foreground object
5. Each of the contours are processed
6. And lastly some text is written on the image
It is incomplete in the sense that it labels every image as a biscuit. The goal is to make the program label the images with either a circle, a rectangle or as broken.

## 2 Improve the Thresholding
From all experiments we've done this semester, as well as the assignment, otsu thresholding seems to work the best. So I changed the fixed value thresholding of the original script with otsus method.

## 3 What Kind of Biscuit is it?
The next step was to try to figure out what kind of biscuit is in the picture given. To figure this out, I did the following:
* To check if a bisquit was square, I found the rotated minimum area rectangle of the contour and compared its area with the area of the contour. If the differences were small enough, it was likely a square.
* To check if the biscuit is round I did the same thing, only using a minimum enclosing circle.
* Lastly, if there were more than 1 contour or if neither of the two previous checks were true, the biscuit was labeled as broken.
This worked reasonably well, although looking through the code now some improvements are definetly possible.
