# Lab 6 Notes

## 1 Introduction
In this lab we explore face detection using the Viola-Jones algorithm.

## 2 Viola-Jones
The program *face-detect* is run by typing the command *./face-detect* followed by nothing if you want it to use the webcam or by an image file if you want it to run on the image.
A problem when running it with the picture *ese-small.jpg* is that not all of the faces are outlined in blue like they are supposed to be. The common characteristics are that they are either too dark/light, are partially hidden by other people or are too close to other faces.
In the detect method, the function *detectMultiScale* has two parameters that can be tweaked, scaleFactor and minNeighbors.
ScaleFactor has to be larger than 1, and from testing keeping it as small as possible provided the best results. If it increased by too much, the faces would start to be outlined twice.
minNeighbors also performed better when smaller, recogising more faces when decreased.
The final question is how I would work out which Haar cascade is the most effective, and I have no idea how I would do that.

## 3 Finding Eyes
The program uses Haar cascade to find eyes, but it doesn't find them consistently. It was hard to create a reliable way of doing this, but this is what I implemented:
* First I check if the eyes is on the top half of the face. This removes any mouth corners that were previously detected as eyes.
* The first eye found is stored and we look for a second one.
* After finding another eye, we look to see that they are at roughly the same height.
* If they are, draw the two eyes and the loop is over.
* If they aren't, draw the eye as a faulty eye and keep looking for the second eye of the pair.
Problems with this is if only one eye is found, no eyes are marked. Theres also the possibility of the first eye being a faulty eye.
From testing it is semi-reliable.
