# Lab 5 Notes

## 1 Introduction
In this lab we are approached by a fictitious company to help them create a stereo system for capturing faces in 3D. To aid in this they are using a modified version of the Candide face model, which they have "imaged" from a model of their stereo rig using the POV-ray-tracer.
Their stereo rig has identical cameras arranged so that their optical axes are precisely parallel, so we can use the equation from the lecture notes *Z = fB/D*.
Where Z is the distance to the object, f is the focal length, B is the baseline and D is the disparity (parallax) of a feature between the left and the right images.

## 2 Estimating the Focal Length
The virtual cameras are placed at (*x,y,z*) locations (+/- 75, 300, 800), which means that B = 150mm.
Vertex 5, the tip of Candides nose, is at location (0,289,280). This means the distance from the camera to it, Z, is approximately 800 - 280 = 520mm.

The next step was to measure the location of the tip of the nose on the images from the left and right camera. I did thi using xv, and I believe my accuracy to be +/- 5 pixels.
Both pictures have a width of 639 pixels and and height of 479 pixels.
The measurments I took were:

### Right Picture
* x: 251 
* y: 251
Both being +/- 5 pixels.
Transforming it to conform with the 3D system yields the following:
* x: 251-(639/2) = -68.5
* y: (479/2)-251 = -11.5

### Left Picture
* x: 389
* y: 251
Both being +/- 5 pixels.
Transforming it to conform with the 3D system yields the following:
* x: 389-(639/2) = 69.5
* y: (479/2)-251 = -11.5

Unsure how to figure out the focal length at this point as there is no mention of how to calulate it in the lecture notes.

## 3 Camera Calibration to Determine the Focal Length
We are given a script to calibrate the camera in *calibrate.py*. This outputs the focal length and uncertainty of the camera, which are:
Focal length: 484.66 
Uncertainty: +/- 1.3

## 4 Distance Calculation from Calibration Data
The distance to the tip of candides nose using the focal length is calulated using the following formula:
Z = fB / Xl - Xr
(484.66 * 150) / (69-68) = 530.6

To calulate the the uncertainty of Z we need to do some more math. Just look at Chapter 9 of the lecture notes if needed.
