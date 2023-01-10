#!/usr/bin/env python3
"Calibrate stereo cameras using POV images."
import sys, os, math, cv2
import numpy as np

# We are looking for 8 horizontal and 7 vertical lines.
# Creates a numpy array filled with zeros, made up of 8*7=56 tuples with
# 3 values each [0, 0, 0]
objpoints = np.zeros ((8*7,3), np.float32)
# Creates a multi-dimensional "meshgrid" with values 0-7 in the first position
# 0-6 in the second position and just 0 in the third position ex. [6 1 0]
# Don't ask what all the other stuff does, jesus christ
objpoints[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

# Termination criterion.
# Defines a termination criteria for the given algorithm
# Basically tells it to run a maximum of 30 timesy
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0)

# Arrays for real-world object points and the corresponding image locations.
points3d = []
points2d = []

# Loop over the calibration images.
for frame in range (0, 30):
    im = cv2.imread ("calib-%3.3d.png" % frame)
    grey = cv2.cvtColor (im, cv2.COLOR_BGR2GRAY)

    # Look for the corners inside the board.  Record any that we find for use
    # in calibrating the camera.
    found, corners = cv2.findChessboardCorners (grey, (8,7), None, 0)
    if found:
        points3d.append (objpoints)
        cv2.cornerSubPix (grey, corners, (11,11), (-1,-1), term_crit)
        points2d.append (corners)
        
        # Mark and display the corners.
        cv2.drawChessboardCorners (im, (8,7), corners, found)
        cv2.imshow ("Corners found", im)
        cv2.waitKey (500)

# Calibrate the camera using the points and image locations.
found, cam, D, R, T = cv2.calibrateCamera (points3d, points2d,
                         grey.shape[::-1], None, None)
            
# Take the mean of the x and y focal lengths as the overall one, and their
# difference as a (poor) estimate of the error.
fx = cam[0,0]
fy = cam[1,1]
F = (cam[0,0] + cam[1,1]) / 2
Ferr =  max (abs (F-fx), abs(F-fy))
print ("Focal length:", F, "+/-", Ferr)

