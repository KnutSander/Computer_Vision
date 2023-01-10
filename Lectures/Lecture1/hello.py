# hello.py -- a first OpenCV program, written during the first Vision lecture.

import cv2, sys, numpy

def mean (im):
    "Return the mean of an image."
    ny, nx, nc = im.shape
    print (ny, nx, nc)

    sum = 0
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                sum = sum + im[y,x,c]
    return sum / (ny * nx * nc)


# The following list of numbers is used to create a known image, one for which
# can work values out if necessary.
vals = [13, 12, 6, 4, 8, 6, 12, 4, 7, 9, 14, 8, 4, 12, 9, 4,
        14, 15, 12, 14, 4, 11, 14, 10, 7, 1, 6, 9, 14, 15, 7, 13,
        4, 12, 8, 0, 11, 6, 13, 10, 0, 7, 13, 3, 5, 13, 3, 9,
        3, 15, 12, 7, 1, 5, 8, 14, 1, 8, 11, 8, 13, 8, 15, 9,
        9, 7, 9, 6, 9, 4, 3, 2, 9, 10, 7, 1, 12, 14, 14, 13,
        14, 14, 8, 6, 11, 4, 12, 13, 14, 9, 15, 9, 7, 10, 15, 14,
        12, 1, 9, 7, 10, 13, 3, 11, 1, 3, 12, 5, 13, 1, 2, 11,
        0, 9, 14, 8, 10, 0, 10, 9, 9, 6, 5, 15, 0, 0, 15, 2,
        1, 3, 12, 14, 0, 6, 1, 4, 3, 10, 5, 2, 8, 0, 1, 15,
        3, 5, 11, 13, 14, 2, 10, 15, 0, 10, 13, 5, 4, 9, 7, 2,
        7, 6, 9, 8, 4, 5, 13, 4, 8, 0, 11, 5, 0, 4, 3, 15,
        5, 4, 5, 15, 10, 9, 11, 6, 6, 10, 0, 3, 5, 3, 10, 6]

# Convert the list of numbers into an image.  We'll use the same 8 unsigned
# bits per pixel that images read in by OpenCV normally use.
im = numpy.array (vals, dtype=numpy.uint8).reshape (12, 16, 1)

# Read in the image given on the command line.  Images read in by OpenCV are
# stored in numpy structures, normally with 8 unsigned bits per pixel.
im = cv2.imread (sys.argv[1])

# Calculate and output the mean.
ave = mean (im)
print (ave)

# Display the image.
cv2.imshow ("have a go", im)
cv2.waitKey (0)

