# Lab 2: Content-Based Image Retrieval


## 1 Introduction
Trying to compare images based on their histograms was attempted in the early 1990s, and is called content-based image retrieval (CBIR). In this lab we were given a 'benchmark' or 'strawman' program and our task was trying to improve it.

## 2 The Strawman CBIR Software
For the first test we run the provided colrec1 Python program. It calculates and compares histograms. We also use a provided test harness called *fact* to run the program and compare all the images using another provided program called colrec1-if. From this program we can create an output that we can check the accuracy of using a seperate function of *fact*.
The accuracy of colrec1 looks like this:

| actual expected | banana | chili | gapple | gfruit | orange | pear | rapple | tomato |
|-----------------|--------|-------|--------|--------|--------|------|--------|--------|
| banana          | 10     | 0     | 0      | 0      | 0      | 0    | 0      | 0      |
| chili           | 0      | 7     | 0      | 0      | 0      | 0    | 0      | 1      |
| gapple          | 0      | 0     | 6      | 0      | 0      | 1    | 4      | 0      |
| gfruit          | 0      | 0     | 0      | 10     | 2      | 0    | 0      | 0      |
| orange          | 0      | 0     | 0      | 0      | 8      | 0    | 0      | 0      |
| pear            | 0      | 0     | 0      | 0      | 0      | 9    | 0      | 0      |
| rapple          | 0      | 0     | 3      | 0      | 0      | 0    | 5      | 0      |
| tomato          | 0      | 3     | 1      | 0      | 0      | 0    | 1      | 9      |

It got a lot of them right, but it's far from perfect.
There are 10 images of each of the objects, so a perfect algorithm would get 10 in each row and column the same item intersects.

## 3 Refining the CBIR Software
In this step we tried improving on colrec1 by creating colrec3, which compares the histogram of each colour channel individually instead of as a combined histogram.
The colrec3 I implemented gave the following results:

| actual expected | banana | chili | gapple | gfruit | orange | pear | rapple | tomato |
|-----------------|--------|-------|--------|--------|--------|------|--------|--------|
| banana          | 10     | 0     | 0      | 0      | 0      | 0    | 0      | 0      |
| chili           | 0      | 7     | 0      | 0      | 0      | 0    | 0      | 1      |
| gapple          | 0      | 0     | 6      | 0      | 0      | 1    | 4      | 0      |
| gfruit          | 0      | 0     | 0      | 10     | 2      | 0    | 0      | 0      |
| orange          | 0      | 0     | 0      | 0      | 8      | 0    | 0      | 0      |
| pear            | 0      | 0     | 1      | 0      | 0      | 9    | 0      | 0      |
| rapple          | 0      | 0     | 3      | 0      | 0      | 0    | 5      | 0      |
| tomato          | 0      | 3     | 0      | 0      | 0      | 0    | 1      | 9      |

The result is basically the same, the only difference being that the green apple that was labeled a tomato is now labeled a pear, which is marginaly better.
Comparing the results using *fact* says that neither is better.

## 4 Segmenting Out the Object
In this step of the lab we try to eliminate the background from the pictures, as it is very similar in colour on all the pictures. We do this in colrec3s.
Using xv, we extract a range of HSV (Hue, Value, Saturation) values from the sample of the pictures and use these values to determine an upper and lower limit of the HSV value of pixels that should be considered background.
HSV values in OpenCV are a bit trikcy, as the normal way to represent these values are in ranges 0-360, 0-100 and 0-100; but OpenCV represents the values in the ranges 0-180, 0-255, 0-255. So to use the HSV values in OpenCV we must calculate these new values, which is easy enough. H is divided by 2, while S and V are multiplied by 2.55/2.56 (Not really sure which).
What colrec3s does it mostly similar to colrec3, but the pixels labeled as background are not counted. Using the preset bounds, I create a seperate image where all the background pixels are turned white. Then, when running through the channels of the original image, I skip the pixel if it's value in the other image is 255.
The ouput of the program results in this table:

| actual expected | banana | chili | gapple | gfruit | orange | pear | rapple | tomato |
|-----------------|--------|-------|--------|--------|--------|------|--------|--------|
| banana          | 10     | 0     | 0      | 0      | 0      | 0    | 0      | 0      |
| chili           | 0      | 10    | 0      | 0      | 0      | 1    | 0      | 0      |
| gapple          | 0      | 0     | 10     | 0      | 0      | 1    | 0      | 0      |
| gfruit          | 0      | 0     | 0      | 10     | 0      | 0    | 0      | 0      |
| orange          | 0      | 0     | 0      | 0      | 10     | 0    | 0      | 0      |
| pear            | 0      | 0     | 0      | 0      | 0      | 8    | 0      | 0      |
| rapple          | 0      | 0     | 0      | 0      | 0      | 0    | 10     | 0      |
| tomato          | 0      | 0     | 0      | 0      | 0      | 0    | 0      | 10     |

Now that's a lot better. Checking it agains the results of the other two yields these results:

colrec3s VS colrec3
Z-score  class    better
   0.00  banana   neither
   1.15  chili    colrec3s.res
   1.50  gapple   colrec3s.res
   0.00  gfruit   neither
   0.71  orange   colrec3s.res
   0.00  pear     neither
   1.79  rapple   colrec3s.res
   0.00  tomato   neither

colrec3s VS colrec1
Z-score  class    better
   0.00  banana   neither
   1.15  chili    colrec3s.res
   1.50  gapple   colrec3s.res
   0.00  gfruit   neither
   0.71  orange   colrec3s.res
   0.00  pear     neither
   1.79  rapple   colrec3s.res
   0.00  tomato   neither

As colrec3 and colrec1 provide virtually the same result, the comparison gives the same result as well. From looking at the differences between the tabels myself it looks like colrec1/colrec3 gets one more pear correct, while colrec3s gets one more tomato correct.
