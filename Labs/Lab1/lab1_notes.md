# Lab 1: Getting to Grips with OpenCV

## 1 Introduction
Basic Linux functions to create a directory and unzippping a file.
mkdir <dir_name> - Create a directory
cd <dir_name> - open the given file
unzip <file_name> - unzips the given file

## 2 Displaying Images
We are using an image display program known as xv to look at images and get useful information from them. It has many useful features that can be used to extract information from the picture.
xv <file_name> - opens the given file in xv

## 3 Summarizing Images
In this step we use the provided Python program called *summarize* that takes a picture file as an argument, then displays the minimum, maximum, average and mean values of it's grey levels. Then finaly computes the histogram of the picture and outputs it to a .dat file.

## 4 Plotting
To plot things in this module, we will be using Gnuplot, general-purpose graph-plotting package available for pretty much every computing platform.
To start it, simply type *gnuplot* in a terminal window.

To me it looks like under-exposed images have the majority of their values towards the lower end of the range, well-exposed images have the majority of their values around the middle of the range and over-exposed images have the majority of their values toward the higher end of the range.

### Functions
unset key - unsets the key (duh) / Seems every plot has a key, and this unsets it and makes it not display when the values are plotted

set grid - sets the plot type to grid (other types exist)
set title "<title_name>" -  sets the title name
set xlabel "<label_name>" - sets the x-label name, normally pixel value 
set ylabel "f<label_name>" - sets the y-label name, normally value frequency
set xrange/yrange [<min_value>:<max_value>] - sets the min and max values of the given axis
set term pdf - sets pdf files as the defalt output
set output "<file_name>" - output the plot to the given file

plot "<file_name>" - attempts to create a plot with the values in the given file
	+ with boxes - plots using boxes, instead of the normal plus-signs (other types exis)
replot - needs to be called to show changes in parameters

## 5 Improving the Histogram Routine
When images only have a few grey level values occupied, the better way of visualising it is to contrast stretch the values based on the min and max values.
Changing the code slightly improves the output and makes the under-exposed image values more readable and less lumped together.

## 6 Per-Channel Colour Histogram
The per-channel based algorithm works almost the same way as the normal histogram routine. The only difference is that a channel number is passed (either 0 for blue, 1 for green or 2 for red) and we only run through the values of that colour channel.

