# Contour Detection and Superformula Fit

The code provides detection of contours using the **OpenCV** library and computes the best fit by **<a href = https://en.wikipedia.org/wiki/Superformula> Gielis' superformula<a/>**.

Fitting is done by the **minimize** method included in **scipy.optimize** using the **COBYLA** algorithm.

Code can be tested with the included sample image by adding the suffix  **--image leaf.JPG** in the command line.

The picture "leaf.JPG" provided was taken by the author of this code.
