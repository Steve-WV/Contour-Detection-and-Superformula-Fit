import cv2
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import tkinter
from tkinter import *
import argparse
import time
import scipy.stats as stats
from scipy.optimize import minimize 
	
#Define superformula function (including possible rotation theta)
def spf(x,m1,n1,n2,n3,a,b,theta):
	return ((abs((np.cos(0.25*m1*(x+theta)))/a))**n2 + (abs(np.sin(0.25*m1*(x+theta)))/b)**n3)**(-1/n1)
		
# Show a gray image
def show_value(val):
    plt.close()
    _, binary = cv2.threshold(gray, int(val), int(val), cv2.THRESH_BINARY_INV)
    plt.figure()
    plt.imshow(binary, cmap="gray")
    global greys
    greys=int(val)
    plt.draw()
    plt.pause(0.001)
    		
# read the image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#possible denoising
image = cv2.fastNlMeansDenoisingColored(image,None,60,60,7,21)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
root = tkinter.Tk()
scale = tkinter.Scale(orient='horizontal',resolution=10,  from_=20, to=270, command=show_value)
scale.set(70)
scale.pack(padx=50, pady=50)
lbl = Label(root, text = "Choose right grayscale")
lbl.pack()
exit_button = Button(root, text="Ok", command=root.destroy)
exit_button.pack(pady=20)
root.mainloop()
plt.close()

# create a binary thresholded image
_, binary = cv2.threshold(gray, greys, greys, cv2.THRESH_BINARY_INV)
ret,thresh = cv2.threshold(binary,10,255,0)
#ret,thresh = cv2.threshold(binary,127,255,0)

# calculate moments and determine centroid
M = cv2.moments(thresh)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# find the contours from the thresholded image and show the image with the drawn contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
font = cv2.FONT_HERSHEY_COMPLEX
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
plt.imshow(image)
plt.xlabel('Contours allright -- then close window', color='r')
plt.show()
plt.close()

#findig number of points in contour
l=0
for i in range(0, len(contours)):
   for j in range(0, len(contours[i])):
    l+=1

# Determining data points from contour list and compute radii and angles
pts1 = np.zeros(l)
pts2 = np.zeros(l)
r=np.zeros(l)
theta1=np.zeros(l)
x_test=np.zeros(l)
y_test=np.zeros(l)

for j in range(0, l):
	pts1[j]+=float(contours[0][j][0][0])
	pts2[j]+=float(contours[0][j][0][1])
	r1_new=(pts1[j]-cX)
	r2_new=(pts2[j]-cY)
	r[j]+=ma.sqrt(r1_new**2+r2_new**2)
	theta1[j]+=np.arctan2(r1_new,r2_new)
	x_test[j]+=r[j]*ma.cos(theta1[j])
	y_test[j]+=r[j]*ma.sin(theta1[j])
 
b_r=np.max(r)	
fx=(1/b_r)*r

def print_value(val):
    global num_arms
    num_arms=float(val)

root = tkinter.Tk()
scale = tkinter.Scale(orient='horizontal', from_=1, to=10, command=print_value)
scale.pack()
scale.set(2)
lbl = Label(root, text = "Choose number of arms m1")
lbl.pack()
exit_button = Button(root, text="Ok", command=root.destroy)
exit_button.pack(pady=50)
root.mainloop()
plt.close()

#Define sum of squares with superformula function (including possible rotation theta)
def spf_res(x):
	res=0.0
	for i in range(0,l):
		res+= (((abs((np.cos(0.25*x[0]*(theta1[i]+x[6])))/x[4]))**x[2] + (abs(np.sin(0.25*x[0]*(theta1[i]+x[6])))/x[5])**x[3])**(-1/x[1])-fx[i])**2
	return res

tic = time.perf_counter()
theta_initial_guess = np.array([num_arms,2.0,2.0,2.0,2.0,1.0,0.0])
result = minimize(spf_res,theta_initial_guess, method='COBYLA')
toc = time.perf_counter()
print(f'Computation took {toc - tic:0.4f} seconds')
para_spf=result['x']
print(f'Parameters in superformula: \n {para_spf}')

# Computing values for deduced parameters m1,n1,n2,n3,a,b,theta
x1=np.zeros(l)
y1=np.zeros(l)
r_test=np.zeros(l)
for i in range(0,l):
	x1[i]+=spf(theta1[i],para_spf[0],para_spf[1],para_spf[2],para_spf[3],para_spf[4],para_spf[5],para_spf[6])*ma.cos(theta1[i])
	y1[i]+=spf(theta1[i],para_spf[0],para_spf[1],para_spf[2],para_spf[3],para_spf[4],para_spf[5],para_spf[6])*ma.sin(theta1[i])
	r_test[i]+=spf(theta1[i],para_spf[0],para_spf[1],para_spf[2],para_spf[3],para_spf[4],para_spf[5],para_spf[6])
	
#Chisquare test and plotting with superformula
print(f'Results of a Chisquare test: \n {stats.chisquare(fx,r_test)}')
plt.plot(np.hstack((x1,x1[0])),np.hstack((y1,y1[0])),linewidth=2.5, label='Superformula Fit')
plt.plot((1/b_r)*np.hstack((x_test,x_test[0])),(1/b_r)*np.hstack((y_test,y_test[0])),linewidth=2.5, label='Measured Contour')
plt.legend(fontsize=12)
plt.show()
