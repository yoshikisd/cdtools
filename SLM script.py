# python script for Controlling the Thorlabs EXULUS-HD2 Spatial Light Modulator
# K Kesinbora 01.19.21 
# 
# https://github.com/wavefrontshaping/slmPy
#
# One has to install the wxPython first as slmPy depends on that module pip install -U wxPython
# Download the slmpy.py to the same folder as the script.
#

from IPython import get_ipython
get_ipython().magic('reset -sf')

# Import all the dependencies

import wx #wxPython module https://wxpython.org/pages/downloads/index.html
import slmpy #slmPy module from https://github.com/wavefrontshaping/slmPy Thanks to wavefrontshaping

import numpy as np
import time

import urllib
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Define an SLM object 

x = 0 # Monitor ID 0 for primary screen, 1 for monitor (default). x = 0 is basically testing in your own computer screen.

slm = slmpy.SLMdisplay(monitor = x) 

# Retreive the size of the SLM

resX, resY = slm.getSize()      

# Generate a meshgrid support

X,Y = np.meshgrid(np.linspace(0,resX,resX),np.linspace(0,resY,resY))

SizeX,SizeY = (1920,1200) # This sets the size of the image to be pulled from the picsum library

sleeptime = 5 # How long the image is displayed on the SLM screen
    
# Display a static test image on the SLM screen    

# Generate a test image

#test = np.round((2**8-1)*(0.5+0.5*np.sin(2*np.pi*X/50))).astype('uint8')
    
#slm.updateArray(test)

#time.sleep(sleeptime)

#slm.close()

# Display a static image from an url

#url = 'https://picsum.photos/'+str(SizeX)+'/'+str(SizeY)

#response = requests.get(url)
#img = np.asarray(Image.open(BytesIO(response.content)).convert('L'))

#imgplot = plt.imshow(img)

#slm.updateArray(img)

#time.sleep(sleeptime)

#slm.close()

# Display a series of random images from an url

maximageno = 10                      # Total number of images to be displayed on the SLM
counter = 1

while(counter < maximageno):
    url = 'https://picsum.photos/'+str(SizeX)+'/'+str(SizeY)
    response = requests.get(url)
    img = np.asarray(Image.open(BytesIO(response.content)).convert('L'))  # Convert to image into an 8bit numpy array

    slm.updateArray(img)
    time.sleep(sleeptime)
    slm.close
    
    # fig = plt.figure()
    # imgplot = plt.imshow(img)  # I couln't get this to work simultaneously with the slm update array
    counter = counter + 1


