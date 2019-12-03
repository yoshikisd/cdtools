
import numpy as np
from matplotlib import pyplot as plt
import imageio
from PIL import Image, ImageSequence
from debugging_mod import ImageSeries


vid = Image.open('data_10_4/kiara_20fps_redlaser_vid.tif')

vidarray = []
for i, page in enumerate(ImageSequence.Iterator(vid)):
    pg = np.array(page)
    vidarray.append(pg)
vidarray = np.array(vidarray)

RedLaserExp = ImageSeries(vidarray, (430,606,590,766),fps=20) #cropping  x1:x2, y1:y2

#with np.load('data_9_28/kiara_data_300sec_green') as data:
#    GreenLaserExp = ImageSeries(data['arr_0'], (500,676,580,756), fps=5)
    #crop to: x1=500, x2=676, y1=580, y2=756

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
#Note: human eye can see at c. 150 fps

#inner top right corner of disk (120,70)
#inner top left corner of disk (66,63)
#inner bottom left corner of disk (66,107)
#inner bottom (87,118)
#inner right (125, 86)

#pixel intensity plot: ---------------------------------------
#RedLaserExp.plot_Ipixel((125,86),description="inner right")
#RedLaserExp.plot_Ipixel((87,118),description="inner bottom")
#RedLaserExp.plot_Ipixel((66,63),description="inner top left")

#pixel fft plot: --------------------------------------------
#RedLaserExp.plot_fftIpixel((125,86))
#RedLaserExp.plot_fftIpixel((87,118))
#RedLaserExp.plot_fftIpixel((66,63))

#waterfall plot for row: -------------------------------------
#RedLaserExp.plot_waterfall(None,118,tint=1)

#waterfall plot for col: -------------------------------------
#RedLaserExp.plot_waterfall(66,None,tint=1)

#save a video: -----------------------------------------------
#RedLaserExp.save_vid("redlaser_10-4_20fps_1x.mp4",1)
