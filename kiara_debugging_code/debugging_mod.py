
import numpy as np
from matplotlib import pyplot as plt
import imageio

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageSeries:

    def __init__(self, images, crop,fps=None):
        self.images = images #list of images (numpy arrays), saved itself as a numpy array
        self.L = len(self.images)
        self.x1, self.x2, self.y1, self.y2 = crop #(x1,y1) and (x2,y2) are cropping coords
        self.crimages = []
        for i in range(self.L):
            self.crimages.append(self.images[i][self.x1:self.x2, self.y1:self.y2])
        self.crimages = np.array(self.crimages)
        self.fps = fps

    def show_frame(self,t): #t = time to show 
        if 0 <= t <= self.L:
            plt.imshow(self.crimages[t],interpolation="none")
            plt.show()
        return

    def save_vid(self, filename, secspersec):
        imageio.mimwrite(filename, self.crimages, fps=self.fps*secspersec)

    def waterfall(self,x,y):
        if x == None:
            #plot row y=y over time
            return self.crimages[:,:,y]
        elif y == None:
            return np.transpose(self.crimages[:,x,:])

    def plot_waterfall(self,x,y,tint=None):
        waterfall = self.waterfall(x,y)
        if x == None:
            #plot col y over time
            fig = plt.figure()
            W = fig.add_subplot(111)
            W.imshow(waterfall,interpolation="none")
            if tint:
                plt.yticks(np.arange(0,self.L,self.fps*tint),np.arange(0,self.L/self.fps,tint))
            W.set_title("waterfall plot of row y = " + str(y))
            W.set_ylabel("time [s]")
            plt.show()
            return
        elif y == None:
            #plot col x=x over time
            fig = plt.figure()
            W = fig.add_subplot(111)
            W.imshow(waterfall,interpolation="none")
            if tint:
                plt.xticks(np.arange(0,self.L,self.fps*tint),np.arange(0,self.L/self.fps,tint))
            W.set_title("waterfall plot of col x = " + str(x))
            W.set_xlabel("time [s]")
            plt.show()
            return
        return

    def Ipixel(self,pixel):
        px,py = pixel
        return self.crimages[:,px,py]

    def plot_Ipixel(self,pixel,description=""):
        I = self.Ipixel(pixel)
        fig = plt.figure()
        f1 = fig.add_subplot(111)
        f1.set_title("I(t) for pixel" + str(pixel) + " (" + description + ")")
        f1.set_xlabel("time [s]")
        f1.plot(np.arange(0,self.L/self.fps,1/self.fps),I)
        plt.show()
        return

    def fftIpixel(self,pixel):
        I = self.Ipixel(pixel)
        IfreqA = np.fft.fft(I)/self.L
        Ifreq = np.fft.fftfreq(self.L,d=(1/self.fps))
        return(Ifreq, IfreqA)

    def plot_fftIpixel(self,pixel):
        Ifreq, IfreqA = self.fftIpixel(pixel)
        fig = plt.figure()
        f1 = fig.add_subplot(111)
        f1.set_title("FFT for pixel" + str(pixel))
        f1.set_xlabel("frequency [1/s]")
        f1.plot(Ifreq, abs(IfreqA))
        plt.show()
        return    

    def g2(self,pixel):
        I = self.Ipixel(pixel)
        g2 = []
        avgsq = np.mean(I)**2
        for tau in range(len(I)-1):
            if tau == 0:
                dotp = 0
                for t in range(len(I)):
                    dotp += I[t]*I[t]
                g2.append(dotp/(len(I)*avgsq) )
            elif tau != 0:
                dotp = 0
                for t in range(len(I)-tau):
                    dotp += I[t]*I[t+tau]
                g2.append( dotp / (len(I[:-tau])*avgsq))
        g2 = np.array(g2)
        return g2



    
