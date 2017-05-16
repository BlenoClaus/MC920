# -*- coding: UTF-8 -*-
# Bleno Humberto Claus
# 145444
# Installing scikit-video: sudo pip install sk-video
# Doc: http://www.scikit-video.org/stable/index.html
import skvideo.io
import skvideo.utils
import skvideo.datasets
import numpy as np
import math
import matplotlib.pyplot as plt

########################################################
#### 						models					####
########################################################
class VideoGray(object):

	def __init__(self, nameFile):
		self.nameFile = nameFile
		self.data = skvideo.io.vread(nameFile, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
		self.nrFrames = self.data.shape[0]
		self.dimensionX = self.data.shape[1]
		self.dimensionY = self.data.shape[2]
		print "video "+nameFile+" successfully read!"
		print "#####################################"
		

class Histogram(object):

	def __init__(self, videoGray):
		self.videoGray = videoGray
		self.H = np.zeros((self.videoGray.nrFrames, 256))
		for f in range(self.videoGray.nrFrames):
			for i in range(self.videoGray.dimensionX):
				for j in range(self.videoGray.dimensionY):
					self.H[f][self.videoGray.data[f][i][j]] += 1
			print "Calculated histogram ["+str(f)+"]"
		print "Histogram calculation completed!"
		print "#####################################"


########################################################
#### 					Controllers					####
########################################################
class VideoGrayController(object):
	
	def __init__(self, videoGray):
		self.videoGray = videoGray

	def diffByPixes(self, T1, T2, writerController):
		n = self.videoGray.nrFrames -1
		diff = np.zeros(n)	
		outputVideo = np.zeros( (2*(n+1),self.videoGray.dimensionX,self.videoGray.dimensionY) )
		j = 0
		for i in range(0,n):
			y = np.array(self.videoGray.data[i+1] - self.videoGray.data[i])
			diff[i] = np.count_nonzero(y > T1)
			if diff[i] > T2:
				outputVideo[j] = self.videoGray.data[i]
				outputVideo[j+1] = self.videoGray.data[i+1]
				j += 2
		print "Diff by Pixes Completed!"
		print "#####################################"
		writerController.plotGraphic("Diff by Pixes", diff)
		return outputVideo

	def diffByBlock8(self, T1, T2, writerController):
		return self.__diffByBlock(8, T1, T2, writerController)

	def diffByBlock16(self, T1, T2, writerController):
		return self.__diffByBlock(16, T1, T2, writerController)

	def __diffByBlock(self, numBlock, T1, T2, writerController):  
		blocksX = self.videoGray.dimensionX/numBlock
		blocksY = self.videoGray.dimensionX/numBlock
		n = self.videoGray.nrFrames -1
		diff = np.zeros(n)
		outputVideo = np.zeros( (2*(n+1),self.videoGray.dimensionX,self.videoGray.dimensionY) )
		j = 0
		for f in range(0,n):
			for lin in range(numBlock):
				for col in range(numBlock):
					block1 = self.videoGray.data[f][lin*blocksX:(lin+1)*blocksX, col*blocksY:(col+1)*blocksY]
					block2 = self.videoGray.data[f+1][lin*blocksX:(lin+1)*blocksX, col*blocksY:(col+1)*blocksY]
					if np.sum((block2-block1)**2) > T1:
						diff[f] +=  1
			if diff[f] > T2:
				outputVideo[j] = self.videoGray.data[f]
				outputVideo[j+1] = self.videoGray.data[f+1]
				j += 2
		print "Diff by Block["+str(numBlock)+"] Completed!"
		print "#####################################"
		writerController.plotGraphic("Diff by Block["+str(numBlock)+"]", diff)
		return outputVideo 

	def __sobelFilter9x9(self,f,i,j):
		Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
		Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
		newPixel = 0
		for lin in range(1,4):
			for col in range(1,4):
				newPixel += Gx[lin-1][col-1]*self.videoGray.data[f][i-lin][j-col]
		if newPixel < 0:
			newPixel = 0
		return newPixel

	def __computNrBords(self,T1):
		nPixesBordas = np.zeros(self.videoGray.nrFrames)
		for f in range (self.videoGray.nrFrames):
			for i in range(1,self.videoGray.dimensionX-1):
				for j in range(1,self.videoGray.dimensionY-1):
					newPixel = self.__sobelFilter9x9(f,i,j)
					if newPixel > T1:
						nPixesBordas[f] += 1
			print "Sobel filter to frame["+str(f)+"] = "+str(newPixel)
		print "Number of edge pixels counted"
		return nPixesBordas

	def diffByBorder(self, T, writerController):
		n = self.videoGray.nrFrames
		x = self.videoGray.dimensionX
		y = self.videoGray.dimensionY
		diff = np.zeros(n-1)
		nPixesBordas = self.__computNrBords(T1 = 10)
		outputVideo = np.zeros((2*n, x, y))
		j = 0
		for f in range (1, n):
			diff[f-1] = math.fabs(nPixesBordas[f]-nPixesBordas[f-1])
			if diff[f-1] > T:
				outputVideo[j] = self.videoGray.data[f-1]
				outputVideo[j+1] = self.videoGray.data[f]
		print "Diff by Border Completed!"
		print "#####################################"
		writerController.plotGraphic("Diff by Border", diff)
		return outputVideo


class HistogramController(object):
	
	def __init__(self, histogram):
		self.histogram = histogram
		n = self.histogram.videoGray.nrFrames
		self.diff = np.zeros(n)
		for f in range(n):
			for index in range(0,255):
				self.diff[f] += math.fabs(self.histogram.H[f][index] - self.histogram.H[f][index+1])
		#limiar T = u + a
		self.u = self.diff.mean()
		self.a = self.diff.std()
		self.T = self.u + self.a
		print "Calculated Histogram Threshold | ["+str(self.T)+"]T = ["+str(self.u)+"]u + ["+str(self.a)+"]a  "
		print "#####################################"
		
	
	def diffByHistogram(self, writerController):
		n = self.histogram.videoGray.nrFrames
		x = self.histogram.videoGray.dimensionX
		y = self.histogram.videoGray.dimensionY
		outputVideo = np.zeros( (2*n, x, y) )
		j = 0
		for f in range(n-1):
			if self.diff[f] > self.T :
				outputVideo[j] = self.histogram.videoGray.data[f]
				outputVideo[j+1] = self.histogram.videoGray.data[f+1]
				j += 2
		print "Diff by Histogram Completed!"
		print "#####################################"
		writerController.plotGraphic("Diff by Histogram", self.diff)
		return outputVideo


class WriterController(object):

	def plotGraphic(self, title, values):
		plt.title(title, fontsize=30)
		plt.xlabel('frames', fontsize=30)
		plt.ylabel('value', fontsize=30)
		plt.bar(range(len(values)), values, width=0.8, color="blue", align="center")
		plt.show()

	def generateFile(self, fileName, frames):
		skvideo.io.vwrite(fileName,frames)
		print "Video "+fileName+" record successfully!"
		print "#####################################"

		
########################################################
#### 						main					####
########################################################

## inputs
videoName          = "lisa"
videoFormat        = "mpg"
limiarT1ByPixes    = 100
limiarT2ByPixes    = 14000
limiarT1ByPBlock8  = 2000
limiarT2ByPBlock8  = 40
limiarT1ByPBlock16 = 1500
limiarT2ByPBlock16 = 130
limiarTByBorder    = 200

## instances
video = VideoGray( videoName + "." + videoFormat)
histogram = Histogram(video)
controllerVideo = VideoGrayController(video)
controllerHistogram = HistogramController(histogram)
writer = WriterController()

## work
outVideoByPixes = controllerVideo.diffByPixes(T1 = limiarT1ByPixes, T2 = limiarT2ByPixes, writerController = writer)
writer.generateFile(videoName +"OutByPixesl."+ videoFormat, outVideoByPixes)

outVideoByBlock8 = controllerVideo.diffByBlock8(T1 = limiarT1ByPBlock8, T2 = limiarT2ByPBlock8, writerController = writer)
writer.generateFile(videoName +"OutByBlock8."+ videoFormat, outVideoByBlock8)

outVideoByBlock16 = controllerVideo.diffByBlock16(T1 = limiarT1ByPBlock16, T2 = limiarT2ByPBlock16, writerController = writer)
writer.generateFile(videoName +"OutByBlock16."+ videoFormat, outVideoByBlock16)

outVideoByHistogram = controllerHistogram.diffByHistogram(writerController = writer)
writer.generateFile(videoName +"OutByHistogram."+ videoFormat, outVideoByHistogram)

outVideoByBorder = controllerVideo.diffByBorder(T = limiarTByBorder, writerController = writer)
writer.generateFile(videoName +"OutByBorder."+ videoFormat, outVideoByBorder)

