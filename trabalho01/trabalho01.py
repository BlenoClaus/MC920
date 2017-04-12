# -*- coding: UTF-8 -*-
# name: Bleno Humberto Claus
# ra: 145444
from scipy import misc
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class Image(object):
	"""classe que representa uma Imagem forncendo algumas operacoes 
		para seu processamento"""
	def __init__(self, name):
		self.name = name
		self.img = misc.imread(self.name)
		self.x = self.img.shape[0]
		self.y = self.img.shape[1]
		self.pix = self.x * self.y
		self.red = { "1"  : np.zeros((256)), 
					 "4"  : np.zeros((4)),
					 "32" : np.zeros((32)),
					 "128": np.zeros((128)),
					 "256": np.zeros((256))
				   } 
		self.gree = {"1"  : np.zeros((256)), 
					 "4"  : np.zeros((4)),
					 "32" : np.zeros((32)),
					 "128": np.zeros((128)),
					 "256": np.zeros((256))
					} 
		self.blue = {"1"  : np.zeros((256)), 
					 "4"  : np.zeros((4)),
					 "32" : np.zeros((32)),
					 "128": np.zeros((128)),
					 "256": np.zeros((256))
					}

	def normalize(self, color):
		return color/(self.x*self.y)

	def euclidianDistance(self, key, otherImage):
			newRed = self.normalize(self.red[key])
			otherRed = otherImage.normalize(otherImage.red[key])
			dinstRed = self.euclidian(newRed,otherRed)

			newBlue = self.normalize(self.blue[key])
			otherBlue = otherImage.normalize(otherImage.blue[key])
			dinstBlue = self.euclidian(newBlue,otherBlue)

			newGreen = self.normalize(self.gree[key])
			otherGreen = otherImage.normalize(otherImage.gree[key])
			dinstGreen = self.euclidian(newGreen,otherGreen)

			return (dinstRed+dinstBlue+dinstGreen)/3

	def euclidian(self, values1, values2):
		dist = 0; 
		for i in range(len(values1)):
			dist = dist + (math.fabs(values1[i]- values2[i]))**2
		return math.sqrt(dist)

	def histogramInitial(self):
		for i in range(self.x):
			for j in range(self.y):
				self.red["1"][self.img[i][j][0]] += 1;
				self.gree["1"][self.img[i][j][1]] += 1;
				self.blue["1"][self.img[i][j][2]] += 1;


	def cumputeHistogram(self):
		self.histogramInitial()
		keysSet = self.red.keys();
		for i in range(len(keysSet))[1:len(keysSet)]:
			b = 256/int(keysSet[i])
			for j in range (int(keysSet[i])):
				for z in range (b):
					self.red[keysSet[i]][j]  +=  self.red[keysSet[0]][b*j + z]
					self.gree[keysSet[i]][j] +=  self.gree[keysSet[0]][b*j + z]
					self.blue[keysSet[i]][j] +=  self.red[keysSet[0]][b*j + z]

	def plotRed(self, numBin):
		self.plot(self.red[numBin],"red")

	def plotGree(self, numBin):
		self.plot(self.gree[numBin],"green")

	def plotBlue(self, numBin):
		self.plot(self.blue[numBin],"blue")

	def plot(self, values, colorName):
		plt.title("Imagem: [" +self.name+ "] - "+colorName , fontsize=30)
		plt.xlabel('qtde:', fontsize=30)
		plt.ylabel('pixel:', fontsize=30)
		plt.bar(range(len(values)), values, width=0.8, color=colorName, align="center")
		plt.show()

class ImagemGray(object):
	"""imagem em tons de cinza apenas"""
	def __init__(self, imgage):
		self.image = cv2.cvtColor(imgage.img, cv2.COLOR_BGR2GRAY)
		self.x = self.image.shape[0]
		self.y = self.image.shape[1]

class BinPlan(object):
	"""Plano de bits"""
	def __init__(self, number, plan, x, y):
		self.number = number
		self.plan = plan
		self.x = x
		self.y = y

	def entropia(self):
		allPixes = float(self.y * self.x)
		one = np.count_nonzero(self.plan)
		zero = allPixes - one
		probOne = one/allPixes
		probZero = zero/allPixes
		return -(probZero*np.log2(probZero) + probOne*np.log2(probOne))

	def plot(self):
		plt.imshow(255*self.plan, cmap='gray')
		plt.show()
	
class BitPlanFactory(object):
	"""fabrica de planos de bits"""
	def __init__(self, imageGray):
		self.imageGray = imageGray

	def getBitPlan(self,number):
		binPlan = np.zeros(shape = (self.imageGray.x,self.imageGray.y))
		index = 8 - number
		for i in range(self.imageGray.x):
			for j in range(self.imageGray.y):
				binPlan[i][j] = bin( int(self.imageGray.image[i][j]))[2:].zfill(8)[index]

		return BinPlan(number,binPlan, self.imageGray.x, self.imageGray.y)		

####################################
##          	main   	   		  ##
####################################

baboom = Image("baboon.png")
peppers = Image("peppers.png") 
watch = Image("watch.png")
baboom.cumputeHistogram()
peppers.cumputeHistogram()

facttoryWatch  = BitPlanFactory(ImagemGray(watch))
facttoryPeppers = BitPlanFactory(ImagemGray(peppers))

print "Computed images: ["+peppers.name+", "+baboom.name+"]"
bins = ["4","32","128","256"]
for k in range(len(bins)):
	baboom.plotRed(bins[k])
	peppers.plotRed(bins[k])
	baboom.plotGree(bins[k])
	peppers.plotGree(bins[k])
	baboom.plotBlue(bins[k])	
	peppers.plotBlue(bins[k])
	print "["+bins[k]+"] Average euclidean distance: "+str(baboom.euclidianDistance(bins[k], peppers))

for i in range(1,9):
	binPlanWatch = facttoryWatch.getBitPlan(i)
	binPlanWatch.plot()
	print "Entropia by image: ["+watch.name+"] plan("+str(i)+") = "+ str(binPlanWatch.entropia())
	binPlanPeppers = facttoryPeppers.getBitPlan(i)
	binPlanPeppers.plot()
	print "Entropia by image: ["+peppers.name+"] plan("+str(i)+") = "+ str(binPlanPeppers.entropia())	
