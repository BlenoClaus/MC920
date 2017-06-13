# -*- coding: UTF-8 -*-
# Nome: Bleno Humberto Claus
# Ra: 145444
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


class Image(object):

	def __init__(self, name, formatImg):
		self.name = name +"."+formatImg
		self.formatImg = formatImg
		self.data = misc.imread(self.name)
		self.x, self.y, self.z = self.data.shape[0], self.data.shape[1], self.data.shape[2]

	#doc: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
	def getUSVbyChannel(self, channel):
		return np.linalg.svd(self.data[:, :, channel])

	def getNrBytes(self):
		num = 0
		for i in range(self.z):
			U, S, V = self.getUSVbyChannel(i)
			num += U.nbytes + S.nbytes + V.nbytes
		return num

class CompressImage(object):

	def __init__(self, img, k):
		self.img = img
		self.k = k
		self.compressImage = self.getCompressedImage()

	def getUSVbyChannel(self, channel):
		U,S,V = self.img.getUSVbyChannel(channel)
		return U[:, 0:self.k], np.diag(S)[0:self.k, 0:self.k], V[0:self.k, :] 

	def getNrBytes(self):
		num = 0
		for i in range(self.img.z):
			Ug,Sg,Vg = self.getUSVbyChannel(i)
			num += (Ug.nbytes + Sg.nbytes + Vg.nbytes)
		return num

	def getCompressedImage(self):
		compressImage = np.zeros((self.img.x, self.img.y, self.img.z))
		for i in range(self.img.z):
			Ug,Sg,Vg = self.getUSVbyChannel(i)
			compressImage[:,:,i] = np.dot(np.dot(Ug,Sg), Vg)
		return compressImage

	def getRMSE(self):
		return np.sqrt(np.mean((self.img.data-self.compressImage)**2))

	def getCompressionRatio(self):
		return np.float(self.getNrBytes())/np.float(self.img.getNrBytes())

	def save(self):
		misc.imsave(self.img.name+"_compress_"+str(self.k)+"."+self.img.formatImg, self.compressImage)


class Graphic(object):

	def plot(self, title,xText, yText, k, values):
		plt.title(title, fontsize=30)
		plt.xlabel(xText, fontsize=30)
		plt.ylabel(yText, fontsize=30)
		plt.bar(k, values, width=1, color="blue", align="center")
		plt.show()


######################################################
####	main
######################################################


## inputs data
imageName = "peppers"
formatImg = "png"
k = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350]

## computing
P = []
RMSE = []

for i in range(len(k)):
	print "Values to k = "+str(k[i])+" :"
	newImagem = Image(imageName, formatImg)
	newCompressImg = CompressImage(newImagem,k[i])
	newCompressImg.save()
	rmse = newCompressImg.getRMSE()
	RMSE.append(rmse)
	print "	RMSE = "+str(rmse)
	p = newCompressImg.getCompressionRatio()
	P.append(p)
	print "	Compression Rate = " + str(p)
	newCompressImg.save()

graphic = Graphic()
graphic.plot("Compression Rate x K", "k", "Compression Rate", k, P)
graphic.plot("RMSE x K", "k","RMSE", k, RMSE)
graphic.plot("RMSE x Compression Rate", "RMSE", "Compression Rate", RMSE, P)
