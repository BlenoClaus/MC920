# -*- coding: UTF-8 -*-
# Nome: Bleno Humberto Claus
# Ra: 145444
import numpy as np
from scipy import misc


class Image(object):

	def __init__(self, name, formatImg):
		self.name = name +"."+formatImg
		self.formatImg = formatImg
		self.data = misc.imread(self.name)
		self.x, self.y, self.z = self.data.shape[0], self.data.shape[1], self.data.shape[2]

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
		return U[:, 0:k], np.diag(S)[0:k, 0:k], V[0:k, :] 

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


######################################################
####	main
######################################################

imageName = "peppers"
formatImg = "png"
k = 50

newImagem = Image(imageName, formatImg)
newCompressImg = CompressImage(newImagem,k)
newCompressImg.save()
print newCompressImg.getRMSE()
print newCompressImg.getCompressionRatio()
newCompressImg.save()


