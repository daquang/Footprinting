import pybedtools
from collections import defaultdict
from numpy import *
from savitzky_golay import *
#This program generates functions so that a 5' end signal wig file (a) and a regions wig file (b) can be passed to genereate
#a dictionary that holds a vector and score for each region in b

def makevecdict(a,b):#a is signal, b is regions
	c = a.intersect(b,wb=True)#temporary file that holds only signal in the regions
	l=lambda:defaultdict(l)
	d = l()#the dictionary that holds each region's vector and score
	for line in b:
		start = int(line[1])
		end = int(line[2])
		chro = line[0]
		score = double(line[6])
		length = end - start
		d[chro][start][end][1] = zeros(length)
		d[chro][start][end][2] = score
	for line in c:
		rstart = int(line[5])
		rend = int(line[6])
		rchro = line[4]
		start = int(line[1]) - rstart
		end = int(line[2]) - rstart
		val = int(line[3])
		d[rchro][rstart][rend][1][start:end] = val
	return d

def filtvec(v,winsize=1001,prime=1):#returns a filtered vector appropriate for Boyle footprinting. It assumes the vector has been appropriately padded by 500 on both sides
	binary = v != 0
	win = ones(winsize)
	sum500 = convolve(v,win,mode='same')#sum of all elements in a 1kb window at each element
	#avg500 = convolve(v,win,mode='same')#average of all elements in a window at each element
	#avg500[avg500 == 0] = 0.0001#this ensures no zero division
	nonzero500 = convolve(binary,win,mode='same')#number of all non-zero elements in a 1kb window at each element
	avg500 = sum500/nonzero500#average of all non-zero elements in a 1kb window at each element
	vn = v/avg500#normalized vector
	vn[isnan(vn)] = 0
	return savitzky_golay(vn, 9, 2, deriv=prime, rate=1)

def extendinterval(feature, padding=0):#extends the interval of each feature by padding
	feature.start = feature.start - padding
	feature.stop = feature.stop + padding
	return feature
