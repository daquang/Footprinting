#!/usr/bin/env python
from numpy import *
from pylab import *
from DNase_functions3 import *
from scipy.stats.mstats import *

"""
A normalization function for comparing hotspots. Signals are mapped to a reference
distribution so that all hotspots, regardless of amplitude, can be compared.
Input:
v - the signal to be transformed
r - a reference distribution that v will be mapped onto
Output:
t - a transformation of v that preserves the rank of its values but mapped to r
"""
def quantilemap(v,r):
	ranks = rankdata(v)/len(v)
	t = mquantiles(r,prob=ranks)
	return t

"""

"""
def superposeplot(vs,i,t="some plot"):
	figure(i)
	title(t)
	for va in vs:
		plot(va)
		
"""
Input:
i - integer. number of figure
v - the Hotspot signal

Output:
a pylab plot of a bunch of comparisons
"""
def comparefilters(v, i):
	figure(i)
	subplot(231)
	#plot(savitzky_golay(v, 9, 2, deriv=1, rate=1))
	#title('Savitzky-Golay w=9 o=2')
	plot(convolve(v,array([-1,-2,0,2,1])/-8.0,mode='same'))
	title('n=2 noise-robust smoothing')
	subplot(232)
	#plot(savitzky_golay(v, 7, 2, deriv=1, rate=1))
	#title('Savitzky-Golay w=7 o=2')
	plot(convolve(v,array([5,-12,-39,0,39,12,-5])/-96.0,mode='same'))
	title('n=4 noise-robust smoothing')
	subplot(233)
	plot(savitzky_golay(v, 5, 2, deriv=1, rate=1))
	title('Savitzky-Golay w=5 o=2')
	subplot(234)
	plot(savitzky_golay(v, 9, 3, deriv=1, rate=1))
	title('Savitzky-Golay w=9 o=3')
	subplot(235)
	plot(savitzky_golay(v, 7, 3, deriv=1, rate=1))
	title('Savitzky-Golay w=7 o=3')
	subplot(236)
	plot(savitzky_golay(v, 5, 3, deriv=1, rate=1))
	title('Savitzky-Golay w=5 o=3')

def filtvec(v,winsize=1001,prime=1):#returns a filtered vector appropriate for Boyle footprinting. It assumes the vector has been appropriately padded by 500 on both sides
	binary = v != 0#vector, same size as v, that tells whether a nonzero value is there or not
	win = ones(winsize)
	sum500 = convolve(v,win,mode='same')#sum of all elements in a 1kb window at each element
	#avg500 = convolve(v,win,mode='same')#average of all elements in a window at each element
	#avg500[avg500 == 0] = 0.0001#this ensures no zero division
	nonzero500 = convolve(binary,win,mode='same')#number of all non-zero elements in a 1kb window at each element
	avg500 = sum500/nonzero500#average of all non-zero elements in a 1kb window at each element
	vn = v/avg500#normalized vector
	vn[isnan(vn)] = 0
	#return savitzky_golay(vn, 9, 2, deriv=prime, rate=1)
	return vn

def filtvec2(v,winsize=1001,prime=1):#returns a filtered vector appropriate for Boyle footprinting. It assumes the vector has been appropriately padded by 500 on both sides
	binary = v != 0#vector, same size as v, that tells whether a nonzero value is there or not
	win = ones(winsize)
	sum500 = convolve(v,win,mode='same')#sum of all elements in a 1kb window at each element
	#avg500 = convolve(v,win,mode='same')#average of all elements in a window at each element
	#avg500[avg500 == 0] = 0.0001#this ensures no zero division
	nonzero500 = convolve(binary,win,mode='same')#number of all non-zero elements in a 1kb window at each element
	avg500 = sum500/winsize#average of all non-zero elements in a 1kb window at each element
	vn = v/avg500#normalized vector
	vn[isnan(vn)] = 0
	#return savitzky_golay(vn, 9, 2, deriv=prime, rate=1)
	return vn


def comparenormals(v, i):
	figure(i)
	subplot(311)
	plot(filtvec(v,2001))
	title('Divide by non-zero average 2000')
	subplot(312)
	plot(filtvec(v,1001))
	title('Divide by non-zero average 1000')
	subplot(313)
	plot(filtvec(v,129))
	title('Divide by non-zero average 128')
	
def comparenormals2(v, i):
	figure(i)
	subplot(311)
	plot(filtvec2(v,2001))
	title('Divide by average 2000')
	subplot(312)
	plot(filtvec2(v,1001))
	title('Divide by average 1000')
	subplot(313)
	plot(filtvec2(v,129))
	title('Divide by average 128')
	
	
def plot9(vs,i,t="some figure"):
	figure(i)
	suptitle(t)
	for x in xrange(9):
		subplot(330 + x + 1)
		plot(vs[x])

"""
Plots and compares the quantile norm method as well as the resulting filtered signal.
Input:
vs - a list of 9 signals to be quantile normalized
v - a reference distribution to be quantile mapped to. usually the MTPN promoter
Output:
plots the unnormalized, normalized, and filtered signals for comparison
"""
def comparequantilenorms(vs,v):
	plot9(vs,1,"9 random hotspots")
	superposeplot(vs,2,"superposed plots")
	vs2 = [quantilemap(a,v) for a in vs]#generate the quantile normalized plots
	plot9(vs2,3,"quantile normalized plots")
	superposeplot(vs2,4,'superposed quantile normalized plots')
	vs3 = [savitzky_golay(a, 5, 2, deriv=1, rate=1) for a in vs]
	vs4 = [savitzky_golay(a, 5, 2, deriv=1, rate=1) for a in vs2]
	plot9(vs3,5,"non-normalized filtered plots")
	superposeplot(vs3,6,"superposed non-normalized filtered plots")
	plot9(vs4,7,"quantile normalized filtered plots")
	superposeplot(vs4,8,'superposed quantile normalized filtered plots')
	
"""
Plots and compares the quantile norm method on the filtered signal (reverse order of above).
The non-normalized plots are not shown since it is assumed comparequantilenorms is ran first.
Input:
vs - a list of 9 signals to be filtered and then quantile normalized
v - a reference distribution to be quantile mapped to. usually the MTPN promoter
Output:
plots the unnormalized, normalized, and filtered signals for comparison
"""
def comparequantilefilts(vs,v):
	f = savitzky_golay(v, 5, 2, deriv=1, rate=1)
	vsf = [savitzky_golay(a, 5, 2, deriv=1, rate=1) for a in vs]
	#quantile map by magnitude, and then recover sign
	vsf2 = [quantilemap(abs(a),abs(f))*sign(a) for a in vsf] 
	plot9(vsf2,9,"9 filtered to quantile normalized plots")
	superposeplot(vsf2,10,"superposed filtered to quantile normalized plots")
	
seqs = load('sortedvectors.npy')	
v = load('MTPN_promoter.npy')

#v=seqs[0]
from random import sample
vs = seqs[range(9)]

comparequantilenorms(vs,v)
comparequantilefilts(vs,v)

""" These lines compare the ordering of filtering and normalizings
plot9(vs,1,"9 random hotspots")
vs2 = [savitzky_golay(a, 5, 2, deriv=1, rate=1) for a in vs]
vs3 = [a/max(a) for a in vs2]
plot9(vs2,2, 'Filtered/non-normalized hotspots')
plot9(vs3,3, 'Filtered/normalized hotspots')
vs4 = [a/max(a) for a in vs]
plot9(vs4, 'Normalized/non-filtered hotspots')
vs5 = [savitzky_golay(a, 5, 2, deriv=1, rate=1) for a in vs4]
plot9(vs5, 'Normalized/filtered hotspots')
"""

"""
figure(1)
plot(v)
comparefilters(v,2)
plot9(vs,3)
comparenormals(v,4)
comparenormals2(v,5)
"""
show()
if __name__ == '__main__':
	print "hello"