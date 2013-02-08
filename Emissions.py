#This contains a class Emission.It encapsulates the "e" matrix, but for the continuous case. It can update for the 
#M step and use several 
#The parameters argument is a list of tuples containing parameters for the pdf. It assumes that the number of tuples
#is equal to the proper number of states.

from numpy import array, sqrt
import numpy.random
from scipy.stats import *

class Emission:
	def __init__(self,parameters=[(0,1)],distribution='normal'):
		samplefs = {'normal':numpy.random.normal, 'poisson':numpy.random.poisson, 'negative_binomial':numpy.random.negative_binomial}#for random sampling
		pdfs = {'normal':norm.pdf, 'poisson':poisson.pmf, 'negative_binomial':nbinom.pmf}
		self.samplef = samplefs[distribution]
		self.pdf = pdfs[distribution]
		self.p = parameters
		self.distribution = distribution

	def single_emit(self,state=0):#this is mostly for the continuous HMM generation
		return self.samplef(*self.p[state])

	def pdf_matrix(self,seq):#returns a matrix of pdf values for each observation in seq
		return array([self.pdf(seq,*q)  for q in self.p])
	
	def updated_emission(self,mu, sigmasq):#returns an updated emission of the same type according to ML guesses
		updaters = {'normal':self.normal_update, 'poisson':self.poisson_update, 'negative_binomial':self.nbinom_update}
		upd = updaters[self.distribution]
		return upd(mu,sigmasq)
	
	def normal_update(self,mu,sigmasq):#note the sqrt is included, because numpy functions use stdev, not variance
		return Emission(parameters=array(zip(mu,numpy.sqrt(sigmasq))),distribution=self.distribution)

	def poisson_update(self,mu,sigmasq):
		return Emission(parameters=mu,distribution=self.distribution)

	def nbinom_update(self,mu,sigmasq):
		p = 1 - mu/sigmasq
		r = mu*(1-p)/p
		return Emission(parameters=zip(r,p),distribution=self.distribution)
			
