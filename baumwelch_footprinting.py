#!/usr/bin/env python
from DNase_functions import *
from normfilts import *
from numpy import *
from Emissions import *
from random import *
import itertools
import functools
from multiprocessing import Pool, cpu_count
from datetime import datetime
import warnings
from exceptions import RuntimeWarning

def hmm_generate(n,a,e):
	seq = zeros(n)
	states = zeros(n,dtype=int)
	ashape = a.shape
	numstates = ashape[0]
	#set first state, each state has equal chance
	states[0] = randint(0,numstates - 1)
	seq[0] = e.single_emit(states[0])
	for i in xrange(n-1):
		nextstate = weighted_choice(a[states[i]])
		nextseq = e.single_emit(nextstate)
		seq[i+1] = nextseq
		states[i+1] = nextstate
	return seq, states

def weighted_choice(weights):#returns weighted random sample
	totals = []
	running_total = 0
	for w in weights:
		running_total += w
		totals.append(running_total)
	rnd = random() * running_total
	for i, total in enumerate(totals):
		if rnd < total:
			return i

def hmm_estimate(seq, states):#returns maximum likelihood a and e for known states
	statesset = sort(list(set(states)))
	binarystates = (statesset[:,newaxis] == states) * 1.0
	#print states
	#print seq
	#print binarystates
	A = dot(binarystates[:,:-1],binarystates[:,1:].transpose())
	mu = dot(binarystates,seq[:,newaxis])[:,0]/binarystates.sum(axis=1)
	print mu
	sigmasq = (((seq - mu[:,newaxis])**2)*binarystates).sum(axis=1)/binarystates.sum(axis=1)
	print (seq - mu[:,newaxis])
	print sigmasq
	#emissets = sort(list(set(seq)))
	#binaryemissions = (seq[:,newaxis] == emissets) * 1.0
	#print binaryemissions
	#E =  dot(binarystates, binaryemissions)
	a = dot(diag(1/A.sum(axis=1)),A)
	e = Emission(parameters=zip(mu,sigmasq))
	#print A
	#print E
	return a, e

def test_estimate():
	p = 0.98
	q = 0.999
	a = array([
	[0.180*p, 0.274*p, 0.426*p, 0.120*p, (1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4],
	[0.171*p,0.368*p,0.274*p,0.188*p,(1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4],
	[0.161*p,0.339*p,0.375*p,0.125*p,(1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4],
	[0.079*p,0.355*p,0.384*p,0.182*p,(1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4],
	[(1-q)/4,(1-q)/4,(1-q)/4,(1-q)/4,0.300*q,0.205*q,0.285*q,0.210*q],
	[(1-q)/4,(1-q)/4,(1-q)/4,(1-q)/4,0.322*q,0.298*q,0.078*q,0.302*q],
	[(1-q)/4,(1-q)/4,(1-q)/4,(1-q)/4,0.248*q,0.246*q,0.298*q,0.208*q],
	[(1-q)/4,(1-q)/4,(1-q)/4,(1-q)/4,0.177*q,0.239*q,0.292*q,0.292*q]])
	e = array([
	[1,0,0,0],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,0,1],
	[1,0,0,0],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,0,1]])
	(seq, states) = hmm_generate(100000,a,e)
	(a2,e2) = hmm_estimate(seq,states)

def hmm_train(seqs, a_guess, e_guess):#start here and train_step 1/10/12
	tol = 1e-6
	maxiter = 500
	(new_logP, a_updated, e_updated) = train_step(seqs, a_guess, e_guess)
	for k in xrange(maxiter):
		print "step " + str(k) + "\n"
		print a_updated
		print e_updated.p
		old_logP = new_logP
		(new_logP, a_updated, e_updated) = train_step(seqs, a_updated, e_updated)
		if abs((old_logP - new_logP)/old_logP) < tol:#second conditional break
			break
	print 'Converged in ' + str(k) + ' steps'
	print a_updated
	print e_updated
	return a_updated, e_updated

def hmm_train2(seqs, a_guess, e_guess):#start here and train_step 1/10/12
	tol = 1e-6
	maxiter = 500
	a_updated = a_guess
	(new_logP, a_updated2, e_updated) = train_step(seqs, a_guess, e_guess)
	e_updated.p[0][0] = 0
	e_updated.p[3][0] = 0
	e_updated.p[4][0] = 0
	e_updated.p[0][1] = 0.05
	e_updated.p[3][1] = 0.05
	e_updated.p[4][1] = 0.05
	for k in xrange(maxiter):
		print "step " + str(k) + "\n"
		print a_updated
		print e_updated.p
		old_logP = new_logP
		(new_logP, a_updated2, e_updated) = train_step(seqs, a_updated, e_updated)
		#do some biasing
		e_updated.p[0][0] = 0
		e_updated.p[3][0] = 0
		e_updated.p[4][0] = 0
		e_updated.p[0][1] = 0.05
		e_updated.p[3][1] = 0.05
		e_updated.p[4][1] = 0.05
		if abs((old_logP - new_logP)/old_logP) < tol:#second conditional break
			break
	print 'Converged in ' + str(k) + ' steps'
	print a_updated
	print e_updated
	return a_updated, e_updated

"""
The mapping function used in train_step_parallel. Each of the outputs are meant to be summed with the respective
outputs from other mapped objects.
Input:
seq - a training sequence
a_guess - guess for the transition state matrix
e_guess - guess for the Emissions parameters
Output:
post - the posterior decoded probabilities matrix
logPi - this sequence's contribution to the overall logP score
e_denominatori - this sequence's contribution to the overall denominator (for calculating ML mean and variance)
mean_numeratori - this sequence's contribution to the overall numerator of the mean (for calculating ML mean)
Ai - this sequence's contribution to the transition state matrix
"""
def train_parallel_map(seq, a2, e2p):
	a_guess = a2
	e_guess = Emission(e2p)
	#As of 2/10/13, posterior_decode is used here. This is meant for the
	#general case of randomly generated HMM sequences.
	#For training on DNase-Seq data, posterior_step, which assumes
	#certain start and end states, should be used instead
	#(s, f_tilde, b_tilde, pdf_matrix) = posterior_decode(seq, a_guess, e_guess)
	
	#As of 2/26/13, posterior_step has been implemented again for DNase-Seq
	(s, f_tilde, b_tilde, pdf_matrix) = posterior_step(seq, a_guess, e_guess)
	post = s*f_tilde*b_tilde
	es = pdf_matrix[:,1:].transpose()
	L = seq.size
	logPi = log(s).sum()
	#Ai = a_guess*dot(f_tilde[:,0:L-1],es*b_tilde[:,1:].transpose())
	#cannot return Ai because dot product is already parallelized
	e_denominatori = post.sum(axis=1)
	mean_numeratori = (seq*post).sum(axis=1)
	#the last three are necessary for a single process calculation of Ai
	return post, mean_numeratori, e_denominatori, logPi, f_tilde, b_tilde, es

"""
"""
def variance_parallel_map(x, m):
	return (((x[0] - m[:,newaxis])**2)*x[1][0]).sum(axis=1)
	

"""
A parallelized version of the train_step function
Input:
seqs - the training sequences
a_guess - guess for the transition state matrix
e_guess - guess for the Emissions parameters
pool - the multiprocessing pooler
"""#start here 2/8/13
def train_step_parallel(seqs,a_guess,e_guess,pool):#one step of iteration, 
#returns log likelihood and updated guesses. accepts sequences and guesses. e_guess is an Emission class
	logP = 0
	A = zeros(a_guess.shape)
	mean_numerator = zeros(a_guess.shape[0])#each row holds the numerator for the mean for each state
	var_numerator = zeros(a_guess.shape[0])#each row holds the numerator for the variance for each state
	e_denominator = zeros(a_guess.shape[0])#each element is the denominator for each state
	"""
	posts = list()
	for seq in seqs:
		(s, f_tilde, b_tilde, pdf_matrix) = posterior_step(seq, a_guess, e_guess)
		post = s*f_tilde*b_tilde
		posts.append(post)
		es = pdf_matrix[:,1:].transpose()
		L = seq.size
		logP = logP + log(s).sum()
		A = A + a_guess*dot(f_tilde[:,0:L-1],es*b_tilde[:,1:].transpose())
		e_denominator = e_denominator + post.sum(axis=1)
		mean_numi = (seq*post).sum(axis=1)
		mean_numerator = mean_numerator + mean_numi
	"""
	infos = pool.map(functools.partial(train_parallel_map,a2=a_guess,e2p=e_guess.p),seqs)
	#this step is for finding all As
	z = [(a_guess*dot(info[4][:,:-1],info[6]*info[5][:,1:].transpose()),info[1],info[2],info[3]) for info in infos]
	(A,mean_numerator,e_denominator,logP) = sum(z,axis=0)
	"""
	for info in infos:#whoops
		A += info[1]
		mean_numerator += info[2]
		e_denominator += info[3]
		logP += info[4]
	"""
	mu = mean_numerator/e_denominator
	variances = pool.map(functools.partial(variance_parallel_map,m=mu), itertools.izip(seqs,infos))
	"""
	for seq,post in itertools.izip(seqs,posts):#this loop establishes the variance, since mu needed to be calculated first
		var_numerator = var_numerator + (((seq - mu[:,newaxis])**2)*post).sum(axis=1)#wrong post lol.forgot to multiply by post here
	"""
	for var_numeratori in variances:#this loop establishes the variance, since mu needed to be calculated first
		var_numerator += var_numeratori
	sigmasq = var_numerator/e_denominator
	a_updated = dot(diag(1/A.sum(axis=1)),A)
	e_updated = e_guess.updated_emission(mu,sigmasq)
	return logP, a_updated, e_updated

"""
A parallelized version of the version 1 of hmm_train (that is, no fixing).
Input:
seqs - the training sequences
a_guess - guess for the transition state matrix
e_guess - guess for the Emissions parameters
pool - the multiprocessing pooler
progress - a file handle for writing results
"""
def hmm_train_parallel(seqs, a_guess, e_guess, pool, progress):#start here and train_step 1/10/12
	print "Initial guess"
	print a_guess
	print e_guess.p
	progress.write("Initial guess:\n")
	progress.write("Transition matrix:\n" + str(a_guess) + "\n") 
	progress.write("Emission parameters:\n" + str(e_guess.p) + "\n\n") 
	tol = 1e-9
	maxiter = 500
	(new_logP, a_updated, e_updated) = train_step_parallel(seqs, a_guess, e_guess,pool)
	for k in xrange(maxiter):
		print "step " + str(k+1)
		print a_updated
		print e_updated.p
		print new_logP
		progress.write("step " + str(k+1) + ":\n")
		progress.write("Transition matrix:\n" + str(a_updated) + "\n") 
		progress.write("Emission parameters:\n" + str(e_updated.p) + "\n") 
		progress.write("Log likelihood:\n" + str(new_logP) + "\n\n")
		old_logP = new_logP
		(new_logP, a_updated, e_updated) = train_step_parallel(seqs, a_updated, e_updated,pool)
		if abs((old_logP - new_logP)/new_logP) < tol:#second conditional break
			break
	print 'Converged in ' + str(k+1) + ' steps'
	print a_updated
	print e_updated.p
	print new_logP
	progress.write('Converged in ' + str(k+1) + ' steps' + '\n')
	progress.write("Transition matrix:\n" + str(a_updated) + "\n") 
	progress.write("Emission parameters:\n" + str(e_updated.p) + "\n") 
	progress.write("Log likelihood:\n" + str(new_logP) + "\n\n")
	return a_updated, e_updated

"""
A parallelized version of the version 2 of hmm_train (that is, with fixing).
At every iteration step, the transition matrix and the standard deviations of 
the FP and HS states are reset to the specified defaults.
Input:
seqs - the training sequences
a_guess - guess for the transition state matrix
e_guess - guess for the Emissions parameters
pool - the multiprocessing pooler
progress - a function handle for writing results
HS_std - the default standard deviation of the HS states

Output:
a_updated - the converged transition matrix
e_updated - the converged emissions parameters
new_logP - the log likelihood of the best guess

These outputs are for saving the best guess since the training will be run several times.
3/1/13: Removed all print statements, since everything gets printed to the Progress file anyways 
"""
def hmm_train_parallel2(seqs, a_guess, e_guess, pool, progress, HS_std):#start here and train_step 1/10/12
	progress.write("Initial guess:\n")
	progress.write("Transition matrix:\n" + str(a_guess) + "\n") 
	progress.write("Emission parameters:\n" + str(e_guess.p) + "\n\n") 
	tol = 1e-9
	maxiter = 500
	a_updated = a_guess
	(new_logP, a_updated2, e_updated) = train_step_parallel(seqs, a_guess, e_guess,pool)
	e_updated.p[0][0] = 0
	e_updated.p[3][0] = 0
	e_updated.p[4][0] = 0
	e_updated.p[0][1] = HS_std
	e_updated.p[3][1] = 1.1*HS_std
	e_updated.p[4][1] = HS_std
	for k in xrange(maxiter):
		progress.write("step " + str(k+1) + ":\n")
		progress.write("Transition matrix:\n" + str(a_updated) + "\n") 
		progress.write("Emission parameters:\n" + str(e_updated.p) + "\n") 
		progress.write("Log likelihood:\n" + str(new_logP) + "\n\n")
		old_logP = new_logP
		(new_logP, a_updated2, e_updated) = train_step_parallel(seqs, a_updated, e_updated,pool)
		e_updated.p[0][0] = 0
		e_updated.p[3][0] = 0
		e_updated.p[4][0] = 0
		e_updated.p[0][1] = HS_std
		e_updated.p[3][1] = 1.1*HS_std
		e_updated.p[4][1] = HS_std
		if abs((old_logP - new_logP)/new_logP) < tol:#second conditional break
			break
	progress.write('Converged in ' + str(k+1) + ' steps' + '\n')
	progress.write("Transition matrix:\n" + str(a_updated) + "\n") 
	progress.write("Emission parameters:\n" + str(e_updated.p) + "\n") 
	progress.write("Log likelihood:\n" + str(new_logP) + "\n\n")
	return a_updated, e_updated, new_logP


def train_step(seqs,a_guess,e_guess):#one step of iteration, 
#returns log likelihood and updated guesses. accepts sequences and guesses. e_guess is an Emission class
	logP = 0
	A = zeros(a_guess.shape)
	mean_numerator = zeros(a_guess.shape[0])#each row holds the numerator for the mean for each state
	var_numerator = zeros(a_guess.shape[0])#each row holds the numerator for the variance for each state
	e_denominator = zeros(a_guess.shape[0])#each element is the denominator for each state
	posts = list()
	for seq in seqs:
		(s, f_tilde, b_tilde, pdf_matrix) = posterior_step(seq, a_guess, e_guess)
		post = s*f_tilde*b_tilde
		posts.append(post)
		es = pdf_matrix[:,1:].transpose()
		L = seq.size
		logP = logP + log(s).sum()
		A = A + a_guess*dot(f_tilde[:,0:L-1],es*b_tilde[:,1:].transpose())
		e_denominator = e_denominator + post.sum(axis=1)
		mean_numi = (seq*post).sum(axis=1)
		mean_numerator = mean_numerator + mean_numi
	mu = mean_numerator/e_denominator
	for seq,post in itertools.izip(seqs,posts):#this loop establishes the variance, since mu needed to be calculated first
		var_numerator = var_numerator + (((seq - mu[:,newaxis])**2)*post).sum(axis=1)#wrong post lol.forgot to multiply by post here
	sigmasq = var_numerator/e_denominator
	a_updated = dot(diag(1/A.sum(axis=1)),A)
	e_updated = e_guess.updated_emission(mu,sigmasq)
	return logP, a_updated, e_updated

def posterior_step(seq, a, e):#accepts a sequence, the current guesses, 
#and returns s, f_tilde, and b_tilde. b_tilde is not transposed
	L = len(seq)
	numstates = a.shape[0]
	aT = a.transpose()
	pdf_matrix = e.pdf_matrix(seq)#compute all emission probabilities at each index for each state
	#forward algorithm
	f_tilde = zeros([numstates,L])
	s = zeros(L)
	#initialisation
	f_tilde[0,0] = 1.0*pdf_matrix[0,0]#only state 0 can begin the sequence
	s[0] = f_tilde[:,0].sum()
	f_tilde[:,0] = f_tilde[:,0]/s[0]
	#recursion
	for i in xrange(L-1):
		y = (f_tilde[:,i] * aT).sum(axis=1)
		s[i+1] = dot(pdf_matrix[:,i+1],y)
		f_tilde[:,i+1] = 1/s[i+1]*pdf_matrix[:,i+1]*y
	#backwards algorithm
	b_tilde = zeros([numstates,L])
	#initilisation
	b_tilde[-1,-1] = 1/(s[-1]*f_tilde[-1,-1]);#only the last state can end the sequence. denominator ensures 1 sums
	for i in xrange(L-2,-1,-1):
		b_tilde[:,i] = 1/s[i]*(pdf_matrix[:,i+1]*b_tilde[:,i+1]*a).sum(axis=1)	 	
	#print b
	#termination
	#post = s*f*b
	#print post.sum(axis=0)
	#P = s[0,1:].prod()
	#print P	
	return s, f_tilde, b_tilde, pdf_matrix#return the pdf matrix to save extra computation

"""
Unlike posterior_step, this function assumes both states have an equal chance
of beginning and ending.
"""
def posterior_decode(seq, a, e):#accepts a sequence, the current guesses, 
#and returns s, f_tilde, and b_tilde. b_tilde is not transposed
	L = len(seq)
	numstates = a.shape[0]
	aT = a.transpose()
	pdf_matrix = e.pdf_matrix(seq)#compute all emission probabilities at each index for each state
	#forward algorithm
	f_tilde = zeros([numstates,L])
	s = zeros(L)
	#initialisation
	f_tilde[:,0] = 1.0/numstates*pdf_matrix[:,0]#only state 0 can begin the sequence
	s[0] = f_tilde[:,0].sum()
	f_tilde[:,0] = f_tilde[:,0]/s[0]
	#recursion
	for i in xrange(L-1):
		y = (f_tilde[:,i] * aT).sum(axis=1)
		s[i+1] = dot(pdf_matrix[:,i+1],y)
		f_tilde[:,i+1] = 1/s[i+1]*pdf_matrix[:,i+1]*y
	#backwards algorithm
	b_tilde = zeros([numstates,L])
	#initilisation
	#b_tilde[-1,-1] = 1/(s[-1]*f_tilde[-1,-1]);#only the last state can end the sequence. denominator ensures 1 sums
	b_tilde[-1,:] = 1/(s[-1]);#only the last state can end the sequence. denominator ensures 1 sums	
	for i in xrange(L-2,-1,-1):
		b_tilde[:,i] = 1/s[i]*(pdf_matrix[:,i+1]*b_tilde[:,i+1]*a).sum(axis=1)	 	
	#print b
	#termination
	#post = s*f*b
	#print post.sum(axis=0)
	#P = s[0,1:].prod()
	#print P	
	return s, f_tilde, b_tilde, pdf_matrix#return the pdf matrix to save extra computation


def train_test():
	a = array([[0.9, 0.1],[0.2,0.8]])
	e = array([[0.6, 0.4],[0.3,0.7]])
	print a
	print e
	(seq,states) = hmm_generate(1000,a,e)
	seq = array(seq,dtype=int)
	seqs = list()
	seqs.append(seq)
	(seq,states) = hmm_generate(1000,a,e)
	seq = array(seq,dtype=int)
	seqs.append(seq)
	hmm_train(seqs,a,e)

def Encode_train():
	from datetime import datetime
	from smooth import smooth
	print 'This is CNN'
	progress = open('Progress','w')
	progress.write('Starting now...\n')
	progress.write(str(datetime.now())+'\n')
	seqs = load('Chr6seqs.npy')
	seqs2 = [v for v in seqs[0:1000]]#only need first one thousand
	win = [-1/2.0,0,1/2.0]
	seqs3 = [v/mean(v[v>4]) for v in seqs2]#normalize the sequences by dividing by average of all above-threshold values
	seqs4 = [savitzky_golay(v,9,2,1,1) for v in seqs3]#apply filter to each sequence
	#seqs4 = [diff(v) for v in seqs3]#apply filter to each sequence
	#for v in seqs4:
	#	v[v>15] = 15
	#	v[v<-15] = -15	
	seqs4 = [v for v in seqs4 if max(v) < 10 and min(v) > -10]
	a = array([[0.9999,0.0001,0,0,0],[0,0.9,0.1,0,0],[0.0150,0.0450,0.9000,0.035,0.005],[0,0.03,0,0.97,0],[0,0,0,0,1]])
	#e = Emission([[0,0.5],[0.623,0.980],[-0.622,0.980],[0,0.5],[0,0.5]])
	e = Emission([[0,0.05],[0.5,1],[-0.5,1],[0,0.05],[0,0.05]])
	#a = load('Transition_guess1_17_13.npy')
	#e = Emission(load('Emission_guess1_17_13.npy'))
	#a = array([[ 0.9999696 ,  0.0000304 ,  0.        ,  0.        ,  0.        ],
       #[ 0.        ,  0.96005199,  0.03994801,  0.        ,  0.        ],
       #[ 0.01089723,  0.03910475,  0.9075653 ,  0.03617333,  0.00625939],
       #[ 0.        ,  0.03923139,  0.        ,  0.96076861,  0.        ],
       #[ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ]])	
	#e = Emission(array([[ 0,   5.58361326e-02],
       #[  1.57859225e+00,   2.73231590e+00],
       #[ -1.59598087e+00,   2.77078514e+00],
       #[  0,   5.04087195e-01],
       #[  0,   5.12224749e-01]]))
	#e = Emission([(-1.0360936142165903e-05, 0.0093638914366972036), (0.1036774861389085, 0.19935290201815278), (-0.11120533806832338, 0.20516367284238468), (0.0002275304045940573, 0.040616641255814594), (0.10308150145476273, 0.13669899797818449)])
	(a2,e2) = hmm_train2(seqs4,a,e)#train on the sequences,including transition
	progress.write('Saving transition matrix to Transition.npy\n')
	save("Transition",a2)
	progress.write('Saving emission parameters to Emission.npy\n')
	save('Emission',array(e2.p))
	progress.write('Ending now...\n')
	progress.write(str(datetime.now()) + '\n')
	progress.close()
	
"""
Trains on the top 1000 Hotspots in chromosome 6 using the parallelized functions
Start here: 3/1/13
"""
def Encode_train_parallel():
	cores = cpu_count()
	pool = Pool(processes=cores)
	chr6seqs = load('Chr6seqs.npy')
	chr6seqs1000 = chr6seqs[0:1000];
	r = load('K562_MTPN_promoter.npy')
	normfiltfunc = functools.partial(normalize_quantilemap,r=r)
	seqs = pool.map(normfiltfunc, chr6seqs1000)#training sequences
	folder = 'QuantileMapToMTPN/'
	progress = open(folder + 'Progress.txt','w')
	start = datetime.now()
	HS_stds = linspace(0.01,0.5,50);
	all_transition = list()
	all_emission = list()
	all_logP = list()
	for HS_std in HS_stds:#the program shouldn't be running longer than a day
		try:
			a_test = array([[0.9999,0.0001,0,0,0],[0,0.9,0.1,0,0],[0.0150,0.0450,0.9000,0.035,0.005],[0,0.03,0,0.97,0],[0,0,0,0,1]])
			e_test = Emission([[0,HS_std],[0.5,1],[-0.5,1],[0,1.1*HS_std],[0,HS_std]])
			warnings.simplefilter('error')
			a_updated, e_updated, new_logP = hmm_train_parallel2(seqs, a_test, e_test, pool, progress, HS_std)
			all_transition.append(a_updated)
			all_emission.append(e_updated.p)
			all_logP.append(new_logP)
		except RuntimeWarning:
			progress.write("Training failed... Trying again...\n\n")
	#Save the best performing parameters
	save(folder + 'Transitions', array(all_transition))
	save(folder + 'Emissions', array(all_emission))
	save(folder + 'logPs', array(all_logP))
	#Add time performance information
	stop = datetime.now()
	delta = stop - start
	days = str(delta.days)
	totalsecs = delta.seconds
	hours = str(totalsecs/3600)
	minutes = str(mod(totalsecs,3600)/60)
	seconds = str(mod(totalsecs,60))
	progress.write('Completed in ' + days + ' days, ' + hours + ' hours, ' + minutes + ' minutes, ' + seconds + ' seconds')
	progress.close()

def annotated_plot(v,threshold = 4):#plots a the Hotspot along with the annotated regions from the posterior decoding
	import pylab
	v2 = v/mean(v[v>threshold])
	v3 = savitzky_golay(v2, 9, 2, deriv=1, rate=1)
	a = load('Transition.npy')
	e = Emission(load('Emission.npy'))
	(s,f,b,pdf_m) = posterior_step(v3,a,e)
	post = s*f*b
	maxstates = post.argmax(axis=0)
	#label the footprints
	foots = (maxstates == 3)*1
	bins = diff(foots)
	start = where(bins == 1)[0] + 1
	stop = where(bins == -1)[0]
	for p,q in zip(start,stop):
		foot = pylab.axvspan(p, q, facecolor='r', alpha=0.5)
	#label the HS1
	hs1s = (maxstates == 0)*1
	bins = diff(hs1s)
	start = concatenate(([0],where(bins == 1)[0] + 1),1)#the first state is hs1. this accounts for that
	stop = where(bins == -1)[0]
	for p,q in zip(start,stop):
		hs1 = pylab.axvspan(p, q, facecolor='g', alpha=0.5)
	#label the HS2
	hs2s = (maxstates == 4)*1
	bins = diff(hs2s)
	start = where(bins == 1)[0] + 1
	stop = concatenate((where(bins == -1)[0],[len(v)-1]),1)#the last state is hs2
	for p,q in zip(start,stop):
		hs2 = pylab.axvspan(p, q, facecolor='c', alpha=0.5)
	pylab.plot(v)
	pylab.legend((hs1,foot,hs2,),('HS1','Footprint','HS2',))
	pylab.xlabel('DHS Coordinates')
	pylab.ylabel('DNase I Cuts')

"""
The mapping function used in outputFootingResults. Only the posterior probabilities
are needed
Input:
seq - a training sequence
a2 - guess for the transition state matrix
e2p - guess for the Emissions parameters. A numpy array (because it is picklable)
normfiltfunc - function for normalization and filtering
Output:
post - the posterior decoded probabilities matrix
"""
def output_parallel_map(seq, a2, e2p, normfiltfunc):
	a_guess = a2
	e_guess = Emission(e2p)
	seq2 = normfiltfunc(seq)
	(s, f_tilde, b_tilde, pdf_matrix) = posterior_step(seq2, a_guess, e_guess)
	post = s*f_tilde*b_tilde
	return post

"""
Function that writes all annotated Footprint sequences to a FASTA file
Intput:
filename - a string. all generated files start with this name
genome - a string. Filename of the genome fasta file
seqs - an iterable collection of numpy arrays representing Hotspot signal
regions - a PybedTools bedfile. holds the location of each Hotspot. corresponding order to seqs
a - a numpy array. The transition matrix between each state
e - an Emission object. The emission parameters for the states
offset - an integer. tells how much each footprint should be extended in both directions
pool - a multiprocessing pool object. For parallel processing
Output:
Writes all footprint sequences to a fasta file (filename.fasta), a wig file (filename.wig), footprint raw signal to .npy (filename_signal.npy), and sequence p-values to .npy (filename_p.npy). Each record in the fasta file is UCSC coordinates.
The wig file contains the sequence and the score (highest p-value). The wig file is generated first and it is then converted to a Fasta file using pybedtools.
"""
def outputFootprintingResults(filename, genome, seqs, regions, a, e, offset=0,pool):
	"""
	from pygr import seqdb
	from Bio import SeqIO
	from Bio.SeqRecord import SeqRecord
	from Bio.Seq import Seq
	g = seqdb.SequenceFileDB(genome)
	"""
	wigfile = open(filename + '.wig', 'w')
	rawsignals = list()
	psignals = list()
	posts = pool.map(functools.partial(output_parallel_map,a2=a,e2p=e.p,normfiltfunc=normalize_filter_threshold4),seqs)
	for seq, post, region in itertools.izip(seqs, posts,regions):
		maxstates = post.argmax(axis=0)
		foots = (maxstates == 3)*1
		bins = diff(foots)
		starts = where(bins == 1)[0] + 1 - offset#offsets are added to the start and stop positions
		stops = where(bins == -1)[0] + 1 + offset#the +1 is because the right index is exclusive
		chromosome = region[0]
		chrstart = int(region[1])
		for start, stop in zip(starts, stops):
			ftstart = chrstart + start#absolute chr start position of footprint in 0-base
			ftstop = chrstart + stop#absolute chr stop position of footprint in 0-base
			ftsignal = seq[start:stop]
			ftp = post[3][start:stop]
			rawsignals.append(ftsignal)
			psignals.append(ftp)
			wigfile.write(chromosome + '\t' + str(ftstart) + '\t' + str(ftstop) + '\t' + str(max(ftp)) + '\n')
	save(filename+"_signal",array(rawsignals))
	save(filename+"_p",array(psignals))	
	wigfile.close()
	bedfile = pybedtools.BedTool(filename + '.wig')
	bedfile.sequence(fi=genome,fo=filename + '.fa')#output the Fasta file
	

def f(x,a,ep):
	#a2 = array([[0.9999,0.0001,0,0,0],[0,0.9,0.1,0,0],[0.0150,0.0450,0.9000,0.035,0.005],[0,0.03,0,0.97,0],[0,0,0,0,1]])
	#e2 = Emission([[0,0.05],[0.5,1],[-0.5,1],[0,0.05],[0,0.05]])
	a2 = a
	e2 = Emission(ep)
	(s, f_tilde, b_tilde, pdf_matrix) = posterior_step(x, a2, e2)
	post = s*f_tilde*b_tilde
	es = pdf_matrix[:,1:].transpose()
	L = x.size
	logPi = log(s).sum()
	#Ai = a2*dot(f_tilde[:,0:L-1],es*b_tilde[:,1:].transpose())
	e_denominatori = post.sum(axis=1)
	mean_numeratori = (x*post).sum(axis=1)
	return (post, logPi, mean_numeratori, e_denominatori)

def parjobs():
	pool = Pool(processes=64)
	seqs = load('Chr6seqs.npy')
	seqs2 = [v for v in seqs[0:1000]]#only need first one thousand
	win = [-1/2.0,0,1/2.0]
	seqs3 = [v/mean(v[v>4]) for v in seqs2]#normalize the sequences by dividing by average of all above-threshold values
	seqs4 = [savitzky_golay(v,9,2,1,1) for v in seqs3]#apply filter to each sequence
	#seqs4 = [diff(v) for v in seqs3]#apply filter to each sequence
	#for v in seqs4:
	#	v[v>15] = 15
	#	v[v<-15] = -15	
	seqs4 = [v for v in seqs4 if max(v) < 10 and min(v) > -10]
	a2 = array([[0.9999,0.0001,0,0,0],[0,0.9,0.1,0,0],[0.0150,0.0450,0.9000,0.035,0.005],[0,0.03,0,0.97,0],[0,0,0,0,1]])
	#e = Emission([[0,0.5],[0.623,0.980],[-0.622,0.980],[0,0.5],[0,0.5]])
	e2 = Emission([[0,0.05],[0.5,1],[-0.5,1],[0,0.05],[0,0.05]])
	print datetime.now()
	print 'ok'
	seqs5 = [posterior_step(seq, a2, e2) for seq in seqs4]
	print 'not ok'
	print datetime.now()
	pool.map(functools.partial(f,a=a2,ep=e2.p),seqs4)
	print datetime.now()


"""
A pickle-able function for generating HMM gaussian sequences in parallel.
"""
def parallel_gen_function(x,a,ep):
	return hmm_generate(1000,a,Emission(ep))[0]

"""
A function to test the parallel algorithm for the Gaussian emission Baum-Welch training.
Benchmarks both parallel and non-parallel algorithms
"""	
def paralleltest(progress):
	atest = array([[0.9,0.1],[0.2,0.8]])
	eptest = array([[5,2],[-3,1.4]])
	etest = Emission(eptest)
	print datetime.now()
	print "Here are the parameters:"
	print atest
	print etest.p
	cores = cpu_count()
	pool = Pool(processes=cores)
	seqs = pool.map(functools.partial(parallel_gen_function,a=atest,ep=eptest),xrange(1000))#training sequences
	print datetime.now()
	#seqs2 = [hmm_generate(1000,atest,etest) for i in xrange(1000)]
	atest = array([[0.85,0.15],[0.3,0.7]])
	eptest = array([[4.0,1.0],[-2.2,1.0]])
	etest = Emission(eptest)
	hmm_train_parallel(seqs, atest, etest, pool, progress)
	print datetime.now()
	#hmm_train(seqs,atest,etest)
	#print datetime.now()
	
if __name__ == '__main__':
	#Encode_train()
	#parjobs()
	"""
	progress = open('Progress.txt','w')
	start = datetime.now()
	while (datetime.now() - start).days < 1:#the program shouldn't be running longer than a day
		try:
			warnings.simplefilter('error')
			paralleltest(progress)
			break
		except RuntimeWarning:
			progress.write("Training failed... Trying again...\n\n")
			break
	stop = datetime.now()
	delta = stop - start
	days = str(delta.days)
	totalsecs = delta.seconds
	hours = str(totalsecs/3600)
	minutes = str(mod(totalsecs,3600)/60)
	seconds = str(mod(totalsecs,60))
	progress.write('Completed in ' + days + ' days, ' + hours + ' hours, ' + minutes + ' minutes, ' + seconds + ' seconds')
	progress.close()
	"""
	Encode_train_parallel()	
		
