### INPUT ARGUMENTS
#1. theta value
#2. experimental SAXS profile
#3. calculated SAXS profiles file
#4. output dir
#5. prior weights (COMPULSORY: if not set uniform weights will be used)

import numpy as np
import sys
import os, os.path
sys.path.append('/groups/sbinlab/fpesce/scripts/BME') #CHECK THIS!!!! POINT IT TO THE FOLDER WHERE YOU HAVE THE BME CODE
import bme_reweight as bme
from sklearn.linear_model import LinearRegression

t = int(sys.argv[1])
exp_file = sys.argv[2]
calc_file = sys.argv[3]
dir_ = sys.argv[4]
if len(sys.argv) == 6:
    w_in = np.loadtxt(sys.argv[5])


inp_exp = np.loadtxt(exp_file)
exp = inp_exp[...,1]
err = inp_exp[...,2]
wlr = 1/(err**2)

def col0(frame):
    return(int(frame[5:]))
calc = np.loadtxt(calc_file,converters = {0: col0})[...,1:]

c = 1
data = []
weights = []
ab = []

if len(sys.argv) == 6:
    Iav = np.average(calc,axis=0,weights=w_in)
else:
    Iav = np.average(calc,axis=0)

tol = 0.001

while c <= 15: #Max N cycles
    
    print("Iteration "+str(c))

    #Using LinearRegression module from scikit learn library to perform linear regression and obtain scale factor/offset
    model = LinearRegression()
    model.fit(Iav.reshape(-1,1),exp,wlr) #Here weights are introduced in doing linear regression
    alpha = model.coef_[0] #Scale factor
    beta = model.intercept_ #Offset
    
    ab.append(np.array([c,alpha,beta]))

    #Applying scale factor and offset obtained in this cycle to the Pepsi-SAXS ensemble and write to file to feed BME
    calc = alpha*calc+beta
    new_calc = open(dir_+'/tmp_t'+str(t), 'w')
    new_calc.write('#\n')
    for i in range(0,np.shape(calc)[0]):
        new_calc.write('frame'+str(i)+' '+' '.join(str(n) for n in calc[i])+'\n')
    new_calc.close()

    #Running BME with the scaled/shifted ensemble
    if len(sys.argv) == 6:
    	rew = bme.Reweight(w0=w_in)
    	rew.load(exp_file, dir_+'/tmp_t'+str(t))
    	chi2_before, chi2_after, srel = rew.optimize(theta=t)
    	data.append(np.array([c, t, chi2_before, chi2_after, np.exp(srel)]))
    	print(data[-1])
    	weights.append(rew.get_weights())
    else:
        rew = bme.Reweight()
        rew.load(exp_file, dir_+'/tmp_t'+str(t))
        chi2_before, chi2_after, srel = rew.optimize(theta=t)
        data.append(np.array([c, t, chi2_before, chi2_after, np.exp(srel)]))
        print(data[-1])
        weights.append(rew.get_weights())
    
    #Weighted average of the Pepsi-SAXS profiles with BME weights
    Iav = np.average(calc,axis=0,weights=weights[-1])

    c += 1


np.savetxt(dir_+'/weights_t'+str(t), weights, fmt='%10.5f', delimiter=' ', newline='\n')
np.savetxt(dir_+'/data_t'+str(t), data, fmt='%10.5f', delimiter=' ', newline='\n')
np.savetxt(dir_+'/ab_t'+str(t), ab, fmt='%10.5f', delimiter=' ', newline='\n')
