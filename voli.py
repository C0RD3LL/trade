# This is a simple model on predicting volatility of a stock
#source venv/bin/activate

import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
import yfinance as yf
import pandas as pd
import datetime
import time
from arch import arch_model
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2021, 8, 1)
s_p500 = yf.download("SPY", start=start, end = end, interval='1d')

ret = 100 * (s_p500.pct_change()[1:]['Adj Close'])
realized_vol = ret.rolling(5).std()
retv= ret.values
#ARCH Model.
n = 252     #Defining the split location and assign the splitted data to split variable. 
split_date= ret.iloc[-n:].index     #Calculating variance of S&P-500.

sgm2= ret.var()        #Calculating kurtosis of S&P-500.
K = ret.kurtosis()     #Identifying the initial value for slope coefficient alpha

alpha= (-3.0* sgm2 + np.sqrt(9.0 * sgm2 ** 2 - 12.0 * (3.0*sgm2-K)*K))/(6*K)
omega = (1 - alpha) * sgm2    #Identifying the initial value for constant term omega

initial_parameters = [alpha, omega]


@jit(nopython=True, parallel=True)  #Using paralel processing to decrease the processing time.
def arch_likelihood(initial_parameters, retv):
    omega = abs(initial_parameters[0])  #Taking absolute values and assigning the initial values into related variables.
    alpha = abs(initial_parameters[1])  #Taking absolute values and assigning the initial values into related variables.
    T = len(retv)
    logliks = 0
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(retv) #Identifying the initial values of volatility. 
    for t in range(1, T):
        sigma2[T] = omega + alpha * (retv[t - 1]) ** 2 #Iterating the variance of S&P-500.
        logliks = np.sum(0.5 * (np.log(sigma2)+retv ** 2 / sigma2)) #Calculation log-likelihood.
    return logliks

def opt_params(x0, retv):
    opt_result = opt.minimize(arch_likelihood, x0=x0, args = (retv),
                              method='Nelder-Mead', 
                              options={'maxiter': 5000})  #Minimizing the log-likelihood function.
    params = opt_result.x  #Creating a variable params for optimized parameters.
    print('\nResults of Nelder-Mead minimization\n{}\n{}'
          .format(''.join(['-'] * 28), opt_result)) 
    print('\nResulting params = {}'.format(params))
    return params 

params= opt_params(initial_parameters, retv)

def arch_apply(ret):
    omega = params[0]
    alpha = params[1]
    T= len(ret)
    sigma2_arch= np.zeros(T+1)
    sigma2_arch[0]= np.var(ret)
    for t in range(1,T):
        sigma2_arch[t]= omega +alpha * ret[t-1] ** 2
    return sigma2_arch

sigma2_arch = arch_apply(ret)

# Test Methods 

arch = arch_model(ret, mean='zero', vol='ARCH', p=1).fit(disp='off')
#print(arch.summary())















