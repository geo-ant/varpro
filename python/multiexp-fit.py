import lmfit as lm
import numpy as np

# Defines the double exponential model with constant offset
# y = c1 exp(-t/tau1) + c2 exp(-t/tau2) + c3
def multiexp_decay(t,c1,c2,c3,tau1,tau2):
    return c1*np.exp(-t/tau1)+c2*np.exp(-t/tau2)+c3

# here are the true parameters of the model for 
# generating synthetic data
c1 = 3.4
c2 = 9.8
c3 = 1.6
tau1 = 2.4
tau2 = 6.0

tdata = np.linspace(0,15,1000)
ydata = multiexp_decay(tdata,c1,c2,c3,tau1,tau2)

model = lm.Model(multiexp_decay)

# guess initial parameters and fit model
params = model.make_params(c1 = 1.0,c2 = 5, c3 = 0.3,tau1 = 1.,tau2 = 9.)
result = model.fit(ydata,params,t=tdata)

print(result.fit_report())
