import lmfit as lm
import numpy as np

# Defines the double exponential model with constant offset
# y = c1 exp(-t/tau1) + c2 exp(-t/tau2) + c3
def multiexp_decay(t,c1,c2,c3,tau1,tau2):
    return c1*np.exp(-t/tau1)+c2*np.exp(-t/tau2)+c3

# here are the true parameters of the model for 
# generating synthetic data
c1 = 2.2
c2 = 6.8
c3 = 1.6
tau1 = 2.4
tau2 = 6.0
ndata = 1000

np.random.seed(0xdeadbeef)
tdata = np.linspace(0,20,ndata)
ydata = multiexp_decay(tdata,c1,c2,c3,tau1,tau2)+np.random.normal(size=ndata,scale=0.01);

model = lm.Model(multiexp_decay)

# guess initial parameters and fit model
params = model.make_params(c1 = 1.0,c2 = 5, c3 = 0.3,tau1 = 1.,tau2 = 7.)
result = model.fit(ydata,params,t=tdata)

print(result.fit_report())
cb = result.eval_uncertainty(sigma=0.88,dscale = 0.0001);

# print(cb)
print(result.ci_report())
