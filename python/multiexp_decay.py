import lmfit as lm
import numpy as np

# Defines the double exponential model with constant offset
# y = c1 exp(-t/tau1) + c2 exp(-t/tau2) + c3
# This is a separable model: linear coefficients (c1,c2,c3) and nonlinear parameters (tau1,tau2)
def multiexp_decay(t,c1,c2,c3,tau1,tau2):
    return c1*np.exp(-t/tau1)+c2*np.exp(-t/tau2)+c3

# Here are the true parameters of the model for
# generating synthetic data
c1 = 2.2
c2 = 6.8
c3 = 1.6
tau1 = 2.4
tau2 = 6.0
ndata = 1000

# Set random seed for reproducible results
np.random.seed(0xdeadbeef)

# Generate time points from 0 to 20 with ndata points
tdata = np.linspace(0,20,ndata)
# Create synthetic data with the true parameters and add small Gaussian noise
ydata = multiexp_decay(tdata,c1,c2,c3,tau1,tau2)+np.random.normal(size=ndata,scale=0.01)

# Create model using lmfit
model = lm.Model(multiexp_decay)

# Set initial parameter guesses and perform model fitting
# Note that initial values are intentionally different from the true values
params = model.make_params(c1=1.0, c2=5.0, c3=0.3, tau1=1.0, tau2=7.0)
result = model.fit(ydata, params, t=tdata)

# Print the fitting results report
print(result.fit_report())
# Calculate confidence intervals (88% confidence level)
confidence_radius = result.eval_uncertainty(sigma=0.88, dscale=0.0000001)

# Uncomment to print additional confidence information
# print(confidence_radius)
# print(result.ci_report())

# Write the data, confidence bands, and covariance matrix to disk
# as raw 64-bit float values for comparison with Rust implementation
tdata.astype(np.float64).tofile(f"xdata_{ndata}_64bit.raw")
ydata.astype(np.float64).tofile(f"ydata_{ndata}_64bit.raw")
confidence_radius.astype(np.float64).tofile(f"conf_{ndata}_64bit.raw")

# Get and save the covariance matrix
cov = result.covar.astype(np.float64)
cov.tofile(f"covmat_{cov.shape[0]}x{cov.shape[1]}_64bit.raw")
