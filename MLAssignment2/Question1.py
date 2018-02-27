
# coding: utf-8

import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, DotProduct,WhiteKernel, ExpSineSquared)
from sklearn.datasets import fetch_mldata
from sklearn import gaussian_process as GP
from matplotlib import pyplot as plt

path = "C:/Users/anand/OneDrive/University McGill/Machine Learning/Assignment2/mauna.csv"


data = []
with open(path, 'rb') as text_file:
    for x in text_file:
        data.append(x)


############################################## Part (a) ###############################################
print("###################################################### Part (a) ##########################################################")
#Squared Exponential Covariance Function (Kernel)
rbf_kernel = 1 * RBF(length_scale=1, length_scale_bounds=(1e-1, 10.0))
#Gaussian Process instatiation
gp = GaussianProcessRegressor(kernel=rbf_kernel)


plt.figure(1, figsize=(7,5))
X_data = np.linspace(0, 5, 10)
y_mean, y_std = gp.predict(X_data[:, np.newaxis], return_std=True)

plt.plot(X_data, y_mean, 'k', lw=3, zorder=9, label = '*')
plt.fill_between(X_data, y_mean - y_std, y_mean + y_std, alpha=0.2, color='k')
y_samples = gp.sample_y(X_data[:, np.newaxis], 10)
plt.plot(X_data, y_samples, lw=1, label = 'x')
plt.title("Prior (kernel:  %s)" % rbf_kernel, fontsize=12)
        
plt.show()        


V0 = [1,100]
l=np.linspace(1e-1, 10.0,5)
kernels = []
for i in V0:
    for j in l:
        kernels.append(i * RBF(length_scale=j, length_scale_bounds=(1e-1, 10.0)))
for fig_index, kernel in enumerate(kernels):
#Gaussian Process instatiation   
    gp = GaussianProcessRegressor(kernel=kernel)

    # Plot the graph of prior
    plt.figure(fig_index, figsize=(5,4))
    X_data = np.linspace(0, 5, 100)
    y_mean, y_std = gp.predict(X_data[:, np.newaxis], return_std=True)
    plt.plot(X_data, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_data, y_mean - y_std, y_mean + y_std, alpha=0.2, color='k')
    y_samples = gp.sample_y(X_data[:, np.newaxis], 10)
    plt.plot(X_data, y_samples, lw=1)
    
    plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)
    plt.tight_layout()

plt.show()


######################################################### Part (c) #################################################################
print("###################################################### Part (c) ##########################################################")
# Specify Gaussian Process
rbf_kernel = 1.1 * RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1.0))
gp = GaussianProcessRegressor(kernel=rbf_kernel)

# Generate data and fit GP
rng = np.random.RandomState(4)
X = rng.uniform(0, 10, 50)[:, np.newaxis]
y = X[:, 0] * np.sin((X[:, 0]))
gp.fit(X, y)

# Plot posterior
X_data = np.linspace(0, 15, 100)
y_mean, y_std = gp.predict(X_data[:, np.newaxis], return_std=True)
plt.plot(X_data, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_data, y_mean - y_std, y_mean + y_std, alpha=0.2, color='k')

y_samples = gp.sample_y(X_data[:, np.newaxis], 10)
plt.plot(X_data, y_samples, lw=1)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))

plt.title("Posterior mean (kernel: %s)\n Log-Likelihood: %.3f" % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),fontsize=12)
plt.tight_layout()
plt.plot()
plt.show()

#plotting Standard deviation of GP
plt.plot(X_data,y_std)
plt.title( 'Standard Deviation of GP')
plt.xlabel('x')
plt.ylabel('Standard Deviation')
plt.show()


############################################## Part (c) i #####################################################
print("###################################################### Part (c) - i ##########################################################")
# Specify Gaussian Process
rbf10_kernel = 1.1 * RBF(length_scale=10.0, length_scale_bounds=(1e-3, 10.0))
gp10 = GaussianProcessRegressor(kernel=rbf10_kernel)

# Generate data and fit GP
rng = np.random.RandomState(4)
X = rng.uniform(0, 10, 50)[:, np.newaxis]
y = X[:, 0] * np.sin((X[:, 0]))
gp10.fit(X, y)

# Plot posterior
X_data = np.linspace(0, 15, 100)
y_mean, y_std = gp10.predict(X_data[:, np.newaxis], return_std=True)
plt.plot(X_data, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_data, y_mean - y_std, y_mean + y_std, alpha=0.2, color='k')

y_samples_10 = gp10.sample_y(X_data[:, np.newaxis], 10)
plt.plot(X_data, y_samples_10, lw=1)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))

plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
          % (gp10.kernel_, gp10.log_marginal_likelihood(gp10.kernel_.theta)), fontsize=12)
plt.tight_layout()
plt.plot()
plt.show()

# y_sample_10 corresponds to samples drawn from GP with length scale = 10
# y_sample corresponds to samples drawn from GP with length scale = 1
y_10_pd = pd.DataFrame(y_samples_10)
y_10_pd_desc = y_10_pd.describe()
y_10_pd_desc_ = y_10_pd_desc.T.max()
y_pd= pd.DataFrame(y_samples)
y_pd_desc = y_pd.describe()
y_pd_desc_= y_pd_desc.T.max()
result = pd.concat([y_pd_desc_,y_10_pd_desc_], axis=1)
result.columns = ['length-scale = 1', 'length-scale = 10']
print(result)


########################################################### Part (c) ii ############################################################
print("###################################################### Part (c) - ii ##########################################################")
# Specify Gaussian Process
kernel_dot = DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) ** 2
gp_dot = GaussianProcessRegressor(kernel=kernel_dot)
# Generate data and fit GP
rng = np.random.RandomState(4)
X = rng.uniform(0, 10, 50)[:, np.newaxis]
y = X[:, 0] * np.sin((X[:, 0]))
gp_dot.fit(X, y)

# Plot posterior
X_data = np.linspace(0, 15, 100)
y_mean, y_std = gp_dot.predict(X_data[:, np.newaxis], return_std=True)
plt.plot(X_data, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_data, y_mean - y_std, y_mean + y_std, alpha=0.2, color='k')

y_samples_dot = gp_dot.sample_y(X_data[:, np.newaxis], 10)
plt.plot(X_data, y_samples_dot, lw=1)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
# plt.xlim(0, 5)
# plt.ylim(-3, 3)
plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f" % (gp_dot.kernel_, gp_dot.log_marginal_likelihood(gp_dot.kernel_.theta)), fontsize=12)
plt.tight_layout()
plt.plot()
plt.show()

# y_samples_dot corresponds to samples drawn from GP with dotproduct Kernel
# y_sample corresponds to samples drawn from GP with length scale = 1
y_dot_pd = pd.DataFrame(y_samples_dot)
y_dot_pd_desc = y_dot_pd.describe()
y_dot_pd_desc_ = y_dot_pd_desc.T.max()
result = pd.concat([y_pd_desc_,y_dot_pd_desc_], axis=1)
result.columns = ['RBF kernel', ' Dotproduct Kernel']
print(result)


############################################ Part (c) iii ######################################################################
print("###################################################### Part (c) - iii ##########################################################")
rational_kernel = 1.0 * RationalQuadratic(length_scale=0.1 , alpha=0.1)
gp_rational= GaussianProcessRegressor(kernel=rational_kernel)
gp_rational.fit(X, y)

# Plot posterior
X_data = np.linspace(0, 15, 100)
y_mean, y_std = gp_rational.predict(X_data[:, np.newaxis], return_std=True)
plt.plot(X_data, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_data, y_mean - y_std, y_mean + y_std,
                 alpha=0.2, color='k')

y_samples_rational = gp_rational.sample_y(X_data[:, np.newaxis], 10)
plt.plot(X_data, y_samples_rational, lw=1)
plt.scatter(X[:, 0], y, c='r', zorder=10, edgecolors=(0, 0, 0))
plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"% (gp_rational.kernel_, gp_rational.log_marginal_likelihood(gp_rational.kernel_.theta)),fontsize=12)
plt.tight_layout()
plt.plot()
plt.show()

# y_samples_rational corresponds to samples drawn from GP with rationalproduct Kernel
# y_sample corresponds to samples drawn from GP with length scale = 1
y_rational_pd = pd.DataFrame(y_samples_rational)
y_rational_pd_desc = y_rational_pd.describe()
y_rational_pd_desc_ = y_rational_pd_desc.T.max()
result = pd.concat([y_pd_desc_,y_rational_pd_desc_], axis=1)
result.columns = ['RBF kernel', ' Rational Kernel']
print(result)

############################################### Part (c) iv ############################################################
print("###################################################### Part (c) - iv ##########################################################")

kernel = DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) + RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gp= GaussianProcessRegressor(kernel=kernel)
gp.fit(X, y)

# Plot posterior
X_data = np.linspace(0, 15, 100)
y_mean, y_std = gp.predict(X_data[:, np.newaxis], return_std=True)
plt.plot(X_data, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_data, y_mean - y_std, y_mean + y_std,
                 alpha=0.2, color='k')

y_samples_multi_kernel = gp.sample_y(X_data[:, np.newaxis], 10)
plt.plot(X_data, y_samples_multi_kernel, lw=1)
plt.scatter(X[:, 0], y, c='r', zorder=10, edgecolors=(0, 0, 0))
plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"% (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),fontsize=12)
plt.tight_layout()
plt.plot()
plt.show()

# y_samples_multi_kernel corresponds to samples drawn from GP with multi_kernelproduct Kernel
# y_sample corresponds to samples drawn from GP with length scale = 1
y_multi_kernel_pd = pd.DataFrame(y_samples_multi_kernel)
y_multi_kernel_pd_desc = y_multi_kernel_pd.describe()
y_multi_kernel_pd_desc_ = y_multi_kernel_pd_desc.T.max()
result = pd.concat([y_pd_desc_,y_dot_pd_desc_,y_rational_pd_desc_,y_multi_kernel_pd_desc_], axis=1)
result.columns = ['RBF kernel', 'Dot product Kernel','Rational Quadratic Kernel','sum of RBF kernel and a RationalQuadratic kernel']
print(result)

################################################### Part (d) ##################################################################
print("###################################################### Part (d) ##########################################################")

# Generate data and fit GP, now with Gaussian
rng = np.random.RandomState(4)
X = rng.uniform(0, 10, 50)[:, np.newaxis]
eps = np.random.normal(0,0.5,50) # the noise term
y = X[:, 0] * np.sin((X[:, 0])) + eps

################################################# Part (d) i ###################################################################
print("###################################################### Part (d) - i ##########################################################")

sum_kernel = DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) + 1.0 * RBF(length_scale=10.0, length_scale_bounds=(1e-1, 10.0))
kernels = [1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 100.0)),
           RationalQuadratic(length_scale=1.0, alpha=0.1),DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)),sum_kernel]

### log_lik_est is used to store log likelihood estimation

log_lik_est = {}

for fig_index, kernel in enumerate(kernels):
    # Specify Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel)
    # Generate data and fit GP
    gp.fit(X, y)
    log_lik_est [("%s") %(kernel)] = gp.log_marginal_likelihood_value_
    # Plot posterior
    plt.figure(fig_index, figsize=(8, 8))
    plt.subplot(2, 1, 1)
    X_data = np.linspace(0, 15, 50)
    y_mean, y_std = gp.predict(X_data[:, np.newaxis], return_std=True)
    plt.plot(X_data, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_data, y_mean - y_std, y_mean + y_std, alpha=0.2, color='k')

    y_samples = gp.sample_y(X_data[:, np.newaxis], 10)
    plt.plot(X_data, y_samples, lw=1)
    plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
#     plt.xlim(0, 5)
#     plt.ylim(-3, 3)
    plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f" % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)), fontsize=12)
    plt.tight_layout()

    
plt.show()

################################### Part (d) ii ######################################################################
print("###################################################### Part (d) - ii ##########################################################")

sum_kernel = DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) + 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) +  WhiteKernel()

kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(),
           RationalQuadratic(length_scale=1.0, alpha=0.1)+ WhiteKernel(),
           DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0))+ WhiteKernel(),sum_kernel]

### log_lik_est is used to store log likelihood estimation
log_lik_est_white = {}
for fig_index, kernel in enumerate(kernels):
    # Specify Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel)
    # Generate data and fit GP
    gp.fit(X, y)
    log_lik_est_white[("%s") %(kernel)] = gp.log_marginal_likelihood_value_
    # Plot posterior
    plt.figure(fig_index, figsize=(8, 8))
    plt.subplot(2, 1, 1)
    X_data = np.linspace(0, 15, 50)
    y_mean, y_std = gp.predict(X_data[:, np.newaxis], return_std=True)
    plt.plot(X_data, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_data, y_mean - y_std, y_mean + y_std, alpha=0.2, color='k')

    y_samples = gp.sample_y(X_data[:, np.newaxis], 10)
    plt.plot(X_data, y_samples, lw=1)
    plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    
    plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f" % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)), fontsize=12)
    plt.tight_layout()
    
plt.show()

######################################################### Part (d) iii #######################################################
print("###################################################### Part (d) - iii ##########################################################")

# the dictionary containing log marginal likelihood value associated with each kernel in the model
print(log_lik_est) 

# containing log marginal likelihood value associated with each kernel in the model( white kernels included)
print(log_lik_est_white) 

######################################### Part (e) ####################################################################
print("###################################################### Part (e)  ##########################################################")

# importing the data
data = fetch_mldata('mauna-loa-atmospheric-co2').data
X = data[:, [1]]
y = data[:, 0]

# ploting the data
plt.plot(X,y)
plt.xlabel("Year")
plt.ylabel(r"CO$_2$ in ppm")
plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
plt.grid()
plt.show()


# Kernel with parameters given in GPML book
k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0)     * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2     * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134)     + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, optimizer=None, normalize_y=True)
gp.fit(X, y)

X_data = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis] 
y_pred, y_std = gp.predict(X_data, return_std=True)

# Illustration
plt.scatter(X, y, c='k')
plt.plot(X_data, y_pred)
plt.fill_between(X_data[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.5, color='k')
plt.xlim(X_data.min(), X_data.max())
plt.xlabel("Year")
plt.ylabel(r"CO$_2$ in ppm")
plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
plt.tight_layout()
plt.show()

print(gp.log_marginal_likelihood_value_)
