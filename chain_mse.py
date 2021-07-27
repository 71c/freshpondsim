import numpy as np
from scipy.stats import norm

n_samples = 20_000
n_samples_2 = 1_000

theta = 5.7

bias_thetahat_theta = -10.0
var_thetahat_theta = 2.5
std_thetahat_theta = np.sqrt(var_thetahat_theta)
theta_hats = theta + norm.rvs(bias_thetahat_theta, std_thetahat_theta, n_samples)

bias_thetahathat_thetahat = 3.0
var_thetahathat_thetahat = 0.6
std_thetahathat_thetahat = np.sqrt(var_thetahathat_thetahat)
theta_hathats = norm.rvs(bias_thetahathat_thetahat, std_thetahathat_thetahat, (n_samples_2, n_samples)) + theta_hats

est_mse_thetahat_theta = np.mean((theta_hats - theta)**2)

est_mses_thetahathat_thetahat = np.mean((theta_hathats - theta_hats)**2, axis=0)
assert est_mses_thetahathat_thetahat.shape[0] == n_samples
est_expected_mse_thetahathat_thetahat = est_mses_thetahathat_thetahat.mean()

est_pred_mse_thetahathat_theta = est_mse_thetahat_theta + est_expected_mse_thetahathat_thetahat + 2 * bias_thetahat_theta * bias_thetahathat_thetahat
est_mse_thetahathat_theta = np.mean((theta_hathats - theta)**2)
print(est_pred_mse_thetahathat_theta, est_mse_thetahathat_theta)



# est_bias_thetahathat_theta = np.mean(theta_hathats - theta)
# print(est_bias_thetahathat_theta, bias_thetahathat_thetahat + bias_thetahat_theta)


# var_thetahathat_thetahats = np.var(theta_hathats, axis=0)
# avg_var_thetahathat_thetahat = np.mean(var_thetahathat_thetahats)

# bias_thetahathat_thetahats = np.mean(theta_hathats - theta_hats, axis=0)
# var_bias_thetahathat_thetahat = np.var(bias_thetahathat_thetahats)


# var_thetahathat_theta = np.var(theta_hathats)

# print(var_thetahathat_theta, avg_var_thetahathat_thetahat + var_thetahat_theta + var_bias_thetahathat_thetahat)
