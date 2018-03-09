import numpy as np
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import scipy.stats as ss

# plotting settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')

# datafiles
ppds = ['cmb', 'loc']
sums = ['ptes', 'prs']

# posterior summaries
post_means = np.genfromtxt('gw_grb_h_0_posterior_means.csv', \
						   delimiter=',')
post_vars = np.genfromtxt('gw_grb_h_0_posterior_vars.csv', \
						  delimiter=',')
n_h_0_true = post_means.shape[0]
n_bs = post_means.shape[1]
print n_bs
h_0_true_col = [cm(col) for col in np.linspace(0.2, 0.8, n_h_0_true)]
fig, axes = mp.subplots(1, 2, figsize=(12, 5))
for i in range(n_h_0_true):
	print '* H_0 = {:5.2f}'.format(post_means[i, 0])
	to_print = 'posterior mean = {:5.2f} +/- {:4.2f}'
	print to_print.format(np.mean(post_means[i, 1:]), \
						  np.std(post_means[i, 1:]))
	to_print = 'posterior sigma = {:5.2f} +/- {:4.2f}'
	print to_print.format(np.mean(np.sqrt(post_vars[i, 1:])), \
						  np.std(np.sqrt(post_vars[i, 1:])))
	kde = ss.gaussian_kde(post_means[i, 1:])
	grid = np.linspace(np.min(post_means[i, 1:]), \
					   np.max(post_means[i, 1:]), \
					   1000)
	axes[0].plot(grid, kde.evaluate(grid), color=h_0_true_col[i])
	axes[0].axvline(post_means[i, 0], color=h_0_true_col[i], ls='--')
	kde = ss.gaussian_kde(np.sqrt(post_vars[i, 1:]))
	grid = np.linspace(np.min(np.sqrt(post_vars[i, 1:])), \
					   np.max(np.sqrt(post_vars[i, 1:])), \
					   1000)
	axes[1].plot(grid, kde.evaluate(grid), color=h_0_true_col[i], \
				 label=r'$H_0 = {:5.2f}$'.format(post_vars[i, 0]))
axes[0].set_xlabel(r'$\bar{H}_0$', fontsize=18)
axes[0].set_ylabel(r'${\rm Pr}(\bar{H}_0)$', fontsize=18)
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[1].set_xlabel(r'$\sigma_{H_0}$', fontsize=18)
axes[1].set_ylabel(r'${\rm Pr}(\sigma_{H_0})$', fontsize=18)
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].legend(loc='upper right', fontsize=14)
fig.suptitle('Bootstrap-Averaged Posterior Means / Sigmas', \
			 fontsize=18)
fig.savefig('gw_grd_h_0_bs_avg_posterior_moments.pdf', \
			bbox_inches = 'tight')
mp.close(fig)

# PPD summaries
for i in range(len(ppds)):

	for j in range(len(sums)):

		# read data
		fname = 'gw_grb_h_0_' + ppds[i] + '_ppd_' + sums[j]
		data = np.genfromtxt(fname + '.csv', delimiter=',')
		n_bs = data.shape[1]
		print n_bs

		# plot
		n_h_0_true = data.shape[0]
		fig, axes = mp.subplots(1, n_h_0_true, \
								figsize=(6 * n_h_0_true, 5))
		if ppds[i] == 'cmb':
			fig.suptitle(r'$\hat{H}_0^{\rm CMB}\, {\rm Prediction}$', \
						 fontsize=18)
		else:
			fig.suptitle(r'$\hat{H}_0^{\rm CDL}\, {\rm Prediction}$', \
						 fontsize=18)
		if sums[j] == 'ptes':
			x_label = r'$p$'
			y_label = r'${\rm Pr}(p)$'
		else:
			x_label = r'$\rho$'
			y_label = r'${\rm Pr}(\rho)$'
		for k in range(n_h_0_true):

			kde = ss.gaussian_kde(data[k, 1:])
			grid = np.linspace(np.min(data[k, 1:]), \
							   np.max(data[k, 1:]), \
							   1000)
			axes[k].plot(grid, kde.evaluate(grid), color=cm(0.5))
			axes[k].set_xlabel(x_label, fontsize=18)
			axes[k].set_ylabel(y_label, fontsize=18)
			axes[k].tick_params(axis='both', which='major', labelsize=12)
			axes[k].set_title(r'$H_0 = {:5.2f}$'.format(data[k, 0]), \
							  fontsize=18)

		# finish plot
		fig.savefig(fname + '.pdf', bbox_inches = 'tight')
		mp.close(fig)

# quick check of required numbers of samples
def rho(d, n, var_ratio, n_event_ref, n_event):
	d_n_event = n_event_ref / n_event
	return np.exp(-0.5 * rho_num(d, n, d_n_event) / \
				  rho_den(var_ratio, d_n_event))
def rho_num(d, n, d_n_event):
	if d > 0.0:
		return (d - n * np.sqrt(d_n_event)) ** 2
	else:
		return (d + n * np.sqrt(d_n_event)) ** 2
def rho_den(var_ratio, d_n_event):
	return var_ratio + d_n_event
def num_ratio(d, n, m, var_ratio):
	term = (m ** 2 * var_ratio - d ** 2)
	print term
	return [((-n * d - \
		      np.sqrt((n * d) ** 2 - term * (m ** 2 - n ** 2))) / \
		     term) ** 2, \
		    ((-n * d + \
		      np.sqrt((n * d) ** 2 - term * (m ** 2 - n ** 2))) / \
		     term) ** 2]

n_ref = 51.0
mu_obs = np.array([67.81, 73.24])
sig_obs = np.array([0.92, 1.74])
n_sigma_sv = 1.0
n_sigma_thresh = 3.0
n_sigma_diff = [(mu_obs[1] - mu_obs[0]) / np.sqrt(post_vars[i, 1]), \
				(mu_obs[0] - mu_obs[1]) / np.sqrt(post_vars[i, 1])]
var_ratio = [sig_obs[1] ** 2 / post_vars[i, 1], \
			 sig_obs[0] ** 2 / post_vars[i, 1]]

print n_sigma_diff
print var_ratio

n_req = np.zeros(2)
n_req[0] = n_ref * num_ratio(n_sigma_diff[0], n_sigma_sv, \
							 n_sigma_thresh, var_ratio[0])[0]
ln_rho = -2.0 * np.log(rho(n_sigma_diff[0], n_sigma_sv, \
						   var_ratio[0], n_ref, n_req[0]))
print n_req[0], ln_rho, n_sigma_thresh ** 2

n_req[1] = n_ref * num_ratio(n_sigma_diff[1], n_sigma_sv, \
							 n_sigma_thresh, var_ratio[1])[1]
ln_rho = -2.0 * np.log(rho(n_sigma_diff[1], n_sigma_sv, \
						   var_ratio[1], n_ref, n_req[1]))
print n_req[1], ln_rho, n_sigma_thresh ** 2

n_grid = np.arange(n_ref, 5000.0)
mp.loglog(n_grid, rho_num(n_sigma_diff[0], n_sigma_sv, n_ref / n_grid), 'r', lw=1.0)
mp.plot(n_grid, 1.0 / rho_den(var_ratio[0], n_ref / n_grid), 'g', lw=1.0)
mp.plot(n_grid, 1.0 / rho_den(var_ratio[1], n_ref / n_grid), 'b', lw=1.0)
mp.plot(n_grid, -2.0 * np.log(rho(n_sigma_diff[0], n_sigma_sv, var_ratio[0], \
								  n_ref, n_grid)), 'g')
mp.plot(n_grid, -2.0 * np.log(rho(n_sigma_diff[1], n_sigma_sv, var_ratio[1], \
								  n_ref, n_grid)), 'b')
mp.axhline(n_sigma_thresh ** 2, color='k', linestyle='-.')
mp.axvline(n_req[0], color='g', linestyle='-.')
mp.axvline(n_req[1], color='b', linestyle='-.')
mp.xlabel(r'$N$')
mp.ylabel(r'$f(N)$')
mp.xlim(n_ref, 5000)
mp.ylim(0.3, 40.0)
mp.savefig('gw_grb_h_0_ppd_samp_var_limits.pdf', bbox_inches='tight')
mp.show()
exit()

print num_ratio(4.53, n_sigma_sv, n_sigma_thresh, 2.1)

print 5.43, mu_obs[1] - mu_obs[0]
print 1.2, np.sqrt(post_vars[i, 1])
print 5.43 / 1.2, n_sigma_diff[0]

m = 3.0
n = 1.0
d = 3.77 # 4.53
vrat = 1.46 # 2.1
print ((d*n+np.sqrt((d*n)**2-(vrat*m**2-d**2)*(m**2-n**2)))/(vrat*m**2-d**2))**2
