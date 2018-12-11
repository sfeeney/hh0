import numpy as np
import numpy.random as npr
import scipy.stats as ss
import scipy.optimize as so
import os
notk = False
if 'DISPLAY' not in os.environ.keys():
		notk = True
elif os.environ['DISPLAY'] == ':0':
		notk = True
if notk:
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import pickle
import pystan as ps
import fnmatch
import getdist as gd
import getdist.plots as gdp
import h5py

def ln_post_h_0_alt(h_0, z_obs, d_obs):
	n_event = len(z_obs)
	ln_post = np.zeros(len(h_0))
	#ln_post = -np.log(h_0) * n_event
	for i in range(n_event):
		ln_post -= 0.5 * ((h_0_grid / c / z_true[i] - \
						   1.0 / d_obs[i]) * amp_s / amp_n) ** 2
	return ln_post

def lprob(h_0, d_obs, z_obs, var_v, var_d, n):
	var_tot = var_d + var_v / h_0 ** 2
	ln_post = -np.log(h_0) * n - \
			  np.log(2.0 * np.pi * var_tot) * n / 2.0 - \
			  (c * z_obs / h_0 - d_obs) ** 2 / var_tot * n / 2.0
	return ln_post

def lprob_no_n(h_0, d_obs, z_obs, var_v, var_d, n):
	var_tot = var_d + var_v / h_0 ** 2
	ln_post = np.log(2.0 * np.pi * var_tot) * n / 2.0 - \
			  (c * z_obs / h_0 - d_obs) ** 2 / var_tot * n / 2.0
	return ln_post

def lprob_norm_like(h_0, d_obs, z_obs, var_v, var_d, n):
	var_tot = var_d * h_0 ** 2 + var_v
	var_y = var_v * var_d * h_0 ** 2 / var_tot
	mu_y = (-c * z_obs * var_d * h_0 ** 2 + \
			h_0 * d_obs * var_v) / var_tot
	ln_post = -np.log(h_0) * 2 * n - \
			  np.log(2.0 * np.pi * var_tot) * n / 2.0 - \
			  (c * z_obs - h_0 * d_obs) ** 2 / var_tot * n / 2.0 + \
			  np.log(mu_y ** 2 + var_y) * n
	return ln_post

def lprob_sample(h_0, d_obs, z_obs, v_obs, var_v, var_d):
	var_tot = var_d + var_v / h_0 ** 2
	n_h_0 = len(h_0)
	n_event = len(d_obs)
	ln_post = -np.log(h_0) * n_event - \
			  np.log(2.0 * np.pi * var_tot) * n_event / 2.0
	for i in range(n_h_0):
		ln_post[i] -= np.sum(((c * z_obs - v_obs) / h_0[i] - d_obs) ** 2 / \
							 var_tot[i]) / 2.0
	return ln_post

def lprob_sample_no_n(h_0, d_obs, z_obs, v_obs, var_v, var_d):
	var_tot = var_d + var_v / h_0 ** 2
	n_h_0 = len(h_0)
	n_event = len(d_obs)
	ln_post = np.log(2.0 * np.pi * var_tot) * n_event / 2.0
	for i in range(n_h_0):
		ln_post[i] -= np.sum(((c * z_obs - v_obs) / h_0[i] - d_obs) ** 2 / \
							 var_tot[i]) / 2.0
	return ln_post

def lprob_sample_alt(h_0, d_obs, z_obs, v_obs, std_v, std_d):
	std_tot = np.sqrt(std_d ** 2 + (std_v / h_0) ** 2)
	n_h_0 = len(h_0)
	n_event = len(d_obs)
	ln_post = np.zeros(n_h_0)
	for i in range(n_event):
		ln_post += ss.norm.logpdf((c * z_obs - v_obs[i]) / h_0, \
								  loc=d_obs[i], \
								  scale=std_tot) - np.log(h_0)
	return ln_post

def log_like_d_cos_i(h_plus_obs, h_cross_obs, d_grid, cos_i_grid):
	n_d = d_grid.shape[0]
	n_cos_i = cos_i_grid.shape[0]
	log_like = np.zeros((n_d, n_cos_i))
	for i in range(n_d):
		h_plus_model = amp_s * (1.0 + cos_i_grid ** 2) / 2.0 / d_grid[i]
		h_cross_model = -amp_s * cos_i_grid / d_grid[i]
		log_like[i, :] = -0.5 * ((h_plus_obs - h_plus_model) / amp_n) ** 2 + \
						 -0.5 * ((h_cross_obs - h_cross_model) / amp_n) ** 2
	return log_like

def log_like_h_0d(z_obs, v_pec_obs, sig_v_pec, d_grid, h_0_grid):
	n_d = d_grid.shape[0]
	n_h_0 = h_0_grid.shape[0]
	log_like = np.zeros((n_d, n_h_0))
	for i in range(n_d):
		h_0d = h_0_grid * d_grid[i]
		log_like[i, :] = -0.5 * ((h_0d - c * z_obs + v_pec_obs) / sig_v_pec) ** 2
	return log_like

def log_like_h_0_gmm(p_gmm, z_obs, v_pec_obs, sig_v_pec, h_0_grid):
	n_comp = len(p_gmm) / 3
	a = p_gmm[0::3]
	mu = p_gmm[1::3]
	sig = p_gmm[2::3]
	n_h_0 = h_0_grid.shape[0]
	like = np.zeros(n_h_0)
	for i in range(n_comp):
		var_tot = (h_0_grid * sig[i]) ** 2 + sig_v_pec ** 2
		like += a[i] * sig[i] * \
				np.exp(-0.5 * (h_0_grid * mu[i] - \
							   c * z_obs + v_pec_obs) ** 2 / var_tot) / \
				np.sqrt(var_tot)
	return np.log(like)

def gmm(x, *p):
	n_comp = len(p) / 3
	amps = p[0::3]
	means = p[1::3]
	sigmas = p[2::3]
	y = np.zeros(len(x))
	for i in range(n_comp):
		y += amps[i] * np.exp(-0.5 * ((x - means[i]) / sigmas[i]) ** 2)
	return y

def normalize_dist(dist, delta_x):
	norm = np.sum(dist) * delta_x
	return dist / norm

def allocate_jobs(n_jobs, n_procs=1, rank=0):
	n_j_allocated = 0
	for i in range(n_procs):
		n_j_remain = n_jobs - n_j_allocated
		n_p_remain = n_procs - i
		n_j_to_allocate = n_j_remain / n_p_remain
		if rank == i:
			return range(n_j_allocated, \
						 n_j_allocated + n_j_to_allocate)
		n_j_allocated += n_j_to_allocate

def complete_array(target_distrib, use_mpi=False):
	if use_mpi:
		target = np.zeros(target_distrib.shape)
		mpi.COMM_WORLD.Allreduce(target_distrib, target, op=mpi.SUM)
	else:
		target = target_distrib
	return target

def d_prior(d, d_max, h_0=None):
	if h_0 is not None:
		return d ** 2 * (1.0 - 4.0 * h_0 * d / c) / \
			   d_max ** 3 / \
			   (1.0 / 3.0 - h_0 * d_max / c)
	else:
		return 3.0 * d ** 2 / d_max ** 3

def max_d_prior(d_max, h_0=None):
	if h_0 is not None:
		# THIS WAS A BUG!
		#d_prior_max = min(c / 6.0 / h_0_true, d_max)
		d_prior_max = min(c / 6.0 / h_0, d_max)
		return 1.0001 * d_prior(d_prior_max, d_max, h_0)
	else:
		return 3.0 / d_max

def rej_sample_d_prior(n_samples, d_max, p_max, h_0=None):
	accepted = np.zeros(n_samples)
	for i in range(0, n_samples):
		reject = True
		while reject:
			d_sample = npr.rand(1) * d_max
			p_sample = npr.rand(1) * p_max
			if d_prior(d_sample, d_max, h_0) > p_sample:
				reject = False
				accepted[i] = d_sample
	return accepted

def z_prior(z, z_max, q_0=None, inc_rate_redshift=False, safe=True):
	if inc_rate_redshift:
		if q_0 is not None:
			prior = z ** 2 * (1.0 - 2.0 * (1.0 + q_0) * z) / \
					(1.0 + z) / \
					(z_max ** 2 / 2.0 - z_max + np.log(1.0 + z_max) - \
					 2.0 * (1.0 + q_0) * \
					(z_max ** 3 / 3.0 - z_max ** 2 / 2.0 + \
					 z_max - np.log(1.0 + z_max)))
		else:
			prior = z ** 2 / (1.0 + z) / \
					(z_max ** 2 / 2.0 - z_max + np.log(1.0 + z_max))
	else:
		if q_0 is not None:
			prior = z ** 2 * (1.0 - 2.0 * (1.0 + q_0) * z) / \
					z_max ** 3 / \
					(1.0 / 3.0 - (1.0 + q_0) * z_max / 2.0)
		else:
			prior = 3.0 * z ** 2 / z_max ** 3
	if safe and np.any(prior < 0.0):
		print 'ERROR! -ve prior!'
		exit()
	return prior

def max_z_prior(z_max, q_0=None, inc_rate_redshift=False):
	if q_0 is not None:
		if q_0 > -1.0:
			if inc_rate_redshift:
				z_prior_max = (np.sqrt(36.0 * q_0 ** 2 + \
									   92.0 * q_0 + 57.0) -
							   5.0 - 6.0 * q_0) / 8.0 / \
							   (1.0 + q_0)
				z_prior_max = min(z_prior_max, z_max)
			else:
				z_prior_max = min(1.0 / 3.0 / (1 + q_0), z_max)
		else:
			z_prior_max = z_max
	else:
		z_prior_max = z_max
	return 1.0001 * z_prior(z_prior_max, z_max, q_0, inc_rate_redshift)

def rej_sample_z_prior(n_samples, z_max, p_max, q_0=None, \
					   inc_rate_redshift=False):
	accepted = np.zeros(n_samples)
	for i in range(0, n_samples):
		reject = True
		while reject:
			z_sample = npr.rand(1) * z_max
			p_sample = npr.rand(1) * p_max
			if z_prior(z_sample, z_max, q_0, inc_rate_redshift) > p_sample:
				reject = False
				accepted[i] = z_sample
	return accepted

def chirp_mass(m_1, m_2):
	return ((m_1 * m_2) ** 3 / (m_1 + m_2)) ** (0.2)

def merger_sim_d(h_0_true, sig_v_pec, amp_s, amp_n, sig_v_pec_obs, \
			   sig_z_obs, d_min, d_max, ntlo=False, q_0_true=None, \
			   p_max=None, vary_m_c=False, m_c_prior_mean=None, \
			   m_c_prior_std=None, sig_m_c_z_obs=None, \
			   cut_snr_true=False):

	# sample objects uniformly in volume
	cos_i_true = npr.uniform(-1.0, 1.0, 1)
	v_pec_true = npr.randn(1) * sig_v_pec
	if ntlo:
		d_true = rej_sample_d_prior(1, d_max, p_max, h_0_true)
		z_cos = h_0_true * d_true / c * \
				(1.0 - 0.5 * (1.0 - q_0_true) * \
				 h_0_true * d_true / c)
		z_true = z_cos + (1.0 + z_cos) * v_pec_true / c
	else:
		d_cum = npr.rand(1)
		d_true = (d_cum * d_max ** 3 + (1.0 - d_cum) * d_min ** 3) ** (1.0 / 3.0)
		z_true = (d_true * h_0_true + v_pec_true) / c
	if vary_m_c:
		m_c_true = m_c_prior_mean + \
				   npr.randn(1) * m_c_prior_std
		m_c_z_true = m_c_true * (1.0 + z_true)

	# convert parameters to amplitudes and SNRs
	if vary_m_c:
		amp_s = g * m_c_z_true / c ** 2
	amp_plus_true = amp_s * (1.0 + cos_i_true ** 2) / 2.0 / d_true
	amp_cross_true = -amp_s * cos_i_true / d_true
	amp_true = np.sqrt(amp_plus_true ** 2 + amp_cross_true ** 2)
	snr_true = amp_true / amp_n

	# sample noise fluctuations
	v_pec_obs = v_pec_true + npr.randn(1) * sig_v_pec_obs
	z_obs = z_true + npr.randn(1) * sig_z_obs
	if vary_m_c:
		m_c_z_obs = m_c_z_true + \
					npr.randn(1) * sig_m_c_z_obs
	amp_plus_obs = amp_plus_true + npr.randn(1) * amp_n
	amp_cross_obs = amp_cross_true + npr.randn(1) * amp_n
	amp_obs = np.sqrt(amp_plus_obs ** 2 + amp_cross_obs ** 2)
	if cut_snr_true:
		snr_obs = amp_true / amp_n
	else:
		snr_obs = amp_obs / amp_n

	# return
	to_return = cos_i_true, v_pec_true, d_true, z_true, \
				amp_plus_true, amp_cross_true, snr_true, v_pec_obs, \
				z_obs, amp_plus_obs, amp_cross_obs, snr_obs
	if vary_m_c:
		to_return += (m_c_true, m_c_z_true, m_c_z_obs)
	return to_return

def merger_sim(h_0_true, sig_v_pec, amp_s, amp_n, sig_v_pec_obs, \
			   sig_z_obs, z_max, ntlo=False, q_0_true=None, \
			   p_max=None, vary_m_c=False, m_c_prior_mean=None, \
			   m_c_prior_std=None, sig_m_c_z_obs=None, \
			   cut_snr_true=False, inc_rate_redshift=False):

	# sample objects uniformly in volume
	cos_i_true = npr.uniform(-1.0, 1.0, 1)
	v_pec_true = npr.randn(1) * sig_v_pec
	if ntlo:
		z_cos = rej_sample_z_prior(1, z_max, p_max, q_0_true, \
								   inc_rate_redshift)
		z_true = z_cos + (1.0 + z_cos) * v_pec_true / c
		d_true = c * z_cos / h_0_true * \
				 (1.0 + (1.0 - q_0_true) * z_cos / 2.0)
	else:
		if inc_rate_redshift:
			z_cos = rej_sample_z_prior(1, z_max, p_max, None, \
									   inc_rate_redshift)
		else:
			z_cos = z_max * np.cbrt(npr.rand(1))
		z_true = z_cos + v_pec_true / c
		d_true = c * z_cos / h_0_true
	if vary_m_c:
		m_c_true = m_c_prior_mean + \
				   npr.randn(1) * m_c_prior_std
		m_c_z_true = m_c_true * (1.0 + z_true)

	# convert parameters to amplitudes and SNRs
	if vary_m_c:
		amp_s = g * m_c_z_true / c ** 2
	amp_plus_true = amp_s * (1.0 + cos_i_true ** 2) / 2.0 / d_true
	amp_cross_true = -amp_s * cos_i_true / d_true
	amp_true = np.sqrt(amp_plus_true ** 2 + amp_cross_true ** 2)
	snr_true = amp_true / amp_n

	# sample noise fluctuations
	v_pec_obs = v_pec_true + npr.randn(1) * sig_v_pec_obs
	z_obs = z_true + npr.randn(1) * sig_z_obs
	if vary_m_c:
		m_c_z_obs = m_c_z_true + \
					npr.randn(1) * sig_m_c_z_obs
	amp_plus_obs = amp_plus_true + npr.randn(1) * amp_n
	amp_cross_obs = amp_cross_true + npr.randn(1) * amp_n
	amp_obs = np.sqrt(amp_plus_obs ** 2 + amp_cross_obs ** 2)
	if cut_snr_true:
		snr_obs = amp_true / amp_n
	else:
		snr_obs = amp_obs / amp_n

	# return
	to_return = cos_i_true, v_pec_true, d_true, z_true, \
				amp_plus_true, amp_cross_true, snr_true, v_pec_obs, \
				z_obs, amp_plus_obs, amp_cross_obs, snr_obs
	if vary_m_c:
		to_return += (m_c_true, m_c_z_true, m_c_z_obs)
	return to_return

def poly_surf(coords, *pars):
	if coords.ndim == 1:
		return pars[0] + \
			   pars[1] * coords + \
			   pars[2] * coords ** 2 + \
			   pars[3] * coords ** 3 + \
			   pars[4] * coords ** 4
	else:
		return pars[0] + \
			   pars[1] * coords[0, :] + \
			   pars[2] * coords[1, :] + \
			   pars[3] * coords[0, :] ** 2 + \
			   pars[4] * coords[1, :] ** 2 + \
			   pars[5] * coords[0, :] * coords[1, :] + \
			   pars[6] * coords[0, :] ** 3 + \
			   pars[7] * coords[1, :] ** 3 + \
			   pars[8] * coords[1, :] * coords[0, :] ** 2 + \
			   pars[9] * coords[0, :] * coords[1, :] ** 2 + \
			   pars[10] * coords[0, :] ** 4 + \
			   pars[11] * coords[1, :] ** 4 + \
			   pars[12] * coords[1, :] * coords[0, :] ** 3 + \
			   pars[13] * coords[0, :] * coords[1, :] ** 3 + \
			   pars[14] * (coords[0, :] * coords[1, :]) ** 2

def h_0_max_post_wrapper(h_0, *gd_density):
	return -gd_density[0].Prob(h_0)

def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

# plotting settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')
cols = [cm(x) for x in np.linspace(0.1, 0.9, 10)]

# settings
h_0_true = 70.0
c = 2.998e5
single_z = False
analytic = False

# toy model for pop @ single redshift or not?
if single_z:

	# settings
	n_event = 2
	sig_d = 20.0
	sig_v = 200.0
	d_true = 100.0
	z_true = h_0_true * d_true / c
	h_0_grid = np.linspace(1.0, 200.0, 1000)
	delta_h_0 = h_0_grid[1] - h_0_grid[0]
	#h_0_grid = np.linspace(68.0, 72.0, 1000)

	n_event = 51
	n_sample = 30
	for j in range(n_sample):
		d_obs = npr.randn(n_event) * sig_d + d_true
		v_obs = npr.randn(n_event) * sig_v
		z_obs = (h_0_true * d_true) / c
		ln_prob = lprob_sample(h_0_grid, d_obs, z_obs, v_obs, \
							   sig_v ** 2, sig_d ** 2)
		prob = np.exp(ln_prob - np.max(ln_prob))
		prob = normalize_dist(prob, delta_h_0)
		mp.plot(h_0_grid, prob)
		ln_prob = lprob_sample_alt(h_0_grid, d_obs, z_obs, v_obs, \
							   sig_v, sig_d)
		prob = np.exp(ln_prob - np.max(ln_prob))
		prob = normalize_dist(prob, delta_h_0)
		mp.plot(h_0_grid, prob, 'k--')
	mp.axvline(h_0_true)
	mp.show()

	# loop over n_events
	n_events = np.array([1, 3, 10, 30, 100, 300, 1000])
	n_event_cols = np.linspace(0.0, 0.9, len(n_events))
	fig, axes = mp.subplots(1, 2, figsize=(12, 6), sharey=True)
	d_obs = npr.randn(np.max(n_events)) * sig_d + d_true
	v_obs = npr.randn(np.max(n_events)) * sig_v
	z_obs = (h_0_true * d_true) / c
	for n in range(len(n_events)):
		ln_prob = lprob_sample(h_0_grid, d_obs[0:n_events[n]], \
							   z_obs, v_obs[0:n_events[n]], \
							   sig_v ** 2, sig_d ** 2)
		prob = np.exp(ln_prob - np.max(ln_prob))
		prob = normalize_dist(prob, delta_h_0)
		axes[0].plot(h_0_grid, prob, color=cm(n_event_cols[n]), \
					 label='{:d} events'.format(n_events[n]))
		ln_prob = lprob_sample_no_n(h_0_grid, d_obs[0:n_events[n]], \
									z_obs, v_obs[0:n_events[n]], \
									sig_v ** 2, sig_d ** 2)
		prob = np.exp(ln_prob - np.max(ln_prob))
		prob = normalize_dist(prob, delta_h_0)
		axes[1].plot(h_0_grid, prob, color=cm(n_event_cols[n]), \
					 label='{:d} events'.format(n_events[n]))
	axes[0].axvline(h_0_true, color='k', ls='--', label=r'True $H_0$')
	axes[1].axvline(h_0_true, color='k', ls='--', label=r'True $H_0$')
	axes[0].legend(loc='upper right')
	axes[1].legend(loc='upper right')
	axes[0].set_xlabel(r'$H_0$')
	axes[1].set_xlabel(r'$H_0$')
	axes[0].set_ylabel(r'${\rm Pr}(H_0|d)$')
	axes[0].set_title(r'$H_0^{-n}$ Included')
	axes[1].set_title(r'$H_0^{-n}$ Discarded')
	axes[0].set_xlim(50.0, 90.0)
	axes[1].set_xlim(50.0, 90.0)
	xticks = axes[1].xaxis.get_major_ticks()
	xticks[0].label1.set_visible(False)
	fig.subplots_adjust(hspace=0, wspace=0)
	mp.savefig('idealized_posteriors_bias_vs_n_events.pdf', \
			   bbox_inches = 'tight')
	mp.show()
	exit()

elif analytic:

	# Daniel's analytic model
	n_event = 100
	n_grid = 500
	d_max = np.linspace(0.0, 300.0, n_grid)
	d_err_frac = 0.1
	sig_z = 0.0005
	sig_v = 200.0
	cols = [cm(0.2), cm(0.7)]
	delta_h_0_sel = 0.0

	d_0 = np.sqrt(c ** 2 * sig_z ** 2 + sig_v ** 2) / \
		  (h_0_true * d_err_frac)
	sig_h_0 = np.sqrt(3.0 / 5.0 / float(n_event)) * h_0_true * \
			  d_err_frac * np.sqrt(5.0 * (d_0 / d_max) ** 2 + 1.0)
	delta_h_0 = -1.0 * d_max / 100.0
	print d_0
	mp.plot(d_max, delta_h_0_sel + sig_h_0, color=cols[0])
	mp.plot(d_max, delta_h_0_sel - sig_h_0, color=cols[0])
	mp.plot(d_max, delta_h_0_sel * np.ones(n_grid), color=cols[0], ls='--')
	mp.plot(d_max, delta_h_0_sel + delta_h_0 + sig_h_0, color=cols[1])
	mp.plot(d_max, delta_h_0_sel + delta_h_0 - sig_h_0, color=cols[1])
	mp.plot(d_max, delta_h_0_sel + delta_h_0, color=cols[1], ls='--')
	mp.ylim(-3.0, 3.0)
	mp.gca().tick_params(axis='both', which='major', labelsize=14)
	mp.xlabel(r'${\rm maximum\,survey\,distance,}\,D_*\,' + \
			  r'{\rm(Mpc)}$', fontsize=16)
	mp.ylabel(r'${\rm error/bias\,in}\,H_0,\,{\rm(km/s/Mpc)}$', \
			  fontsize=16)
	mp.show()
	exit()


else:

	# settings for toy problem
	n_event = 100 # 100
	n_rpt = 1#100
	d_min = 0.0
	d_max = 1000.0 # 500.0
	z_max = 0.18 # 0.235
	amp_s = 3000.0
	amp_n =  1.0
	snr_lim = 12.0
	c = 2.998e5  # km / s
	g = 4.301e-9 # km^2 Mpc / M_sol / s^2
	h_0_true = 70.0
	q_0_true = -0.5
	if n_event < 10:
		h_0_min = 20.0
		h_0_max = 200.0
	elif n_event < 100:
		h_0_min = 50.0
		h_0_max = 100.0
	elif n_event < 5000:
		h_0_min = 60.0
		h_0_max = 80.0
	else:
		h_0_min = 60.0
		h_0_max = 72.0
	n_event_str = '{:d}_det_events'.format(n_event)
	n_event_det = n_event
	cut_snr_true = False
	zero_noise_like = False
	stub = 'sel_eff_'
	if cut_snr_true:
		stub += 'true_snr_cut_'
	if zero_noise_like:
		stub += 'zero_noise_like_'
	plot_events = False
	use_mpi = True
	constrain = True
	sample = True
	sample_d = False
	n_samples = 1000
	n_chains = 4
	recompile = False
	stan_constrain = False
	sig_v_pec = 500.0 # LVC take U(-1000,1000)
	sig_v_pec_obs = 200.0 # 185.0 # 150.0
	sig_z_obs = 0.001 # 75.0 / c
	plot_cos_i = False
	fixed_n_bns = False
	ntlo = True
	vary_m_c = True
	inc_rate_redshift = True
	save_distances = False
	plot_dist_cos_i_posts = True
	sample_dist_cos_i_posts = False
	find_n_bar = False
	m_i_mean = 1.4
	m_i_std = 0.2
	base = 'bias_test'
	pars = ['h_0']
	par_names = ['H_0']
	par_ranges = {}
	if ntlo:
		base += '_hq'
		pars += ['q_0']
		par_names += ['q_0']
		d_max = c * z_max / h_0_true * \
				(1.0 + 0.5 * (1.0 - q_0_true) * z_max)
	else:
		d_max = c * z_max / h_0_true
	n_pars = len(pars)
	if vary_m_c:
		base += '_vary_m_c'
	if inc_rate_redshift:
		base += '_rr'
	if save_distances:
		pars += ['true_d']
		for i in range(n_event):
			par_names += ['d_{:d}'.format(i)]
		n_pars += n_event

	# MPI fun
	if use_mpi:
		import mpi4py.MPI as mpi
		n_procs = mpi.COMM_WORLD.Get_size()
		rank = mpi.COMM_WORLD.Get_rank()
	else:
		n_procs = 1
		rank = 0

	# test case with fixed seed
	if constrain:
		seed = 102314 + rank
	else:
		seed = np.random.randint(102314, 221216, 1)[0]
	npr.seed(seed)

	# useful calculation
	snr_min = amp_s / d_max / amp_n / 2.0
	if rank == 0:
		print 'minimum snr at max distance [cos(i)=0]:', snr_min
		print 'minimum snr at max distance [cos(i)=1]:', \
			  '{:.1f}'.format(snr_min * np.sqrt(8.0))

	# convert priors on individual masses to prior on chirp mass; also
	# adjust noise level to ensure detectable fraction is similar to 
	# case in which amplitude is held fixed
	if vary_m_c:
		n_sample_test = 100000
		m_1 = m_i_mean + npr.randn(n_sample_test) * m_i_std
		m_2 = m_i_mean + npr.randn(n_sample_test) * m_i_std
		m_c = chirp_mass(m_1, m_2)
		hist, bin_edges = np.histogram(m_c, bins=50, density=True)
		bins = np.zeros(len(bin_edges) - 1)
		for i in range(len(bin_edges) - 1):
			bins[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])
		m_c_prior_mean = np.sum(bins * hist) / np.sum(hist)
		m_c_prior_std = np.sqrt(np.sum((bins - m_c_prior_mean) ** 2 * \
									   hist) / np.sum(hist))
		amp_n = g / c ** 2 / amp_s * m_c_prior_mean
		sig_m_c_z_obs = m_c_prior_mean * 0.01
		plot_chirp_mass_prior = False
		if plot_chirp_mass_prior:
			mp.hist(m_1, normed=True, alpha=0.5, bins=50)
			mp.hist(m_2, normed=True, alpha=0.5, bins=50)
			mp.hist(m_c, normed=True, alpha=0.5, bins=50)
			x = np.linspace(0.0, 2.0, 1000)
			mp.plot(x, np.exp(-0.5 * ((x - m_c_prior_mean) / \
									  m_c_prior_std) ** 2) / \
					   np.sqrt(2.0 * np.pi * m_c_prior_std ** 2), 'k')
			mp.show()
			exit()
	else:
		m_c_prior_mean = None
		m_c_prior_std = None
		sig_m_c_z_obs = None

	# fit the parameter dependence of the number of detectable sources
	if find_n_bar:

		# set out grid of parameters
		n_grid = 10
		h_0_min, h_0_max = 60.0, 80.0
		q_0_min, q_0_max = -2.0, 1.0
		h_0_grid = np.linspace(h_0_min, h_0_max, n_grid)
		q_0_grid = np.linspace(q_0_min, q_0_max, n_grid)
		rate = 2.0e-4 # 2.0e-5
		time = 1.0
		if ntlo:
			n_bar_det = np.zeros((n_grid, n_grid))
			z_max_det = np.zeros((n_grid, n_grid))
		else:
			n_bar_det = np.zeros((n_grid, 1))
			z_max_det = np.zeros((n_grid, 1))

		# loop over jobs
		if ntlo:
			n_jobs = n_grid ** 2
		else:
			n_jobs = n_grid
		job_list = allocate_jobs(n_jobs, n_procs, rank)
		print 'process {:d} jobs: '.format(rank), job_list
		for ij in job_list:

			# extract indices
			if ntlo:
				i = ij / n_grid
				j = ij - i * n_grid
			else:
				i = ij
				j = 0

			# find expected total number of events from rate, 
			# observing time and integral of dV/dz * 1/(1+z) [to 
			# account for redshifting of otherwise constant 
			# merger rate] out to maximum redshift z_max
			p_max = max_z_prior(z_max, q_0_grid[j], inc_rate_redshift)
			if inc_rate_redshift:
				z_integral = z_max ** 2 / 2.0 - z_max + np.log(1 + z_max)
				if ntlo:
					z_integral -= 2.0 * (1.0 + q_0_grid[j]) * \
								  (z_max ** 3 / 3.0 - z_max ** 2 / 2.0 + \
								   z_max - np.log(1 + z_max))
			else:
				z_integral = z_max ** 3 / 3.0
				if ntlo:
					z_integral -= (1.0 + q_0_grid[j]) * z_max ** 4 / 2.0
			z_integral *= 4.0 * np.pi * (c / h_0_grid[i]) ** 3
			n_bar_tot = z_integral * rate * time
			check = False
			if check:
				if not inc_rate_redshift:
					d_max = z_max
					if ntlo:
						d_max += (1.0 - q_0_true) * z_max / 2.0
					d_max *= c / h_0_true
					d_integral = d_max ** 3 / 3.0
					if ntlo:
						d_integral -= h_0_grid[i] * d_max / c
					d_integral *= 4.0 * np.pi
					print z_max, d_max, d_integral / z_integral
					exit()
			'''
			if ntlo:
				n_bar_det[i, j] = n_bar_tot
			else:
				n_bar_det[i] = n_bar_tot
			'''
			# calculate detectable n_bar
			for k in range(int(n_bar_tot)):
				sim = merger_sim(h_0_grid[i], sig_v_pec, \
								 amp_s, amp_n, sig_v_pec_obs, \
								 sig_z_obs, z_max, ntlo, \
								 q_0_grid[j], p_max, \
								 vary_m_c, m_c_prior_mean, \
								 m_c_prior_std, sig_m_c_z_obs, \
								 cut_snr_true)
				if sim[11] > snr_lim:
					n_bar_det[i, j] += 1
					if sim[3] > z_max_det[i, j]:
						z_max_det[i, j] = sim[3]
			
		# accumulate
		n_bar_det = complete_array(n_bar_det, use_mpi)
		z_max_det = complete_array(z_max_det, use_mpi)

		# fit n_bar_det as function of sampled parameters with a 2D
		# polynomial. need to feed curve_fit coordinates in 2 * 
		# n_grid ** 2 array and values to fit in n_grid ** 2 array
		if ntlo:
			coords = np.zeros((2, n_grid ** 2))
			counts = np.zeros(n_grid ** 2)
			for i in range(n_grid):
				for j in range(n_grid):
					coords[0, i * n_grid + j] = h_0_grid[i]
					coords[1, i * n_grid + j] = q_0_grid[j]
					counts[i * n_grid + j] = n_bar_det[i, j]
			popt, pcov = so.curve_fit(poly_surf, coords, counts, \
									  p0=np.zeros(15))
			n_bar_fit = np.zeros((n_grid, n_grid))
			for i in range(n_grid):
				for j in range(n_grid):
					coord = np.zeros((2, 1))
					coord[0, 0] = h_0_grid[i]
					coord[1, 0] = q_0_grid[j]
					n_bar_fit[i, j] = poly_surf(coord, *popt)
			results = np.column_stack([coords[0, :], coords[1, :], counts])
		else:
			n_bar_det = n_bar_det[:, 0]
			popt, pcov = so.curve_fit(poly_surf, h_0_grid, n_bar_det, \
									  p0=np.zeros(5))
			n_bar_fit = poly_surf(h_0_grid, *popt)
			results = np.column_stack([h_0_grid, n_bar_det])
		if rank == 0:
			np.savetxt(base + '_n_bar_det.txt', results)
			np.savetxt(base + '_n_bar_det_fit.txt', popt)

		# plot results
		if rank == 0:

			# fit quality
			if ntlo:
				fig, axes = mp.subplots(1, 3, figsize=(15, 5))
				im0 = axes[0].imshow(n_bar_det.T, \
							   extent=[h_0_min, h_0_max, q_0_min, q_0_max], \
							   interpolation='nearest', cmap='plasma', \
							   vmin=np.min(n_bar_det), vmax=np.max(n_bar_det))
				axes[1].imshow(n_bar_fit.T, \
							   extent=[h_0_min, h_0_max, q_0_min, q_0_max], \
							   interpolation='nearest', cmap='plasma', \
							   vmin=np.min(n_bar_det), vmax=np.max(n_bar_det))
				im2 = axes[2].imshow((n_bar_det.T - n_bar_fit.T) / n_bar_det.T, \
							   extent=[h_0_min, h_0_max, q_0_min, q_0_max], \
							   interpolation='nearest', cmap='plasma')
				fig.subplots_adjust(right=0.9)
				fig.subplots_adjust(left=0.1)
				cbar0_ax = fig.add_axes([0.05, 0.15, 0.025, 0.7])
				cbar2_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
				fig.colorbar(im0, cax=cbar0_ax)
				fig.colorbar(im2, cax=cbar2_ax)
				cbar0_ax.yaxis.set_ticks_position('left')
				axes[0].set_title('Input')
				axes[1].set_title('Fit')
				axes[2].set_title('Fractional Difference')
				for i in range(3):
					axes[i].set_aspect((h_0_max - h_0_min) / \
									   (q_0_max - q_0_min))
					axes[i].set_xlabel(r'$H_0$')
					axes[i].set_ylabel(r'$q_0$')
			else:
				fig, axes = mp.subplots(1, 2, figsize=(16, 5))
				axes[0].plot(h_0_grid, n_bar_det, color=cm(0.3))
				axes[0].plot(h_0_grid, n_bar_fit, color=cm(0.6), ls='--')
				axes[1].plot(h_0_grid, (n_bar_det - n_bar_fit) / \
									   n_bar_det, color=cm(0.3))
				axes[0].set_title('Input and Fit')
				axes[1].set_title('Fractional Difference')
				for i in range(2):
					axes[i].set_xlabel(r'$H_0$')
					axes[i].set_ylabel(r'$\bar{N}_{\rm det}$')
			mp.savefig(base + '_n_bar_det_fit.pdf', bbox_inches='tight')
			mp.close()

			# safety checks for selection effects
			fig, ax = mp.subplots(figsize=(8, 6))
			if ntlo:
				im = ax.imshow(z_max_det.T, \
							   extent=[h_0_min, h_0_max, q_0_min, q_0_max], \
							   interpolation='nearest', cmap='plasma')
				fig.subplots_adjust(right=0.9)
				cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
				fig.colorbar(im, cax=cbar_ax)
				ax.set_aspect((h_0_max - h_0_min) / (q_0_max - q_0_min))
				ax.set_xlabel(r'$H_0$')
				ax.set_ylabel(r'$q_0$')
			else:
				ax.plot(h_0_grid, z_max_det, color=cm(0.5))
				ax.set_xlabel(r'$H_0$')
				ax.set_ylabel(r'${\rm max}(z[det])$')
			ax.set_title('Maximum Detected Redshift')
			mp.savefig(base + '_z_max_det.pdf', bbox_inches='tight')
			mp.close()

	else:

		# read in n_bar_det fit
		if rank == 0:
			popt = np.genfromtxt(base + '_n_bar_det_fit.txt')
		else:
			if ntlo:
				popt = np.zeros(15)
			else:
				popt = np.zeros(5)
	if use_mpi:
		mpi.COMM_WORLD.Bcast(popt, 0)

	# ensure n_bar_det is correctly renormalized to account for 
	# different rates and sample sizes: we want the number of 
	# expected detections to match the sample size at the fiducial
	# cosmology
	if ntlo:
		coord = np.zeros((2, 1))
		coord[:, 0] = [h_0_true, q_0_true]
		renorm = poly_surf(coord, *popt)
	else:
		renorm = poly_surf(np.array([h_0_true]), *popt)
	popt *= float(n_event) / renorm

	# sample or read in samples
	if sample:

		# iterate over realisations
		samples = np.zeros((n_samples / 2, n_chains, n_pars + 1, n_rpt))
		if save_distances:
			d_trues = np.zeros((n_event, n_rpt))
		job_list = allocate_jobs(n_rpt, n_procs, rank)
		print 'process {:d} jobs: '.format(rank), job_list
		for i in job_list:

			print 'process {:d}, repeat {:d}'.format(rank, i)

			# placeholder lists
			cos_i_true = []
			d_true = []
			v_pec_true = []
			z_true = []
			if vary_m_c:
				m_c_true = []
				m_c_z_true = []
			amp_plus_true = []
			amp_cross_true = []
			v_pec_obs = []
			z_obs = []
			if vary_m_c:
				m_c_z_obs = []
			amp_plus_obs = []
			amp_cross_obs = []
			snr_true = []
			snr_obs = []
			if not ntlo and not inc_rate_redshift:
				p_max = None
			else:
				p_max = max_z_prior(z_max, q_0_true, inc_rate_redshift)

			# ensure count is correct
			n_det = 0
			while n_det < n_event_det:
				
				# simulate data
				sim = merger_sim(h_0_true, sig_v_pec, amp_s, amp_n, \
								 sig_v_pec_obs, sig_z_obs, z_max, \
								 ntlo, q_0_true, p_max, vary_m_c, \
								 m_c_prior_mean, m_c_prior_std, \
								 sig_m_c_z_obs, cut_snr_true)
				cos_i_true.append(sim[0])
				v_pec_true.append(sim[1])
				d_true.append(sim[2])
				z_true.append(sim[3])
				amp_plus_true.append(sim[4])
				amp_cross_true.append(sim[5])
				snr_true.append(sim[6])
				v_pec_obs.append(sim[7])
				z_obs.append(sim[8])
				amp_plus_obs.append(sim[9])
				amp_cross_obs.append(sim[10])
				snr_obs.append(sim[11])
				if vary_m_c:
					m_c_true.append(sim[12])
					m_c_z_true.append(sim[13])
					m_c_z_obs.append(sim[14])
				
				# check for detection
				if snr_obs[-1] > snr_lim:
					n_det += 1

			# convert placeholder lists
			n_event = len(cos_i_true)
			cos_i_true = np.array(cos_i_true)
			d_true = np.array(d_true)
			v_pec_true = np.array(v_pec_true)
			z_true = np.array(z_true)
			if vary_m_c:
				m_c_true = np.array(m_c_true)
				m_c_z_true = np.array(m_c_z_true)
			amp_plus_true = np.array(amp_plus_true)
			amp_cross_true = np.array(amp_cross_true)
			v_pec_obs = np.array(v_pec_obs)
			z_obs = np.array(z_obs)
			if vary_m_c:
				m_c_z_obs = np.array(m_c_z_obs)
			amp_plus_obs = np.array(amp_plus_obs)
			amp_cross_obs = np.array(amp_cross_obs)
			snr_obs = np.array(snr_obs)
			if ntlo:
				z_cos_true = (z_true - v_pec_true / c) / \
							 (1.0 + v_pec_true / c)
			else:
				z_cos_true = z_true - v_pec_true / c

			# sanity checks
			print('simulation sanity checks!')
			std_check_txt = '{:s}: est = {:9.3e}; exp = {:9.3e}'
			print(std_check_txt.format('v_pec_true', np.std(v_pec_true), \
									   sig_v_pec))
			if vary_m_c:
				print(std_check_txt.format('m_c_true', np.std(m_c_true), \
										   m_c_prior_std))
			print(std_check_txt.format('v_pec_obs', \
									   np.std(v_pec_obs - v_pec_true), \
									   sig_v_pec_obs))
			print(std_check_txt.format('z_obs', np.std(z_obs - z_true), \
									   sig_z_obs))
			if vary_m_c:
				print(std_check_txt.format('m_c_z_obs', \
										   np.std(m_c_z_obs - m_c_z_true), \
										   sig_m_c_z_obs))
			
			# cut sample
			det = snr_obs > snr_lim
			if n_rpt == 1:
				print '{:d}/{:d} objects detected'.format(np.sum(det), n_event)
				fig, axes = mp.subplots(1, 2, figsize=(12, 5))
				bins = np.linspace(0.0, z_max, 50)
				axes[0].hist(z_cos_true, bins=bins, normed=True, fc=cols[2], \
							 ec=cols[2], alpha=0.9, label='all (samples)')
				axes[0].hist(z_cos_true[det], bins=bins, normed=True, fc=cols[8], \
							 ec=cols[8], alpha=0.7, label='detected')
				axes[0].plot(bins, z_prior(bins, z_max, q_0_true, \
										   inc_rate_redshift), 'k--', \
							 label='all (analytic)')
				axes[0].set_xlabel(r'$z_{\rm cos}$')
				axes[0].set_ylabel(r'${\rm Pr}(z_{\rm cos})$')
				axes[0].set_xlim(0.0, z_max)
				axes[0].legend(loc='upper right')
				bins = np.linspace(-1, 1, 50)
				axes[1].hist(cos_i_true, bins=bins, normed=True, fc=cols[2], \
							 ec=cols[2], alpha=0.9)
				axes[1].hist(cos_i_true[det], bins=bins, normed=True, fc=cols[8], \
							 ec=cols[8], alpha=0.7)
				axes[1].plot(bins, np.ones(len(bins)) * 0.5, 'k--')
				axes[1].set_xlabel(r'$\cos\,\i$')
				axes[1].set_ylabel(r'${\rm Pr}(\cos\,\i)$')
				mp.savefig(base + '_prior_samples_' + n_event_str + '.pdf', \
						   bbox_inches='tight')
				mp.close(fig)
				fig, axes = mp.subplots(1, 2, figsize=(12, 5))
				axes[0].scatter(z_cos_true, snr_obs, color=cols[2], marker='.', \
								label='rejected')
				axes[0].scatter(z_cos_true[det], snr_obs[det], color=cols[8], \
								marker='.', label='selected')
				axes[0].axhline(snr_lim, color='k', ls='--')
				axes[0].set_xlabel(r'${\rm cosmological\,redshift},\,z_{\rm cos}$', fontsize=18)
				axes[0].set_ylabel(r'${\rm signal\,to\,noise},\,\hat{\rho}$', fontsize=18)
				axes[0].set_xlim(0.0, z_max)
				axes[0].set_ylim(0.0, np.max(snr_obs))
				axes[0].legend(loc='upper right', fontsize=16)
				axes[0].tick_params(axis='both', which='major', labelsize=16)
				if plot_cos_i:
					axes[1].scatter(cos_i_true, snr_obs, color=cols[2], \
									marker='.', label='rejected')
					axes[1].scatter(cos_i_true[det], snr_obs[det], \
									color=cols[8], marker='.', \
									label='selected')
				else:
					axes[1].scatter(np.arccos(cos_i_true) * 180.0 / np.pi, \
									snr_obs, color=cols[2], \
									marker='.', label='rejected')
					axes[1].scatter(np.arccos(cos_i_true[det]) * \
									180.0 / np.pi, snr_obs[det], \
									color=cols[8], marker='.', \
									label='selected')
				axes[1].axhline(snr_lim, color='k', ls='--')
				if plot_cos_i:
					axes[1].set_xlabel(r'$\cos\,\i$', fontsize=18)
					axes[1].set_xlim(-1.0, 1.0)
				else:
					axes[1].set_xlabel(r'${\rm inclination},\,\iota\,(\degree)$', \
									   fontsize=18)
					axes[1].set_xlim(0.0, 180.0)
					axes[1].set_xticks(np.linspace(0.0, 180.0, 7))
				axes[1].set_ylabel(r'${\rm signal\,to\,noise},\,\hat{\rho}$', fontsize=18)
				axes[1].set_ylim(0.0, np.max(snr_obs))
				axes[1].tick_params(axis='both', which='major', labelsize=16)
				mp.savefig(base + '_par_vs_snr_' + n_event_str + '.pdf', \
						   bbox_inches='tight')
				mp.close(fig)
				fig, axes = mp.subplots(1, 2, figsize=(12, 5))
				axes[0].scatter(d_true, snr_obs, color=cols[8], marker='.', \
								label='rejected')
				axes[0].scatter(d_true[det], snr_obs[det], color=cols[2], \
								marker='.', label='selected')
				axes[0].axhline(snr_lim, color='k', ls='--')
				axes[0].set_xlabel(r'${\rm distance},\,D\,({\rm Mpc})$', fontsize=18)
				axes[0].set_ylabel(r'${\rm signal\,to\,noise},\,\hat{\rho}$', fontsize=18)
				axes[0].set_xlim(0.0, d_max)
				axes[0].set_ylim(0.0, np.max(snr_obs))
				axes[0].legend(loc='upper right', fontsize=16)
				axes[0].tick_params(axis='both', which='major', labelsize=16)
				if plot_cos_i:
					axes[1].scatter(cos_i_true, snr_obs, color=cols[8], \
									marker='.', label='rejected')
					axes[1].scatter(cos_i_true[det], snr_obs[det], \
									color=cols[2], marker='.', \
									label='selected')
				else:
					axes[1].scatter(np.arccos(cos_i_true) * 180.0 / np.pi, \
									snr_obs, color=cols[8], \
									marker='.', label='rejected')
					axes[1].scatter(np.arccos(cos_i_true[det]) * \
									180.0 / np.pi, snr_obs[det], \
									color=cols[2], marker='.', \
									label='selected')
				axes[1].axhline(snr_lim, color='k', ls='--')
				if plot_cos_i:
					axes[1].set_xlabel(r'$\cos\,\i$', fontsize=18)
					axes[1].set_xlim(-1.0, 1.0)
				else:
					axes[1].set_xlabel(r'${\rm inclination},\,\iota\,(\degree)$', \
									   fontsize=18)
					axes[1].set_xlim(0.0, 180.0)
					axes[1].set_xticks(np.linspace(0.0, 180.0, 7))
				axes[1].set_ylabel(r'${\rm signal\,to\,noise},\,\hat{\rho}$', fontsize=18)
				axes[1].set_ylim(0.0, np.max(snr_obs))
				axes[1].tick_params(axis='both', which='major', labelsize=16)
				mp.savefig(base + '_par_vs_snr_dist_' + n_event_str + '.pdf', \
						   bbox_inches='tight')
				mp.close(fig)

				# optionally generate distance-cos i posteriors considering 
				# only GW data using Stan
				if plot_dist_cos_i_posts:

					if sample_dist_cos_i_posts:

						# compile/load Stan model
						if recompile:
							stan_model = ps.StanModel('gw_d_l_cos_i.stan')
							with open('gw_d_l_cos_i_model.pkl', 'wb') as f:
								pickle.dump(stan_model, f)
						else:
							try:
								with open('gw_d_l_cos_i_model.pkl', 'rb') as f:
									stan_model = pickle.load(f)
							except EnvironmentError:
								print 'ERROR: pickled Stan model ' + \
									  '(gw_d_l_cos_i_model.pkl) ' + \
									  'not found. Please set recompile = True'
								exit()

						# set up stan inputs and sample
						stan_data = {'ntlo': int(ntlo), \
									 'vary_m_c': int(vary_m_c), \
									 'n_bns': n_event, \
									 'obs_amp_plus': amp_plus_obs[:, 0], \
									 'obs_amp_cross': amp_cross_obs[:, 0], \
									 'amp_n': amp_n, 'z_max': z_max, \
									 'd_max': d_max}
						if vary_m_c:
							stan_data['mu_m_c'] = m_c_prior_mean
							stan_data['sig_m_c'] = m_c_prior_std
							stan_data['amp_s'] = 0.0
							stan_data['obs_m_c_z'] = m_c_z_obs[:, 0]
							stan_data['sig_obs_m_c_z'] = sig_m_c_z_obs
						else:
							stan_data['mu_m_c'] = 0.0
							stan_data['sig_m_c'] = 0.0
							stan_data['amp_s'] = amp_s
							stan_data['obs_m_c_z'] = np.zeros(n_event)
							stan_data['sig_obs_m_c_z'] = 0.0
						if save_distances:
							if ntlo:
								stan_pars = pars[2:]
							else:
								stan_pars = pars[1:]
						else:
							stan_pars = ['true_d']
						stan_pars += ['true_cos_i']
						if stan_constrain:
							stan_seed = 23102014
						else:
							stan_seed = None
						fit = stan_model.sampling(data = stan_data, \
												  iter = n_samples, \
												  chains = n_chains, \
												  seed = stan_seed, \
												  pars = stan_pars)
						print fit

						# save samples and truths
						gw_samples = fit.extract(permuted = False, \
												 inc_warmup = False)
						with h5py.File(base + '_d_cos_i_samples_' + \
									   '{:d}'.format(n_rpt) + \
									   '.h5', 'w') as f:
							f.create_dataset('gw_samples', data=gw_samples)
						with h5py.File(base + '_d_cos_i_truths_' + \
									   '{:d}'.format(n_rpt) + \
									   '.h5', 'w') as f:
							f.create_dataset('d_true', data=d_true)
							f.create_dataset('cos_i_true', data=cos_i_true)
							f.create_dataset('det', data=det)

					# read samples and ground truths
					with h5py.File(base + '_d_cos_i_samples_' + \
								   '{:d}'.format(n_rpt) + \
								   '.h5', 'r') as f:
						raw_samples = f['gw_samples'][:]
					with h5py.File(base + '_d_cos_i_truths_' + \
								   '{:d}'.format(n_rpt) + \
								   '.h5', 'r') as f:
						d_true = f['d_true'][:][:, 0]
						cos_i_true = f['cos_i_true'][:][:, 0]
						det = f['det'][:][:, 0]
					n_event = len(d_true)
					n_pars = raw_samples.shape[2] - 1
					n_chains = raw_samples.shape[1]
					n_samples = raw_samples.shape[0]
					gw_samples = np.zeros((n_chains * n_samples, n_pars))
					for k in range(0, n_chains):
						for j in range(0, n_pars):
							gw_samples[k * n_samples: (k + 1) * n_samples, j] = \
								raw_samples[:, k, j]

					# plot inferred distance and cosine-inclination biases
					fig, ax = mp.subplots(figsize=(6, 6))
					gw_d_mean = np.mean(gw_samples, axis=0)[0: n_pars / 2]
					gw_d_var = np.var(gw_samples, axis=0)[0: n_pars / 2]
					delta_d = gw_d_mean - d_true
					'''
					d_true_det = d_true[det]
					delta_d_det = delta_d[det]
					ijk = np.argsort(d_true_det)
					mp.plot(d_true_det[ijk], delta_d_det[ijk])
					mp.show()
					'''
					no_sel_eff_bias = np.mean(delta_d[det] / d_true[det]) * \
									  -10.0 / 0.15
					#far = d_true[det] > 250.0
					#no_sel_eff_bias = np.mean(delta_d[det][far] / d_true[det][far]) * \
					#				  -10.0 / 0.15
					print 'expected bias neglecting sel eff ' + \
						  '{:7.3f} km/s/Mpc'.format(no_sel_eff_bias)

					x = np.zeros(n_det)
					v = np.zeros(n_det)
					temp = gw_samples[:, 0: n_pars / 2]
					temp = temp[:, det]
					delta = np.zeros((n_samples * n_chains, n_det))
					for k in range(n_det):
						delta[:, k] = -10.0 * (temp[:, k] - d_true[det][k]) / \
									  d_true[det][k] / 0.15
					far = d_true[det] > 250.0
					far = (d_true[det] > 200.0) & (d_true[det] < 300.0)
					x = np.mean(delta[:, far], axis=0)
					v = np.var(delta[:, far], axis=0)
					x_hat = np.sum(x / v) / np.sum(1.0 / v) * \
							float(np.sum(far)) / float(n_det)
					print 'expected bias neglecting sel eff ' + \
						  '{:7.3f} km/s/Mpc'.format(x_hat)
					'''
					gw_d_max = np.zeros(n_det)
					ijk = np.where(det)
					for k in range(n_det):
	 					gd_samples = gd.MCSamples(samples=gw_samples[:, ijk[0][k]], \
												  names=['d'], \
												  labels=['d'], \
												  ranges={})
						h_0_post = gd_samples.get1DDensity('d')
						h_0_post_max = so.minimize(h_0_max_post_wrapper, d_true[ijk[0][k]], \
												   args=h_0_post)['x'][0]
						gw_d_max[k] = h_0_post_max
						#print h_0_post_max, gw_d_mean[ijk[0][k]]

					delta_d = gw_d_max - d_true[det]
					no_sel_eff_bias = np.mean(delta_d / d_true[det]) * \
									  -10.0 / 0.15
					print 'expected bias neglecting sel eff ' + \
						  '{:7.3f} km/s/Mpc'.format(no_sel_eff_bias)
					ax.scatter(d_true[det], gw_d_max, color=cols[5], \
							   label='selected')
					exit()
					'''

					ax.scatter(d_true, gw_d_mean, color=cols[8], \
							   label='rejected')
					ax.scatter(d_true[det], gw_d_mean[det], color=cols[2], \
							   label='selected')
					ax.plot([0.0, 500.0], [0.0, 500.0], 'k-')
					ax.set_xlabel(r'$D\,({\rm Mpc})$', fontsize=16)
					ax.set_ylabel(r'$\hat{D}\,({\rm Mpc})$', fontsize=16)
					ax.set_xlim(0.0, 500.0)
					ax.set_ylim(0.0, 500.0)
					ax.set_aspect('equal')
					ax.legend(loc='lower right')
					fig.savefig(base + '_d_bias.pdf', bbox_inches='tight')
					mp.close(fig)
					fig, ax = mp.subplots(figsize=(6, 6))
					gw_cos_i_mean = np.mean(gw_samples, axis=0)[n_pars / 2:]
					ax.scatter(cos_i_true, gw_cos_i_mean, color=cols[8], \
							   label='rejected')
					ax.scatter(cos_i_true[det], gw_cos_i_mean[det], \
							   color=cols[2], label='selected')
					ax.plot([-1.0, 1.0], [-1.0, 1.0], 'k-')
					ax.set_xlabel(r'$\cos \iota$', fontsize=16)
					ax.set_ylabel(r'${\rm estimated}\,\cos \iota$', fontsize=16)
					ax.set_xlim(-1.0, 1.0)
					ax.set_ylim(-1.0, 1.0)
					ax.set_aspect('equal')
					ax.legend(loc='lower right')
					fig.savefig(base + '_cos_i_bias.pdf', bbox_inches='tight')
					mp.close(fig)

					# plot subset of distance-inclination posteriors: 
					# half detected, half not
					'''
					n_plot = 6
					i_det = np.sort(npr.choice(n_det, n_plot / 2, \
											   replace=False))
					i_det = np.argwhere(det)[i_det, 0]
					i_miss = np.sort(npr.choice(n_event - n_det, \
												n_plot / 2, \
												replace=False))
					i_miss = np.argwhere(~det)[i_miss, 0]
					i_plot = np.append(i_det, i_miss)
					i_plot = np.append(i_plot, i_plot + n_pars / 2)
					'''
					n_plot = 3
					i_det = np.array([75, 86, 93])
					i_det = np.argwhere(det)[i_det, 0]
					i_plot = np.append(i_det, i_det + n_pars / 2)
					plot_samples = gw_samples[:, i_plot]
					gd_names = ['d_{:d}'.format(j) for j in range(n_plot)] + \
 							   ['c_{:d}'.format(j) for j in range(n_plot)]
 					gd_labels = [r'D\,({\rm Mpc})'] * n_plot + \
 								[r'\cos\iota'] * n_plot
 					gd_pairs = [['d_{:d}'.format(j), 'i_{:d}'.format(j)] \
 								for j in range(n_plot)]
 					gd_limits = merge_dicts({'d_{:d}'.format(j): (0.0, d_max) \
 											 for j in range(n_plot)}, \
 											{'c_{:d}'.format(j): (-1.0, 1.0) \
 											 for j in range(n_plot)})
 					gd_samples = gd.MCSamples(samples=plot_samples, \
											  names=gd_names, \
											  labels=gd_labels, \
											  ranges=gd_limits)
					for j in range(n_plot):
						gd_name = 'i_{:d}'.format(j)
						gd_label = r'\iota\,(\degree)'
						c_j = gw_samples[:, i_plot[j] + n_pars / 2]
						gd_samples.addDerived(np.arccos(c_j) * 180.0 / np.pi, \
											  name=gd_name, label=gd_label)
						gd_samples.setRanges({'i_{:d}'.format(j):(0.0, 180.0)})
					gd_samples.updateBaseStatistics()
					g = gdp.getSubplotPlotter(subplot_size=3)
					g.settings.lw_contour = lw
					g.plots_2d(gd_samples, param_pairs=gd_pairs, \
							   nx=3, filled=True, \
							   colors=[mpc.rgb2hex(cols[2])])
					axes = g.subplots.flatten()
					for j in range(n_plot):
						#axes[j].set_xlim(0.0, 500.0)
						#axes[j].set_ylim(0.0, 180.0)
						g.add_x_marker(d_true[i_plot[j]], ax=axes[j], \
									   lw=lw, color='k')
						g.add_y_marker(np.arccos(cos_i_true[i_plot[j]]) * \
									   180.0 / np.pi, ax=axes[j], lw=lw, \
									   color='k')
					mp.savefig(base + '_dist_inc_posts.pdf', \
							   bbox_inches='tight')
 					gd_pairs = [['d_{:d}'.format(j), 'c_{:d}'.format(j)] \
 								for j in range(n_plot)]
					g = gdp.getSubplotPlotter(subplot_size=3)
					g.settings.lw_contour = lw
					g.plots_2d(gd_samples, param_pairs=gd_pairs, \
							   nx=3, filled=True, \
							   colors=[mpc.rgb2hex(cols[2])])
					axes = g.subplots.flatten()
					for j in range(n_plot):
						g.add_x_marker(d_true[i_plot[j]], ax=axes[j], \
									   lw=lw, color='k')
						g.add_y_marker(cos_i_true[i_plot[j]], \
									   ax=axes[j], lw=lw, color='k')
					mp.savefig(base + '_dist_cos_inc_posts.pdf', \
							   bbox_inches='tight')
					exit()

			# STAN!
			#print m_c_prior_mean, m_c_prior_std
			#m_c_prior_mean += m_c_prior_std
			#m_c_prior_std *= 2.0
			#print m_c_prior_mean, m_c_prior_std
			if sample_d:
				model_base = 'bias_test_d_sample'
				base += '_d_sample'
			else:
				model_base = 'bias_test'
			if recompile:
				stan_model = ps.StanModel(model_base + '.stan')
				with open(model_base + '_model.pkl', 'wb') as f:
					pickle.dump(stan_model, f)
			else:
				try:
					#with open(base + '_model.pkl', 'rb') as f:
					with open(model_base + '_model.pkl', 'rb') as f:
						stan_model = pickle.load(f)
				except EnvironmentError:
					print 'ERROR: pickled Stan model (' + model_base + \
						  '_model.pkl) not found. Please set recompile = True'
					exit()

			# set up stan inputs and sample
			stan_data = {'ntlo': int(ntlo), 'vary_m_c': int(vary_m_c), \
						 'z_dep_rate': int(inc_rate_redshift), \
						 'fixed_n_bns': int(fixed_n_bns), \
						 'n_bns': n_det, 'obs_amp_plus': amp_plus_obs[det], \
						 'obs_amp_cross': amp_cross_obs[det], \
						 'obs_v_pec': v_pec_obs[det], 'obs_z': z_obs[det], \
						 'amp_n': amp_n, 'sig_v_pec': sig_v_pec, \
						 'sig_obs_v_pec': sig_v_pec_obs, \
						 'sig_z': sig_z_obs, 'z_max': z_max, \
						 'n_coeffs': len(popt), 'n_bar_det_coeffs': popt}
			if vary_m_c:
				stan_data['mu_m_c'] = m_c_prior_mean
				stan_data['sig_m_c'] = m_c_prior_std
				stan_data['amp_s'] = 0.0
				stan_data['obs_m_c_z'] = m_c_z_obs[det]
				stan_data['sig_obs_m_c_z'] = sig_m_c_z_obs
			else:
				stan_data['mu_m_c'] = 0.0
				stan_data['sig_m_c'] = 0.0
				stan_data['amp_s'] = amp_s
				stan_data['obs_m_c_z'] = np.zeros(n_det)
				stan_data['sig_obs_m_c_z'] = 0.0
			if sample_d:
				if ntlo:
					stan_data['d_max'] = c * z_max / h_0_true * (1.0 + \
										 0.5 * (1.0 - q_0_true) * z_max)
				else:
					stan_data['d_max'] = c * z_max / h_0_true
			stan_pars = pars
			if stan_constrain:
				stan_seed = 23102014
			else:
				stan_seed = None
			report_true_m_c = False
			if report_true_m_c:
				stan_pars.append('true_m_c')		
				fit = stan_model.sampling(data = stan_data, \
										  iter = n_samples, \
										  chains = n_chains, \
										  seed = stan_seed, \
										  pars = stan_pars)
				print fit
				raw_smplz = fit.extract(permuted = False, \
									inc_warmup = False)
				n_samples = n_samples / 2
				smplz = np.zeros((n_chains * n_samples, n_pars + n_det))
				for ii in range(0, n_chains):
					for jj in range(0, n_pars + n_det):
						smplz[ii * n_samples: (ii + 1) * n_samples, jj] = \
							raw_smplz[:, ii, jj]
				m_c_smplz = smplz[:, 2:]
				delta = np.mean(m_c_smplz, axis=0) - m_c_true[det]
				avg_std = np.mean(np.std(m_c_smplz, axis=0))
				print 'avg M_c bias: {:19.12e}'.format(np.mean(delta))
				print 'M_c prior std: {:19.12e}'.format(m_c_prior_std)
				print 'avg M_c std: {:19.12e}'.format(avg_std)
				exit()
			fit = stan_model.sampling(data = stan_data, \
									  iter = n_samples, \
									  chains = n_chains, \
									  seed = stan_seed, \
									  pars = stan_pars)
			samples[..., i] = fit.extract(permuted = False, \
										  inc_warmup = False)
			print fit
			if save_distances:
				d_trues[:, i] = d_true[det]

		# accumulate results and save to disk
		samples = complete_array(samples, use_mpi)
		if rank == 0:
			with h5py.File(base + '_samples_' + '{:d}'.format(n_rpt) + \
						   '_rpts.h5', 'w') as f:
				f.create_dataset('samples', data=samples)
		if save_distances:
			d_trues = complete_array(d_trues, use_mpi)
			if rank == 0:
				with h5py.File(base + '_true_ds_' + '{:d}'.format(n_rpt) + \
							   '_rpts.h5', 'w') as f:
					f.create_dataset('true_ds', data=d_trues)

	# read in samples on master process and broadcast to others
	if rank == 0:

		# retrieve samples and convert to required GetDist format
		with h5py.File(base + '_samples_' + '{:d}'.format(n_rpt) + \
					   '_rpts.h5', 'r') as f:
			raw_samples = f['samples'][:]
		n_chains = raw_samples.shape[1]
		n_samples = raw_samples.shape[0]
		samples = np.zeros((n_chains * n_samples, n_pars, n_rpt))
		for i in range(0, n_chains):
			for j in range(0, n_pars):
				samples[i * n_samples: (i + 1) * n_samples, j, :] = \
					raw_samples[:, i, j, :]

		# plots
		if save_distances:
			with h5py.File(base + '_true_ds_' + '{:d}'.format(n_rpt) + \
						   '_rpts.h5', 'r') as f:
				d_trues = f['true_ds'][:]
			i = pars.index('true_d')
			pars[i] = 'true_d1'
			for j in range(1, n_event):
				pars.insert(i + j, 'true_d{:d}'.format(j + 1))
			gd_samples = gd.MCSamples(samples=samples[..., 0], names=pars, 
									  labels=par_names, ranges=par_ranges)
			print gd_samples.getMeans()
			print d_trues[:, 0]
			mp.hist(gd_samples.getMeans()[2:] - d_trues[:,0])
			mp.show()
			exit()
			g = gdp.getSubplotPlotter()
			g.settings.lw_contour = lw
			g.settings.axes_fontsize = 8
			g.triangle_plot(gd_samples, pars, filled = True, \
							line_args = {'lw': lw, 'color': mpc.rgb2hex(cols[0])}, \
							contour_args = {'lws': [lw, lw]}, \
							colors = [mpc.rgb2hex(cols[0])])
			for i in range(0, n_pars):
				sp_title = '$' + gd_samples.getInlineLatex(pars[i], \
														   limit=1) + '$'
				g.subplots[i, i].set_title(sp_title, fontsize=12)
				if par_vals[i] is not None:
					for ax in g.subplots[i, :i]:
						ax.axhline(par_vals[i], color='gray', ls='--')
					for ax in g.subplots[i:, i]:
						ax.axvline(par_vals[i], color='gray', ls='--')
			mp.savefig(base + '_triangle_plot.pdf', bbox_inches = 'tight')
		else:
			gd_samples = gd.MCSamples(samples=samples[..., 0], names=pars, 
									  labels=par_names, ranges=par_ranges)
			if ntlo:
				par_vals = [h_0_true, q_0_true]
			else:
				par_vals = [h_0_true]
			g = gdp.getSubplotPlotter()
			g.settings.lw_contour = lw
			g.settings.axes_fontsize = 8
			g.triangle_plot(gd_samples, pars, filled = True, \
							line_args = {'lw': lw, 'color': mpc.rgb2hex(cols[0])}, \
							contour_args = {'lws': [lw, lw]}, \
							colors = [mpc.rgb2hex(cols[0])])
			for i in range(0, n_pars):
				sp_title = '$' + gd_samples.getInlineLatex(pars[i], \
														   limit=1) + '$'
				g.subplots[i, i].set_title(sp_title, fontsize=12)
				if par_vals[i] is not None:
					for ax in g.subplots[i, :i]:
						ax.axhline(par_vals[i], color='gray', ls='--')
					for ax in g.subplots[i:, i]:
						ax.axvline(par_vals[i], color='gray', ls='--')
			mp.savefig(base + '_triangle_plot.pdf', bbox_inches = 'tight')

	else:

		samples = np.zeros((n_chains * n_samples / 2, n_pars, n_rpt))

	if use_mpi:
		mpi.COMM_WORLD.Bcast(samples, 0)

	# find posterior maxima and means
	job_list = allocate_jobs(n_rpt, n_procs, rank)
	print 'process {:d} jobs: '.format(rank), job_list
	h_0_post_summaries = np.zeros((3, n_rpt))
	for i in job_list:
		gd_samples = gd.MCSamples(samples=samples[..., i], names=pars, 
								  labels=par_names, ranges=par_ranges)
		h_0_post = gd_samples.get1DDensity('h_0')
		h_0_post_max = so.minimize(h_0_max_post_wrapper, h_0_true, \
								   args=h_0_post)['x'][0]
		h_0_post_mean = gd_samples.getMeans()[0]
		h_0_post_var = gd_samples.getVars()[0]
		h_0_post_summaries[:, i] = (h_0_post_max, \
									h_0_post_mean, \
									np.sqrt(h_0_post_var))
	h_0_post_summaries = complete_array(h_0_post_summaries, use_mpi)
	if rank == 0:
		with h5py.File(base + '_h_0_summaries_' + '{:d}'.format(n_rpt) + \
					   '_rpts.h5', 'w') as f:
			f.create_dataset('h_0_post_summaries', \
							 data=h_0_post_summaries)

