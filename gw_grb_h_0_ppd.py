import numpy as np
import numpy.random as npr
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import sklearn.neighbors as skn
import sklearn.model_selection as skms
import scipy.stats as ss
import scipy.integrate as si
import scipy.interpolate as sin
import scipy.optimize as so
import os
import pickle
import getdist as gd
import getdist.plots as gdp

######################################################################
######################################################################
######################################################################

def like_d(d, mus, sigmas):

	# determine number of distances at which to evaluate
	if np.isscalar(d):
		like = 0.0
	else:
		like = np.zeros(len(d))

	# determine number of Gaussian components
	if np.isscalar(mus):
		n_comp = 1
	else:
		n_comp = len(mus)

	# return mean of probabilities of each component
	for i in range(n_comp):
		like += np.exp(-0.5 * ((d - mus[i]) / sigmas[i]) ** 2) / \
				np.sqrt(2.0 * np.pi) / sigmas[i]
	return like / n_comp

def ln_post_h_0(h_0, z, like_kdes, d_l_samples, sig_v_pec):

	# determine dimensionality
	if np.isscalar(z):
		n_event = 1
	else:
		n_event = len(z)
	if np.isscalar(h_0):
		n_h_0 = 1
	else:
		n_h_0 = len(h_0)
	ln_prob = -np.log(h_0) * n_event
	#ln_prob = np.zeros(n_h_0)
	for i in range(n_event):

		# convert H_0 grid into distances for this event
		d_grid = c * z[i] / h_0

		# extract means and std devs of KDE components
		kde_mus = d_l_samples[i]
		bw = like_kdes[i].get_params()['bandwidth']

		# loop over H_0 grid
		if n_h_0 == 1:

			# marginalize over peculiar velocities, introducing an H_0-
			# dependent scaling factor
			kde_sig = np.sqrt(bw ** 2 + (sig_v_pec / h_0) ** 2)
			kde_sigs = np.ones(len(kde_mus)) * kde_sig

			# evaluate likelihood!
			ln_prob += np.log(like_d(d_grid, kde_mus, kde_sigs))

		else:

			for j in range(n_h_0):

				# marginalize over peculiar velocities, introducing an H_0-
				# dependent scaling factor
				kde_sig = np.sqrt(bw ** 2 + (sig_v_pec / h_0[j]) ** 2)
				kde_sigs = np.ones(len(kde_mus)) * kde_sig

				# evaluate likelihood!
				ln_prob[j] += np.log(like_d(d_grid[j], kde_mus, kde_sigs))

	return ln_prob

def ln_post_h_0_no_kde(h_0, z, d_l_samples, sig_v_pec):

	# determine dimensionality
	if np.isscalar(z):
		n_event = 1
	else:
		n_event = len(z)
	if np.isscalar(h_0):
		n_h_0 = 1
	else:
		n_h_0 = len(h_0)
	ln_prob = -np.log(h_0) * n_event
	for i in range(n_event):

		# convert H_0 grid into distances for this event
		d_grid = c * z[i] / h_0

		# extract means and std devs of KDE components
		mus = d_l_samples[i]

		# loop over H_0 grid
		if n_h_0 == 1:

			# marginalize over peculiar velocities, introducing an H_0-
			# dependent scaling factor
			sigs = np.ones(len(mus)) * sig_v_pec / h_0

			# evaluate likelihood!
			ln_prob += np.log(like_d(d_grid, mus, sigs))

		else:

			for j in range(n_h_0):

				# marginalize over peculiar velocities, introducing an H_0-
				# dependent scaling factor
				sigs = np.ones(len(mus)) * sig_v_pec / h_0[j]

				# evaluate likelihood!
				ln_prob[j] += np.log(like_d(d_grid[j], mus, sigs))

	return ln_prob

def sampler(z, like_kdes, d_l_samples, sig_v_pec, n_samples=10, \
			h_0_init=70.0, prop_width=2.0):
	
	samples = np.zeros(n_samples)
	samples[0] = h_0_init
	accepts = 0.0
	for i in range(1, n_samples):

		# propose new sample, accept/reject
		prop = npr.randn(1)[0] * prop_width + samples[i - 1]
		ln_like_sample = ln_post_h_0(samples[i - 1], z, like_kdes, \
									 d_l_samples, sig_v_pec)
		ln_like_prop = ln_post_h_0(prop, z, like_kdes, \
								   d_l_samples, sig_v_pec)
		p_accept = np.exp(ln_like_prop - ln_like_sample)
		accept = p_accept >= npr.rand()
		if accept:
			samples[i] = prop
			accepts += 1.0
		else:
			samples[i] = samples[i - 1]
		
	return samples, accepts / float(n_samples)

def gd_opt_wrapper(h_0, gd_1d_density):
	return -gd_1d_density.Prob(h_0)[0]

def int_opt_wrapper(h_0, interpolant):
	return -interpolant(h_0)

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

def allocate_jobs_inc_time(n_jobs, n_procs=1, rank=0):
	allocated = []
	for i in range(n_jobs):
		if rank == np.mod(n_jobs-i, n_procs):
			allocated.append(i)
	return allocated

def complete_array(target_distrib, use_mpi=False):
	if use_mpi:
		target = np.zeros(target_distrib.shape)
		mpi.COMM_WORLD.Allreduce(target_distrib, target, op=mpi.SUM)
	else:
		target = target_distrib
	return target

def complete_list(target_distrib, use_mpi=False, n_rpt=0):
	if use_mpi:
		target = mpi.COMM_WORLD.allreduce(target_distrib, op=mpi.SUM)
	else:
		target = target_distrib
	if n_rpt == 0:
		return target
	else:
		n_target = len(target)
		n_slice = n_target / n_rpt
		reshaped = []
		for i in range(n_slice):
			reshaped.append(target[i::n_slice])
		return reshaped

######################################################################
######################################################################
######################################################################

# settings
recompute = False
use_mpi = True
ppd_via_sampling = False
sample = False
constrain = True
plot_event_h_0 = False
animate_p_h_0 = False
debug_plot = False
verbose = True
n_h_0_samples = 10000
c = 2.998e5
h_0_cmb = 67.81
sig_h_0_cmb = 0.92
h_0_loc = 73.24
sig_h_0_loc = 1.74
sig_v_pec = 200.0
bw_grid = np.logspace(-0.5, 3.5, 100)
d_l_grid = np.linspace(0.0, 500.0, 1000)
n_h_0_eval_grid = 100
h_0_eval_grid = np.linspace(50.0, 100.0, n_h_0_eval_grid)
d_h_0 = h_0_eval_grid[1] - h_0_eval_grid[0]
n_h_0_eval_grid_ext = 500
h_0_eval_grid_ext = np.linspace(10.0, 200.0, n_h_0_eval_grid_ext)
d_h_0_ext = h_0_eval_grid_ext[1] - h_0_eval_grid_ext[0]
n_h_0_true_grid = 2#3#25
h_0_true_grid = np.linspace(67.81, 73.24, n_h_0_true_grid)
n_bs = 1
if use_mpi:
	import mpi4py.MPI as mpi
	n_procs = mpi.COMM_WORLD.Get_size()
	rank = mpi.COMM_WORLD.Get_rank()
else:
	n_procs = 1
	rank = 0

# plotting settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')

# define colourblind-friendly colours. order is dark blue, orange,
# light blue, orange, then black and grey
cbf_cols = [ (0.0 / 255.0, 107.0 / 255.0, 164.0 / 255.0), \
			 (200.0 / 255.0, 82.0 / 255.0, 0.0 / 255.0), \
			 (95.0 / 255.0, 158.0 / 255.0, 209.0 / 255.0), \
			 (255.0 / 255.0, 128.0 / 255.0, 14.0 / 255.0), \
			 (0.0, 0.0, 0.0), \
			 (89.0 / 255.0, 89.0 / 255.0, 89.0 / 255.0) ]

# loop over files in directory
fnames = []
d_l = []
d_l_samples = []
kdes = []
snr = []
if recompute and rank == 0:
	fig, axes = mp.subplots(1, 2, figsize=(16, 5))
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.join(script_dir, 'data'), 'gw_grb')
for fname in sorted(os.listdir(data_dir)):
	if fname.endswith('.txt'):
		fnames.append(fname)
n_event = len(fnames)
job_list = allocate_jobs(n_event, n_procs, rank)
for i in job_list:

	# read samples
	samples = np.genfromtxt(os.path.join(data_dir, \
							fnames[i]))
	d_l_samples.append(samples)
	pkl_fname = os.path.join(data_dir, \
							 fnames[i].replace('.txt', '.pkl'))

	if recompute:

		# fit KDE to samples and pickle for later
		print 'fitting ' + fnames[i]
		gs = skms.GridSearchCV(skn.KernelDensity(), \
							   {'bandwidth': bw_grid})
		gs.fit(samples[:, None])
		kde = gs.best_estimator_
		pickle.dump(kde, open(pkl_fname, 'wb'))
		print 'optimal bw: {:9.3e}'.format(kde.bandwidth)

		# plot fits
		hist, bin_edges = np.histogram(samples, bins=50, \
									   density=True)
		bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
		pdf = np.exp(kde.score_samples(d_l_grid[:, None]))
		if rank == 0:
			axes[0].plot(bin_centres, hist)
			axes[1].plot(d_l_grid, pdf)

	else:

		# unpickle KDEs
		kde = pickle.load(open(pkl_fname, 'rb'))

	# gather data
	split = fnames[i].split('-')
	d_l_true = float(split[1][:-8].replace('p', '.'))
	d_l.append(d_l_true)
	snr.append(float(split[2][:-4].replace('p', '.')))
	kdes.append(kde)

if use_mpi:
	mpi.COMM_WORLD.barrier()
d_l = complete_list(d_l, use_mpi)
d_l = np.array(d_l)
d_l_samples = complete_list(d_l_samples, use_mpi)
kdes = complete_list(kdes, use_mpi)
snr = complete_list(snr, use_mpi)
event_cols = np.linspace(0.0, 0.9, n_event)
if rank == 0:
	if not recompute:
		for i in range(n_event):
			pdf = np.exp(kdes[i].score_samples(d_l_grid[:, None]))
			mp.plot(d_l_grid, pdf, color=cm(event_cols[i]))
		mp.xlabel(r'$d_l$')
		mp.ylabel(r'${\rm Pr}(d_l|\{\mathbf{x}\})$')
		mp.savefig('gw_d_l_posteriors.pdf', bbox_inches = 'tight')
	mp.close()
	#mp.show()

# ensure random number generators for different processes have same 
# seed
if constrain:
	seed = 102314
else:
	seed = np.random.randint(102314, 221216, 1)[0]
	if use_mpi:
		seed = mpi.COMM_WORLD.bcast(seed, root=0)
npr.seed(seed)

# draw one set of peculiar velocities for the events for each 
# bootstrapped sample
# NB: these peculiar velocity residuals technically have the wrong
# sign, but they're random and drawn from an even distribution so...
#v_pec = npr.randn(n_event) * sig_v_pec
v_pec = np.transpose(np.reshape(npr.randn(n_event * n_bs), \
					 (n_bs, n_event))) * sig_v_pec

# set up bootstraps: create array containing the source indices that 
# go into each bootstrap
bs_ids = np.zeros((n_event, n_bs), dtype=int)
bs_ids[:, 0] = np.arange(n_event)
bs_ids[:, 1:] = npr.choice(n_event, (n_event, n_bs - 1))

# draw 'instrumental noise' from CMB and local H_0 estimates for PPDs
# @TODO: should probably redo this for each bootstrap but we use so 
# many samples and it's a sample variance we don't want to explore
delta_h_0_cmb = sig_h_0_cmb * \
				np.reshape(npr.randn(n_h_0_samples * n_h_0_true_grid), \
						   (n_h_0_samples, n_h_0_true_grid))
delta_h_0_loc = sig_h_0_loc * \
				np.reshape(npr.randn(n_h_0_samples * n_h_0_true_grid), \
						   (n_h_0_samples, n_h_0_true_grid))

# optionally plot H_0 posteriors for individual objects
if plot_event_h_0:

	# cheeky hidden setting variables
	oplot_no_kde = False
	oplot_no_v_pec_hist = False

	# sort events by signal-to-noise. pick first bootstrap and Planck
	# H_0, convert distances into redshifts and evaluate individual 
	# posteriors
	snr_srt, kdes_srt, d_l_srt, d_l_samples_srt, v_pec_srt = \
		zip(*sorted(zip(snr, kdes, d_l, d_l_samples, v_pec[:, 0])))
	d_l_srt = np.array(d_l_srt)
	v_pec_srt = np.array(v_pec_srt)
	z = (v_pec_srt + h_0_true_grid[0] * d_l_srt) / c
	ln_prob = np.zeros((n_h_0_eval_grid_ext, n_event))
	if oplot_no_kde:
		ln_prob_no_kde = np.zeros((n_h_0_eval_grid_ext, n_event))
	job_list = allocate_jobs(n_event, n_procs, rank)
	print 'process {:d} jobs: '.format(rank), job_list
	for i in job_list:
		ln_prob_i = ln_post_h_0(h_0_eval_grid_ext, [z[i]], \
								[kdes_srt[i]], \
								[d_l_samples_srt[i]], sig_v_pec)
		norm_prob_i = np.sum(np.exp(ln_prob_i)) * d_h_0_ext
		ln_prob[:, i] = ln_prob_i - np.log(norm_prob_i)
		if oplot_no_kde:
			ln_prob_i = ln_post_h_0_no_kde(h_0_eval_grid_ext, \
										   [z[i]], \
										   [d_l_samples_srt[i]], \
										   sig_v_pec)
			norm_prob_i = np.sum(np.exp(ln_prob_i)) * d_h_0_ext
			ln_prob_no_kde[:, i] = ln_prob_i - np.log(norm_prob_i)
	if use_mpi:
		mpi.COMM_WORLD.barrier()
	ln_prob = complete_array(ln_prob, use_mpi)
	if oplot_no_kde:
		ln_prob_no_kde = complete_array(ln_prob_no_kde, use_mpi)

	# also calculate the combined posterior
	z = (v_pec[:, 0] + h_0_true_grid[0] * d_l) / c	
	n_h_0_eval_grid = 500
	h_0_eval_grid = np.linspace(50.0, 100.0, n_h_0_eval_grid)
	d_h_0 = h_0_eval_grid[1] - h_0_eval_grid[0]
	ln_prob_tot = ln_post_h_0(h_0_eval_grid, z, kdes, \
							  d_l_samples, sig_v_pec)
	norm_prob_tot = np.sum(np.exp(ln_prob_tot)) * d_h_0
	ln_prob_tot -= np.log(norm_prob_tot)

	# plot the results and terminate evaluation
	if rank == 0:
		fname_str = \
			'{:6.3f}'.format(h_0_true_grid[0]).replace('.', 'p')
		for i in range(n_event):
			mp.plot(h_0_eval_grid_ext, np.exp(ln_prob[:, i]), \
					color=cm(event_cols[i]), alpha=0.7)
			if oplot_no_kde:
				mp.plot(h_0_eval_grid_ext, \
						np.exp(ln_prob_no_kde[:, i]), \
						color=cm(event_cols[i]), ls='--', alpha=0.7)
			if oplot_no_v_pec_hist:
				mp.hist(h_0_true_grid[0] * d_l_srt[i] / \
						d_l_samples_srt[i], \
						normed=True, bins=50, fc=cm(event_cols[i]), \
						alpha=0.3)
		mp.plot(h_0_eval_grid, np.exp(ln_prob_tot) / 3.0, \
				color='k')
		mp.axvline(h_0_true_grid[0], color='k', ls='--')
		mp.xlabel(r'$H_0\,{\rm [km/s/Mpc]}$', fontsize=18)
		#mp.ylabel(r'${\rm Pr}(H_0|\{\mathbf{x},z\})$', fontsize=18)
		mp.ylabel(r'${\rm Pr}\left(H_0|\{\mathbf{x}\}, ' + \
				  r'\{\hat{v}^{\rm p}\}, \{\hat{z}\}, I\right)$', \
				  fontsize=18)
		mp.tick_params(axis='both', which='major', labelsize=16)
		mp.xlim(10, 170)
		mp.ylim(0.0, 0.12)
		mp.savefig('gw_grb_h_0_' + fname_str + \
				   '_individual_posteriors.pdf', \
				   bbox_inches = 'tight')
		mp.close()
	if use_mpi:
		mpi.Finalize()
	exit()

# optionally animate build up of posterior
if animate_p_h_0:

	# cheeky hidden setting variable
	images_exist = False
	if not images_exist:

		# sort events by signal-to-noise. pick first bootstrap
		snr_srt, kdes_srt, d_l_srt, d_l_samples_srt, v_pec_srt = \
			zip(*sorted(zip(snr, kdes, d_l, d_l_samples, v_pec[:, 0]), \
				reverse=True))
		d_l_srt = np.array(d_l_srt)
		v_pec_srt = np.array(v_pec_srt)
		z = (v_pec_srt + h_0_true_grid[0] * d_l_srt) / c

		# loop over events
		ln_prob_ind = np.zeros((n_h_0_eval_grid_ext, n_event))
		ln_prob_tot = np.zeros((n_h_0_eval_grid_ext, n_event))
		job_list = allocate_jobs_inc_time(n_event, n_procs, rank)
		print 'process {:d} jobs: '.format(rank), job_list
		for n in job_list:

			# calculate individual posterior and product
			ln_prob_n = ln_post_h_0(h_0_eval_grid_ext, [z[n]], \
									[kdes_srt[n]], [d_l_samples_srt[n]], \
									sig_v_pec)
			norm_prob_n = np.sum(np.exp(ln_prob_n)) * d_h_0_ext
			ln_prob_ind[:, n] = ln_prob_n - np.log(norm_prob_n)
			if n > 0:
				ln_prob_n = ln_post_h_0(h_0_eval_grid_ext, z[0: n + 1], \
										kdes_srt[0: n + 1], \
										d_l_samples_srt[0: n + 1], \
										sig_v_pec)
				norm_prob_n = np.sum(np.exp(ln_prob_n)) * d_h_0_ext
				ln_prob_tot[:, n] = ln_prob_n - np.log(norm_prob_n)
			else:
				ln_prob_tot[:, n] = ln_prob_ind[:, n]
		if use_mpi:
			mpi.COMM_WORLD.barrier()
		ln_prob_ind = complete_array(ln_prob_ind, use_mpi)
		ln_prob_tot = complete_array(ln_prob_tot, use_mpi)

		# plot the results
		snr_bins = np.linspace(snr_srt[-1], snr_srt[0], 11)
		if rank == 0:
			fname_str = \
				'{:6.3f}'.format(h_0_true_grid[0]).replace('.', 'p')
			for i in range(n_event):
				fig, axes = mp.subplots(1, 3, figsize=(16, 5))
				for j in range(i + 1):
					axes[0].plot(h_0_eval_grid_ext, \
								 np.exp(ln_prob_ind[:, j]), \
								 color=cm(event_cols[j]), alpha=0.7)
				axes[1].plot(h_0_eval_grid_ext, np.exp(ln_prob_tot[:, i]), \
							 color=cm(event_cols[n_event / 2]))
				axes[2].hist(snr_srt[0: i + 1], bins=snr_bins, \
							 fc=cm(event_cols[n_event / 2]))
				axes[0].axvline(h_0_true_grid[0], color='k', ls='--')
				axes[1].axvline(h_0_true_grid[0], color='k', ls='--')
				axes[0].set_xlabel(r'$H_0\,[{\rm km/s/Mpc}]$')
				axes[0].set_ylabel(r'${\rm Pr}(H_0|\mathbf{x}_i,\hat{z}_i)$')
				axes[0].set_xlim(10, 200)
				axes[0].set_ylim(0, 0.12)
				axes[1].set_xlabel(r'$H_0\,[{\rm km/s/Mpc}]$')
				axes[1].set_ylabel(r'${\rm Pr}(H_0|\{\mathbf{x},\hat{z}\})$')
				axes[1].set_xlim(50, 100)
				axes[1].set_ylim(0, 0.35)
				axes[2].set_xlabel(r'${\rm SNR}$')
				axes[2].set_ylabel(r'$N({\rm SNR})$')
				axes[2].set_ylim(0, 25)
				mp.savefig('gw_grb_h_0_' + fname_str + \
						   '_posterior_frame_' + \
						   '{:03d}.png'.format(i + 1), \
						   bbox_inches = 'tight', dpi=200)
				mp.close()

	# build animation and exit
	if rank == 0:
		import imageio
		images = []
		fname_str = \
			'{:6.3f}'.format(h_0_true_grid[0]).replace('.', 'p')
		for i in range(n_event):
			fname = 'gw_grb_h_0_' + fname_str + \
					'_posterior_frame_' + \
					'{:03d}.png'.format(i + 1)
			print fname
			images.append(imageio.imread(fname))
		imageio.mimsave('gw_grb_h_0_' + fname_str + \
						'_posterior_anim.gif', images, \
						duration=0.25)
	if use_mpi:
		mpi.Finalize()
	exit()

# set up jobs: want to loop over true H_0 values and bootstraps. 
# ij contains the true H_0 and bootstrap indices for each job
n_jobs = n_h_0_true_grid * n_bs
ij = np.zeros((n_jobs, 2), dtype=int)
for k in range(n_bs):
	ij[k * n_h_0_true_grid: (k + 1) * n_h_0_true_grid, 0] = \
		range(n_h_0_true_grid)
	ij[k * n_h_0_true_grid: (k + 1) * n_h_0_true_grid, 1] = k

# loop over underlying true H_0 values
job_list = allocate_jobs(n_jobs, n_procs, rank)
print 'process {:d} jobs: '.format(rank), job_list
ln_prob = np.zeros((n_h_0_eval_grid, n_h_0_true_grid, n_bs))
ppd_cmb = np.zeros((n_h_0_eval_grid, n_h_0_true_grid, n_bs))
ppd_loc = np.zeros((n_h_0_eval_grid, n_h_0_true_grid, n_bs))
if ppd_via_sampling:
	h_0_samples = np.zeros((n_h_0_samples, n_h_0_true_grid, n_bs))
	h_0_cmb_samples = np.zeros((n_h_0_samples, n_h_0_true_grid, n_bs))
	h_0_loc_samples = np.zeros((n_h_0_samples, n_h_0_true_grid, n_bs))
	post_means = []
	post_stds = []
	p_h_0 = []
	p_h_0_cmb = []
	p_h_0_loc = []
pars = ['h_0', 'h_0_cmb', 'h_0_loc']
par_names = [r'H_0', r'\hat{H}_0^{\rm cmb}', r'\hat{H}_0^{\rm loc}']
par_ranges = {}
for k in job_list:

	# grab true H_0 (i) and bootstrap (j) indices from array
	i, j = ij[k, :]

	# helpful filename string
	h_0_true_str = \
		'{:6.3f}'.format(h_0_true_grid[i]).replace('.', 'p')

	# generate bootstraps: need to ensure bootstraps are consistently 
	# constructed from kdes, d_l_samples and true d_l lists/arrays. 
	# different v_pecs have already been randomly sampled
	kdes_loc = []
	d_l_samples_loc = []
	d_l_loc = np.zeros(n_event)
	for m in range(n_event):
		kdes_loc.append(kdes[bs_ids[m, j]])
		d_l_samples_loc.append(d_l_samples[bs_ids[m, j]])
		d_l_loc[m] = d_l[bs_ids[m, j]]

	# convert velocities to redshifts
	# c * z = v_tot = v_pec + H_0 d
	z = (v_pec[:, j] + h_0_true_grid[i] * d_l_loc) / c
	
	# evaluate posterior and (roughly) normalize!
	#ln_prob[:, i] = ln_post_h_0(h_0_eval_grid, z, kdes, d_l_samples, \
	#							sig_v_pec)
	ln_prob_i = ln_post_h_0(h_0_eval_grid, z, kdes_loc, \
							d_l_samples_loc, sig_v_pec)
	norm_prob_i = np.sum(np.exp(ln_prob_i)) * d_h_0
	ln_prob[:, i, j] = ln_prob_i - np.log(norm_prob_i)

	# convolve posterior & likelihood to generate PPDs. make sure the 
	# convolution kernel covers at least the range +/- 5 sigma
	n_h_0_like = int(np.ceil(sig_h_0_loc / d_h_0 * 10)) + 1
	if np.mod(n_h_0_like, 2) == 0:
		n_h_0_like += 1
	h_0_like = (np.arange(n_h_0_like) - (n_h_0_like - 1) / 2) * d_h_0
	ppd_i = np.convolve(np.exp(ln_prob[:, i, j]), \
						ss.norm.pdf(h_0_like, 0.0, sig_h_0_cmb), \
						mode='same')
	norm_ppd_i = np.sum(ppd_i) * d_h_0
	ppd_cmb[:, i, j] = ppd_i / norm_ppd_i
	ppd_i = np.convolve(np.exp(ln_prob[:, i, j]), \
						ss.norm.pdf(h_0_like, 0.0, sig_h_0_loc), \
						mode='same')
	norm_ppd_i = np.sum(ppd_i) * d_h_0
	ppd_loc[:, i, j] = ppd_i / norm_ppd_i
	
	# sample posterior and PPDs!
	if ppd_via_sampling:
		if sample:
			h_0_samples[:, i, j], eff = sampler(z, kdes_loc, \
												d_l_samples_loc, \
												sig_v_pec, \
												n_samples=n_h_0_samples, \
												prop_width=5.0)
			h_0_cmb_samples[:, i, j] = h_0_samples[:, i, j] + \
									   delta_h_0_cmb[:, i]
			h_0_loc_samples[:, i, j] = h_0_samples[:, i, j] + \
									   delta_h_0_loc[:, i]
			np.savetxt('gw_grb_h_0_' + h_0_true_str + '_chain_' + \
					   '{:d}.csv'.format(j + 1), \
					   np.transpose([h_0_samples[:, i, j], \
					   				 h_0_cmb_samples[:, i, j], \
					   				 h_0_loc_samples[:, i, j]]), \
					   delimiter=',')
			all_samples = np.stack((h_0_samples[:, i, j], \
									h_0_cmb_samples[:, i, j], \
									h_0_loc_samples[:, i, j]), 1)
			print rank, i, j, eff
		else:
			all_samples = np.genfromtxt('gw_grb_h_0_' + h_0_true_str + \
										'_chain_' + \
										'{:d}.csv'.format(j + 1), \
										delimiter=',')
			h_0_samples[:, i, j] = all_samples[:, 0]
			h_0_cmb_samples[:, i, j] = all_samples[:, 1]
			h_0_loc_samples[:, i, j] = all_samples[:, 2]

		# process using GetDist. store means and interpolated probability
		# distributions
		gd_samples = gd.MCSamples(samples=all_samples, names=pars, 
								  labels=par_names, ranges=par_ranges)
		post_means.append(gd_samples.getMeans())
		post_stds.append(np.sqrt(gd_samples.getVars()))
		p_h_0.append(gd_samples.get1DDensity('h_0'))
		p_h_0_cmb.append(gd_samples.get1DDensity('h_0_cmb'))
		p_h_0_loc.append(gd_samples.get1DDensity('h_0_loc'))

# gather posteriors
if use_mpi:
	mpi.COMM_WORLD.barrier()
ln_prob = complete_array(ln_prob, use_mpi)
ppd_cmb = complete_array(ppd_cmb, use_mpi)
ppd_loc = complete_array(ppd_loc, use_mpi)
if ppd_via_sampling:
	h_0_samples = complete_array(h_0_samples, use_mpi)
	h_0_cmb_samples = complete_array(h_0_cmb_samples, use_mpi)
	h_0_loc_samples = complete_array(h_0_loc_samples, use_mpi)
	post_means = complete_list(post_means, use_mpi, n_bs)
	post_stds = complete_list(post_stds, use_mpi, n_bs)
	p_h_0 = complete_list(p_h_0, use_mpi, n_bs)
	p_h_0_cmb = complete_list(p_h_0_cmb, use_mpi, n_bs)
	p_h_0_loc = complete_list(p_h_0_loc, use_mpi, n_bs)

# plots
event_cols = np.linspace(0.0, 0.9, n_h_0_true_grid)
if rank == 0:

	# plot posteriors
	if debug_plot and ppd_via_sampling:
		for i in range(n_h_0_true_grid):
			for j in range(n_bs):
				mp.plot(h_0_eval_grid, np.exp(ln_prob[:, i, j]), \
						color=cm(event_cols[i]), label='analytic')
				p_h_0_norm = np.sum(p_h_0[i][j].Prob(h_0_eval_grid)) * \
							 d_h_0
				mp.plot(h_0_eval_grid, p_h_0[i][j].Prob(h_0_eval_grid) / \
						p_h_0_norm, ls=':', color='k', label='GetDist')
				mp.axvline(h_0_true_grid[i], color=cm(event_cols[i]), ls='--')
				mp.xlabel(r'$H_0$')
				mp.ylabel(r'${\rm Pr}(H_0|\{\mathbf{x}, z\})$')
				mp.legend(loc='upper right', fontsize=12)
				h_0_true_str = \
					'{:6.3f}'.format(h_0_true_grid[i]).replace('.', 'p')
			mp.savefig('gw_grb_h_0_' + h_0_true_str + \
					   '_posteriors.pdf', bbox_inches = 'tight')
			mp.close()
	else:
		for j in range(n_bs):
			for i in range(n_h_0_true_grid):
				mp.plot(h_0_eval_grid, np.exp(ln_prob[:, i, j]), \
						color=cm(event_cols[i]))
				mp.axvline(h_0_true_grid[i], \
						   color=cm(event_cols[i]), ls='--')
			mp.xlabel(r'$H_0$')
			mp.ylabel(r'${\rm Pr}(H_0|\{\mathbf{x}, z\})$')
			mp.savefig('gw_grb_h_0_posterior_' + 
					   '{:d}.pdf'.format(j + 1), \
					   bbox_inches = 'tight')
			mp.close()

	# plot samples
	cmb_ppd_max = np.zeros((n_h_0_true_grid, n_bs))
	cmb_pte = np.zeros((n_h_0_true_grid, n_bs))
	cmb_pr = np.zeros((n_h_0_true_grid, n_bs))
	cmb_pr_alt = np.zeros((n_h_0_true_grid, n_bs))
	loc_ppd_max = np.zeros((n_h_0_true_grid, n_bs))
	loc_pte = np.zeros((n_h_0_true_grid, n_bs))
	loc_pr = np.zeros((n_h_0_true_grid, n_bs))
	loc_pr_alt = np.zeros((n_h_0_true_grid, n_bs))
	post_mean = np.zeros((n_h_0_true_grid, n_bs))
	post_var = np.zeros((n_h_0_true_grid, n_bs))
	for j in range(n_bs):

		if verbose:
			print '** bootstrap {:d}'.format(j + 1)

		if ppd_via_sampling:

			# do things one way if we have samples to play with
			for i in range(n_h_0_true_grid):

				# plot posteriors and PPDs
				hist, bin_edges = np.histogram(h_0_samples[:, i, j], \
											   bins=100, density=True)
				bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
				mp.plot(bin_centres, hist, color=cm(event_cols[i]))
				hist, bin_edges = np.histogram(h_0_cmb_samples[:, i, j], \
											   bins=100, density=True)
				bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
				mp.plot(bin_centres, hist, color=cm(event_cols[i]), ls=':')
				hist, bin_edges = np.histogram(h_0_loc_samples[:, i, j], \
											   bins=100, density=True)
				bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
				mp.plot(bin_centres, hist, color=cm(event_cols[i]), \
						ls='-.')
				mp.axvline(h_0_true_grid[i], color=cm(event_cols[i]), \
						   ls='--')

				# copy some variables. not elegant, but meh
				post_mean[i, j] = post_means[i][j][0]
				post_var[i, j] = post_stds[i][j][0] ** 2

				# calculate PTEs
				cmb_inds = h_0_cmb_samples[:, i, j] < h_0_cmb
				cmb_pte[i, j] = np.sum(cmb_inds) / float(n_h_0_samples)
				loc_inds = h_0_loc_samples[:, i, j] > h_0_loc
				loc_pte[i, j] = np.sum(loc_inds) / float(n_h_0_samples)

				# find PPD maxima
				cmb_ppd_max_res = so.minimize(gd_opt_wrapper, \
											  h_0_true_grid[i], \
											  args=(p_h_0_cmb[i][j]))
				cmb_ppd_max[i, j] = cmb_ppd_max_res['x'][0]
				loc_ppd_max_res = so.minimize(gd_opt_wrapper, \
											  h_0_true_grid[i], \
											  args=(p_h_0_loc[i][j]))
				loc_ppd_max[i, j] = loc_ppd_max_res['x'][0]

				# calculate probability ratios
				cmb_pr[i, j] = p_h_0_cmb[i][j].Prob(h_0_cmb)[0] / \
							   p_h_0_cmb[i][j].Prob(post_means[i][j][1])[0]
				cmb_pr_alt[i, j] = p_h_0_cmb[i][j].Prob(h_0_cmb)[0] / \
								   p_h_0_cmb[i][j].Prob(cmb_ppd_max[i, j])[0]
				loc_pr[i, j] = p_h_0_loc[i][j].Prob(h_0_loc)[0] / \
							   p_h_0_loc[i][j].Prob(post_means[i][j][2])[0]
				loc_pr_alt[i, j] = p_h_0_loc[i][j].Prob(h_0_loc)[0] / \
								   p_h_0_loc[i][j].Prob(loc_ppd_max[i, j])[0]

				# report progress
				if verbose:
					print '* true H_0 = {:6.3f}'.format(h_0_true_grid[i])
					cmb_str = '{:d} of {:d} CMB samples ({:8.1e}) exceed ' + \
							  'observation'
					print cmb_str.format(np.sum(cmb_inds), n_h_0_samples, \
										 cmb_pte[i, j])
					loc_str = '{:d} of {:d} local samples ({:8.1e}) exceed ' + \
							  'observation'
					print loc_str.format(np.sum(loc_inds), n_h_0_samples, \
										 loc_pte[i, j])
					pos_str = 'PPD mean; mode; meas: {:11.5e}; {:11.5e}; {:11.5e}'
					val_str = 'PPD @ mean; mode; meas: {:11.5e}; {:11.5e}; {:11.5e}'
					rat_str = 'PPD @ meas/mean; meas/mode: {:11.5e}; {:11.5e}'
					print 'post mean: {:11.5e}'.format(post_means[i][j][0]) + \
						  ' +/- {:11.5e}'.format(post_stds[i][j][0])
					print 'CMB ' + \
						  pos_str.format(post_means[i][j][1], \
						  				 cmb_ppd_max[i, j], h_0_cmb)
					print 'CMB ' + \
						  val_str.format(p_h_0_cmb[i][j].Prob(post_means[i][j][1])[0], \
						  				 p_h_0_cmb[i][j].Prob(cmb_ppd_max[i, j])[0], \
						  				 p_h_0_cmb[i][j].Prob(h_0_cmb)[0])
					print 'CMB ' + \
						  rat_str.format(cmb_pr[i, j], cmb_pr_alt[i, j])
					print 'loc ' + \
						  pos_str.format(post_means[i][j][2], \
						  				 loc_ppd_max[i, j], h_0_loc)
					print 'loc ' + \
						  val_str.format(p_h_0_loc[i][j].Prob(post_means[i][j][2])[0], \
						  				 p_h_0_loc[i][j].Prob(cmb_ppd_max[i, j])[0], \
						  				 p_h_0_loc[i][j].Prob(h_0_loc)[0])
					print 'loc ' + \
						  rat_str.format(loc_pr[i, j], loc_pr_alt[i, j])

			mp.xlabel(r'$H_0$')
			mp.ylabel(r'${\rm Pr}(H_0|\{\mathbf{x}, z\})$')
			mp.savefig('gw_grb_h_0_posterior_samples_' + \
					   '{:d}.pdf'.format(j + 1), bbox_inches = 'tight')
			mp.close()

			mp.plot(h_0_true_grid, cmb_pte[:, j], color=cm(0.2), \
					label='CMB')
			mp.plot(h_0_true_grid, loc_pte[:, j], color=cm(0.8), \
					ls='--', label='distance ladder')
			mp.xlabel(r'$H_0^{\rm true} \, [{\rm km/s/Mpc}]$', \
					  fontsize=18)
			mp.ylabel(r'${\rm PTE}$', fontsize=18)
			mp.legend(loc='upper right', fontsize=18)
			mp.tick_params(axis='both', which='major', labelsize=18)
			mp.savefig('gw_grb_h_0_ppd_' + \
					   '{:d}.pdf'.format(j + 1), bbox_inches = 'tight')
			mp.close()

			mp.plot(h_0_true_grid, cmb_pr[:, j], color=cm(0.2), \
					label='CMB')
			mp.plot(h_0_true_grid, loc_pr[:, j], color=cm(0.8), \
					ls='--', label='distance ladder')
			mp.xlabel(r'$H_0^{\rm true} \, [{\rm km/s/Mpc}]$', \
					  fontsize=18)
			mp.ylabel(r'${\rm PPD}(\hat{H}_0^{\rm obs}) / ' + \
					  r'{\rm PPD}(\langle\hat{H}_0\rangle)$', \
					  fontsize=18)
			mp.legend(loc='upper right', fontsize=18)
			mp.tick_params(axis='both', which='major', labelsize=18)
			mp.savefig('gw_grb_h_0_ppd_prob_ratio_' + \
					   '{:d}.pdf'.format(j + 1), bbox_inches = 'tight')
			mp.close()

			# plot selected PPDs
			fig, axes = mp.subplots(1, 3, figsize=(18, 5), sharey=True)
			inds = [0, n_h_0_true_grid / 2, n_h_0_true_grid - 1]
			n_h_0_eval_grid = 1000
			h_0_eval_grid = np.linspace(50.0, 100.0, n_h_0_eval_grid)
			d_h_0 = h_0_eval_grid[1] - h_0_eval_grid[0]
			h_0_true_str = ''
			for i in range(3):
				ind = inds[i]
				axes[i].plot()
				axes[i].axvline(h_0_cmb, color=cbf_cols[2])
				axes[i].axvline(h_0_loc, color=cbf_cols[3])
				axes[i].axvline(h_0_true_grid[ind], color=cbf_cols[5], ls='--')
				p_h_0_norm = np.sum(p_h_0_cmb[ind][j].Prob(h_0_eval_grid)) * \
							 d_h_0
				axes[i].plot(h_0_eval_grid, \
							 p_h_0_cmb[ind][j].Prob(h_0_eval_grid) / \
							 p_h_0_norm, color=cbf_cols[0])
				p_h_0_norm = np.sum(p_h_0_loc[ind][j].Prob(h_0_eval_grid)) * \
							 d_h_0
				axes[i].plot(h_0_eval_grid, \
							 p_h_0_loc[ind][j].Prob(h_0_eval_grid) / \
							 p_h_0_norm, color=cbf_cols[1])
				h_0_str = r'$H_0^{\rm true} = ' + \
						  '{:6.2f}'.format(h_0_true_grid[ind]) + \
						  r'\,{\rm km/s/Mpc}$'
				axes[i].text(55.5, 0.276, h_0_str, fontsize=15)
				axes[i].set_xlabel(r'$\hat{H}_0\,[{\rm km/s/Mpc}]$', \
								   fontsize=18)
				if i == 0:
					axes[i].set_ylabel(r'${\rm Pr}(\hat{H}_0|\{\mathbf{x}, z\})$', \
									   fontsize=18)
				else:
					xticks = axes[i].xaxis.get_major_ticks()
					xticks[0].label1.set_visible(False)
				axes[i].tick_params(axis='both', which='major', labelsize=16)
				axes[i].set_xlim(55.0, 80.0)
				axes[i].set_ylim(0.0, 0.3)
				h_0_true_str += \
					'_{:6.3f}'.format(h_0_true_grid[ind]).replace('.', 'p')
			fig.subplots_adjust(hspace=0, wspace=0)
			mp.savefig('gw_grb_h_0' + h_0_true_str + \
					   '_ppds_{:d}.pdf'.format(j + 1), \
					   bbox_inches = 'tight')
			mp.close()

		else:

			# and another way if we've used np.convolve. first set up
			# some plots
			fig_dist, ax_dist = mp.subplots()
			fig_ppd, axes_ppd = mp.subplots(1, 3, figsize=(18, 5), \
											sharey=True)
			fig_ppd2, ax_ppd2 = mp.subplots()
			ax_ppd2.axvline(h_0_cmb, color=cbf_cols[2])
			ax_ppd2.axvline(h_0_loc, color=cbf_cols[3])
			lss = ['-', '--']
			h_0_true_str = ''
			h_0_true_str_alt = ''

			# then loop through H_0 true values
			for i in range(n_h_0_true_grid):

				# plot posteriors and PPDs
				ax_dist.plot(h_0_eval_grid, \
							 np.exp(ln_prob[:, i, j]), \
							 color=cm(event_cols[i]))
				ax_dist.plot(h_0_eval_grid, \
							 ppd_cmb[:, i, j], \
							 color=cm(event_cols[i]), ls=':')
				ax_dist.plot(h_0_eval_grid, \
							 ppd_loc[:, i, j], \
							 color=cm(event_cols[i]), ls='-.')
				ax_dist.axvline(h_0_true_grid[i], \
								color=cm(event_cols[i]), ls='--')

				# find posterior mean and std dev
				post_mean[i, j] = np.sum(np.exp(ln_prob[:, i, j]) * 
										 h_0_eval_grid) / \
						    np.sum(np.exp(ln_prob[:, i, j]))
				post_var[i, j] = np.sum(np.exp(ln_prob[:, i, j]) * 
										(h_0_eval_grid - \
										 post_mean[i, j]) ** 2) / \
								 np.sum(np.exp(ln_prob[:, i, j]))

				# interpolate PPDs onto finer grid
				n_h_0_int_grid = 500
				h_0_int_grid = np.linspace(50.0, 100.0, \
										   n_h_0_int_grid)
				d_h_0_int = h_0_int_grid[1] - h_0_int_grid[0]
				int_cmb = sin.interp1d(h_0_eval_grid, \
									   ppd_cmb[:, i, j], \
									   kind='cubic', \
									   bounds_error=False, \
									   fill_value=0.0)
				int_loc = sin.interp1d(h_0_eval_grid, \
									   ppd_loc[:, i, j], \
									   kind='cubic', \
									   bounds_error=False, \
									   fill_value=0.0)

				# plot selected PPDs
				if i == 0 or i == n_h_0_true_grid / 2 or \
					i == n_h_0_true_grid - 1:
					if i == 0:
						ind = 0
					elif i == n_h_0_true_grid / 2:
						ind = 1
					else:
						ind = 2
					axes_ppd[ind].axvline(h_0_cmb, color=cbf_cols[2])
					axes_ppd[ind].axvline(h_0_loc, color=cbf_cols[3])
					axes_ppd[ind].axvline(h_0_true_grid[i], \
										  color=cbf_cols[5], ls='--')
					p_h_0_norm = np.sum(int_cmb(h_0_int_grid)) * \
								 d_h_0_int
					axes_ppd[ind].plot(h_0_int_grid, \
									   int_cmb(h_0_int_grid) / \
									   p_h_0_norm, color=cbf_cols[0])
					p_h_0_norm = np.sum(int_loc(h_0_int_grid)) * \
								 d_h_0_int
					axes_ppd[ind].plot(h_0_int_grid, \
									   int_loc(h_0_int_grid) / \
									   p_h_0_norm, color=cbf_cols[1])
					h_0_str = r'$H_0^{\rm true} = ' + \
							  '{:6.2f}'.format(h_0_true_grid[i]) + \
							  r'\,{\rm km/s/Mpc}$'
					axes_ppd[ind].text(55.5, 0.276, h_0_str, \
									   fontsize=15)
					axes_ppd[ind].set_xlabel(r'$\hat{H}_0\,[{\rm km/s/Mpc}]$', \
											 fontsize=18)
					if i == 0:
						axes_ppd[ind].set_ylabel(r'${\rm Pr}(\hat{H}_0|\{\mathbf{x}, z\})$', \
												 fontsize=18)
					else:
						xticks = axes_ppd[ind].xaxis.get_major_ticks()
						xticks[0].label1.set_visible(False)
					axes_ppd[ind].tick_params(axis='both', \
											  which='major', \
											  labelsize=16)
					axes_ppd[ind].set_xlim(55.0, 80.0)
					axes_ppd[ind].set_ylim(0.0, 0.3)
					h_0_true_str += \
						'_{:6.3f}'.format(h_0_true_grid[i]).replace('.', 'p')

				# alternative version of above
				if i == 0 or i == n_h_0_true_grid / 2:
					if i == 0:
						ind = 0
					else:
						ind = 1
					p_h_0_norm = np.sum(int_cmb(h_0_int_grid)) * \
								 d_h_0_int
					ax_ppd2.plot(h_0_int_grid, \
								 int_cmb(h_0_int_grid) / \
								 p_h_0_norm, color=cbf_cols[0], \
								 ls=lss[ind])
					p_h_0_norm = np.sum(int_loc(h_0_int_grid)) * \
								 d_h_0_int
					ax_ppd2.plot(h_0_int_grid, \
								 int_loc(h_0_int_grid) / \
								 p_h_0_norm, color=cbf_cols[1], \
								 ls=lss[ind])
					h_0_true_str_alt += \
						'_{:6.3f}'.format(h_0_true_grid[i]).replace('.', 'p')
					ax_ppd2.fill_between([post_mean[i, j] - \
										  np.sqrt(post_var[i, j]), \
										  post_mean[i, j] + \
										  np.sqrt(post_var[i, j])], \
										  0.0, 0.3, color='lightgray', \
										  alpha=0.7, linewidth=0)

				# calculate PTEs: requires integration
				cmb_pte[i, j] = si.quad(int_cmb, 0.0, h_0_cmb)[0] / \
								si.quad(int_cmb, 0.0, 200.0)[0]
				loc_pte[i, j] = si.quad(int_loc, h_0_loc, 200.0)[0] / \
								si.quad(int_loc, 0.0, 200.0)[0]

				# find PPD means
				cmb_ppd_mean = np.sum(ppd_cmb[:, i, j] * \
									  h_0_eval_grid) / \
							   np.sum(ppd_cmb[:, i, j])
				loc_ppd_mean = np.sum(ppd_loc[:, i, j] * \
									  h_0_eval_grid) / \
							   np.sum(ppd_loc[:, i, j])

				# find PPD maxima
				cmb_ppd_max_res = so.minimize(int_opt_wrapper, \
											  h_0_true_grid[i], \
											  args=(int_cmb))
				cmb_ppd_max[i, j] = cmb_ppd_max_res['x'][0]
				loc_ppd_max_res = so.minimize(int_opt_wrapper, \
											  h_0_true_grid[i], \
											  args=(int_loc))
				loc_ppd_max[i, j] = loc_ppd_max_res['x'][0]

				# calculate probability ratios
				cmb_pr[i, j] = int_cmb(h_0_cmb) / \
							   int_cmb(cmb_ppd_mean)
				cmb_pr_alt[i, j] = int_cmb(h_0_cmb) / \
								   int_cmb(cmb_ppd_max[i, j])
				loc_pr[i, j] = int_loc(h_0_loc) / \
							   int_loc(loc_ppd_mean)
				loc_pr_alt[i, j] = int_loc(h_0_loc) / \
								   int_loc(loc_ppd_max[i, j])

				# report progress
				if verbose:
					print '* true H_0 = {:6.3f}'.format(h_0_true_grid[i])
					print 'post mean: {:11.5e}'.format(post_mean[i, j]) + \
						  ' +/- {:11.5e}'.format(np.sqrt(post_var[i, j]))
					cmb_str = 'CMB observation PTE = {:8.1e}'
					print cmb_str.format(cmb_pte[i, j])
					loc_str = 'local observation PTE = {:8.1e}'
					print loc_str.format(loc_pte[i, j])
					pos_str = 'PPD mean; mode; meas: {:11.5e}; {:11.5e}; {:11.5e}'
					val_str = 'PPD @ mean; mode; meas: {:11.5e}; {:11.5e}; {:11.5e}'
					rat_str = 'PPD @ meas/mean; meas/mode: {:11.5e}; {:11.5e}'
					print 'CMB ' + \
						  pos_str.format(cmb_ppd_mean, \
						  				 cmb_ppd_max[i, j], h_0_cmb)
					print 'CMB ' + \
						  val_str.format(int_cmb(cmb_ppd_mean).item(), \
						  				 int_cmb(cmb_ppd_max[i, j]).item(), \
						  				 int_cmb(h_0_cmb).item())
					print 'CMB ' + \
						  rat_str.format(cmb_pr[i, j], cmb_pr_alt[i, j])
					print 'loc ' + \
						  pos_str.format(loc_ppd_mean, \
						  				 loc_ppd_max[i, j], h_0_loc)
					print 'loc ' + \
						  val_str.format(int_loc(loc_ppd_mean).item(), \
						  				 int_loc(loc_ppd_max[i, j]).item(), \
						  				 int_loc(h_0_loc).item())
					print 'loc ' + \
						  rat_str.format(loc_pr[i, j], loc_pr_alt[i, j])

			# finish off posterior vs PPD plot
			ax_dist.set_xlim(55.0, 80.0)
			ax_dist.set_xlabel(r'$H_0$')
			ax_dist.set_ylabel(r'${\rm Pr}(H_0|\{\mathbf{x}, z\})$')
			fig_dist.savefig('gw_grb_h_0_posterior_convolutions_' + \
							 '{:d}.pdf'.format(j + 1), \
							 bbox_inches = 'tight')
			mp.close(fig_dist)

			# finish selected PPD plot
			fig_ppd.subplots_adjust(hspace=0, wspace=0)
			fig_ppd.savefig('gw_grb_h_0' + h_0_true_str + \
							'_ppds_{:d}.pdf'.format(j + 1), \
							bbox_inches = 'tight')
			mp.close(fig_ppd)

			# finish alternative selected PPD plot
			ax_ppd2.set_xlabel(r'$\hat{H}_0\,[{\rm km/s/Mpc}]$', \
							   fontsize=18)
			ax_ppd2.set_ylabel(r'${\rm Pr}\left(\hat{H}_0|\{\mathbf{x}\}, ' + \
							   r'\{\hat{v}^{\rm p}\}, \{\hat{z}\}, I\right)$', \
							   fontsize=18)
			ax_ppd2.tick_params(axis='both', \
								which='major', \
								labelsize=16)
			ax_ppd2.set_xlim(57.5, 82.5)
			ax_ppd2.set_ylim(0.0, 0.3)
			fig_ppd2.subplots_adjust(hspace=0, wspace=0)
			fig_ppd2.savefig('gw_grb_h_0' + h_0_true_str + \
							 '_ppds_{:d}.pdf'.format(j + 1), \
							 bbox_inches = 'tight')
			mp.close(fig_ppd)

			# PTE plots
			mp.plot(h_0_true_grid, cmb_pte[:, j], color=cm(0.2), \
					label='CMB')
			mp.plot(h_0_true_grid, loc_pte[:, j], color=cm(0.8), \
					ls='--', label='distance ladder')
			mp.xlabel(r'$H_0^{\rm true} \, [{\rm km/s/Mpc}]$', \
					  fontsize=18)
			mp.ylabel(r'${\rm PTE}$', fontsize=18)
			mp.legend(loc='upper right', fontsize=18)
			mp.tick_params(axis='both', which='major', labelsize=18)
			mp.savefig('gw_grb_h_0_ppd_' + \
					   '{:d}.pdf'.format(j + 1), bbox_inches = 'tight')
			mp.close()

			mp.plot(h_0_true_grid, cmb_pr[:, j], color=cm(0.2), \
					label='CMB')
			mp.plot(h_0_true_grid, loc_pr[:, j], color=cm(0.8), \
					ls='--', label='distance ladder')
			mp.xlabel(r'$H_0^{\rm true} \, [{\rm km/s/Mpc}]$', \
					  fontsize=18)
			mp.ylabel(r'${\rm PPD}(\hat{H}_0^{\rm obs}) / ' + \
					  r'{\rm PPD}(\langle\hat{H}_0\rangle)$', \
					  fontsize=18)
			mp.legend(loc='upper right', fontsize=18)
			mp.tick_params(axis='both', which='major', labelsize=18)
			mp.savefig('gw_grb_h_0_ppd_prob_ratio_' + \
					   '{:d}.pdf'.format(j + 1), bbox_inches = 'tight')
			mp.close()

	# save summaries
	np.savetxt('gw_grb_h_0_posterior_means.csv', \
			   np.hstack((h_0_true_grid[:, None], \
			   			  post_mean)), \
			   delimiter=',')
	np.savetxt('gw_grb_h_0_posterior_vars.csv', \
			   np.hstack((h_0_true_grid[:, None], \
			   			  post_var)), \
			   delimiter=',')
	np.savetxt('gw_grb_h_0_cmb_ppd_ptes.csv', \
			   np.hstack((h_0_true_grid[:, None], \
			   			  cmb_pte)), \
			   delimiter=',')
	np.savetxt('gw_grb_h_0_cmb_ppd_prs.csv', \
			   np.hstack((h_0_true_grid[:, None], \
			   			  cmb_pr_alt)), \
			   delimiter=',')
	np.savetxt('gw_grb_h_0_loc_ppd_ptes.csv', \
			   np.hstack((h_0_true_grid[:, None], \
			   			  loc_pte)), \
			   delimiter=',')
	np.savetxt('gw_grb_h_0_loc_ppd_prs.csv', \
			   np.hstack((h_0_true_grid[:, None], \
			   			  loc_pr_alt)), \
			   delimiter=',')

	# summarize the summaries! summery.
	for i in range(n_h_0_true_grid):
		print '* true H_0 = {:6.3f}'.format(h_0_true_grid[i])
		to_print = '  cmb_pte = {:11.5e} +/- {:11.5e}'
		print to_print.format(np.mean(cmb_pte[i, :]), \
							  np.sqrt(np.var(cmb_pte[i, :], ddof=1)))
		to_print = '  cmb_pr = {:11.5e} +/- {:11.5e}'
		print to_print.format(np.mean(cmb_pr[i, :]), \
							  np.sqrt(np.var(cmb_pr[i, :], ddof=1)))
		to_print = '  cmb_pr_alt = {:11.5e} +/- {:11.5e}'
		print to_print.format(np.mean(cmb_pr_alt[i, :]), \
							  np.sqrt(np.var(cmb_pr_alt[i, :], ddof=1)))
		to_print = '  loc_pte = {:11.5e} +/- {:11.5e}'
		print to_print.format(np.mean(loc_pte[i, :]), \
							  np.sqrt(np.var(loc_pte[i, :], ddof=1)))
		to_print = '  loc_pr = {:11.5e} +/- {:11.5e}'
		print to_print.format(np.mean(loc_pr[i, :]), \
							  np.sqrt(np.var(loc_pr[i, :], ddof=1)))
		to_print = '  loc_pr_alt = {:11.5e} +/- {:11.5e}'
		print to_print.format(np.mean(loc_pr_alt[i, :]), \
							  np.sqrt(np.var(loc_pr_alt[i, :], ddof=1)))

if use_mpi:
	mpi.Finalize()
exit()

