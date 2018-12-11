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

def pretty_hist(data, bins, axis, color, density=False, fill=True, \
				ls='-', zorder=None, label=None):

	hist, bin_edges = np.histogram(data, bins=bins, density=density)
	bins_to_plot = np.append(bins, bins[-1])
	hist_to_plot = np.append(np.insert(hist, 0, 0.0), 0.0)
	if zorder is not None:
		if label is not None:
			axis.step(bins_to_plot, hist_to_plot, where='pre', \
					  color=color, linestyle=ls, zorder=zorder, \
					  label=label)
		else:
			axis.step(bins_to_plot, hist_to_plot, where='pre', \
					  color=color, linestyle=ls, zorder=zorder)
		if fill:
			axis.fill_between(bins_to_plot, hist_to_plot, \
							  color=color, alpha=0.7, step='pre', \
							  zorder=zorder)
	else:
		if label is not None:
			axis.step(bins_to_plot, hist_to_plot, where='pre', \
					  color=color, linestyle=ls, label=label)
		else:
			axis.step(bins_to_plot, hist_to_plot, where='pre', \
					  color=color, linestyle=ls)
		if fill:
			axis.fill_between(bins_to_plot, hist_to_plot, \
							  color=color, alpha=0.7, step='pre')

def norm_dist(dist, delta_x):
	norm = np.sum(dist) * delta_x
	return dist / norm

# plotting settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')
cols = [cm(x) for x in np.linspace(0.1, 0.9, 10)]

# settings for toy problem
n_event = 100#100#100#10000
n_rpt = 1000#100
h_0_true = 70.0
q_0_true = -0.5
if n_event == 10:
	h_0_min = 20.0
	h_0_max = 200.0
elif n_event == 100:
	h_0_min = 67.0
	h_0_max = 73.0
elif n_event < 5000:
	h_0_min = 60.0
	h_0_max = 80.0
else:
	h_0_min = 60.0
	h_0_max = 72.0
fixed_n_bns = True
ntlo = True
vary_m_c = True
inc_rate_redshift = True
cut_bad_runs = True
n_overlay = 25
base = 'bias_test'
pars = ['h_0']
par_names = ['H_0']
par_ranges = {}
if ntlo:
	base += '_hq'
	pars += ['q_0']
	par_names += ['q_0']
else:
	q_0_true = None
n_pars = len(pars)
if vary_m_c:
	base += '_vary_m_c'
if inc_rate_redshift:
	base += '_rr'

# read data
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
with h5py.File(base + '_h_0_summaries_' + '{:d}'.format(n_rpt) + \
			   '_rpts.h5', 'r') as f:
	summaries = f['h_0_post_summaries'][:]

# select posteriors to plot, cutting weird runs if necessary
if cut_bad_runs:
	bad = summaries[2, :] > 2.0
	print 'removed {:d} bad runs'.format(np.sum(bad))
	summaries = summaries[:, ~bad]
	samples = samples[:, :, ~bad]
print 'mean(MAP H_0):', np.mean(summaries[0, :]), \
	  '+/-', np.std(summaries[0, :])
print 'mean(mean H_0):', np.mean(summaries[1, :]), \
	  '+/-', np.std(summaries[1, :])
print 'sqrt(mean(var H_0)):', np.sqrt(np.mean(summaries[2, :] ** 2)), \
	  '+/-', np.sqrt(np.std(summaries[2, :] ** 2))

# sort in terms of increasing MAP(H_0) and select subset to plot
i_map = np.argsort(summaries[0, :])
i_plot = np.sort(npr.choice(np.sum(~bad), n_overlay, replace=False))
cols = [cm(x) for x in np.linspace(0.1, 0.9, n_overlay)]

# plot posteriors
fig, axes = mp.subplots(1, 3, figsize=(18, 5))
n_grid = 500
h_0_min_plot = 60.0
h_0_max_plot = 80.0
h_0_grid = np.linspace(h_0_min_plot, h_0_max_plot, n_grid)
d_h_0 = h_0_grid[1] - h_0_grid[0]
if ntlo:
	fig2d, ax2d = mp.subplots(figsize=(6, 5))
	n_2d_grid = 40
	q_0_min_plot = 1.0
	q_0_max_plot = -2.0
	h_0_grid_2d = np.linspace(h_0_min_plot, h_0_max_plot, n_2d_grid)
	q_0_grid_2d = np.linspace(q_0_min_plot, q_0_max_plot, n_2d_grid)
	h_0_mgrid, q_0_mgrid = np.meshgrid(h_0_grid_2d, q_0_grid_2d)
for i in range(n_overlay):
	gd_samps = gd.MCSamples(samples=samples[..., i_map[i_plot[i]]], \
							names=pars, labels=par_names, \
							ranges=par_ranges)
	h_0_post = norm_dist(gd_samps.get1DDensity('h_0').Prob(h_0_grid), d_h_0)
	axes[0].plot(h_0_grid, h_0_post, color=cols[i], alpha=0.6)
if ntlo:
	for i in range(n_overlay):
		gd_samps = gd.MCSamples(samples=samples[..., i_map[i_plot[i]]], \
								names=pars, labels=par_names, \
								ranges=par_ranges)
		hq_post = gd_samps.get2DDensity('h_0', 'q_0')
		hq_post = hq_post.Prob(h_0_mgrid.flatten(), q_0_mgrid.flatten())
		hq_post = hq_post.reshape(n_2d_grid, n_2d_grid)
		#mp.contour(h_0_grid_2d, q_0_grid_2d, hq_post, \
		#		   levels=[np.exp(-2.0), np.exp(-0.5)], \
		#		   colors=[cols[i]], linestyles=['dashed', 'solid'], \
		#		   alpha=0.6)
		#mp.contourf(h_0_grid_2d, q_0_grid_2d, hq_post, \
		#		   levels=[np.exp(-2.0), np.exp(-0.5), 1.0], \
		#		   colors=[cm(0.9),cm(0.4)], alpha=0.2)
		ax2d.contourf(h_0_grid_2d, q_0_grid_2d, hq_post, \
					  levels=[np.exp(-2.0), np.exp(-0.5)], \
					  colors=[cm(0.5)], alpha=0.1, zorder=1, \
					  linestyles=None)
		ax2d.contourf(h_0_grid_2d, q_0_grid_2d, hq_post, \
					  levels=[np.exp(-0.5), 1.0], \
					  colors=[cm(0.3)], alpha=0.1, zorder=2, \
					  linestyles=None)
	ax2d.set_xlabel(r'$H_0\,{\rm (km/s/Mpc)}$', fontsize=20)
	ax2d.set_ylabel(r'$q_0$', fontsize=20)
	fname = base + '_h_0_q_0_posts_' + '{:d}'.format(n_rpt) + '_rpts.pdf'
	fig2d.savefig(fname, bbox_inches='tight')
	print 'saved', fname

# plot H_0 summaries
h_0_bins = np.linspace(np.min(summaries[0, :]), np.max(summaries[0, :]), 11)
std_bins = np.linspace(np.min(summaries[2, :]), np.max(summaries[2, :]), 11)
std_bins = np.linspace(0.7, 1.5, 11)
pretty_hist(summaries[0, :], h_0_bins, axes[1], cm(0.2), \
			density=True, fill=False)
axes[1].axvline(h_0_true, ls='--', color='grey')
axes[1].set_xlim(h_0_min, h_0_max)
pretty_hist(summaries[2, :], std_bins, axes[2], cm(0.2), \
			density=True, fill=False)
axes[0].set_xlim(63.0, 77.0)
axes[0].set_ylim(0.0, axes[0].get_ylim()[1])
if ntlo:
	axes[2].set_xlim(1.0, 1.4)
else:
	axes[2].set_xlim(0.75, 1.2)
axes[2].set_xlim(0.7, 1.5)
axes[0].set_xlabel(r'$H_0\,{\rm (km/s/Mpc)}$', fontsize=20)
axes[1].set_xlabel(r'$\hat{H}_{0,{\rm MAP}}\,{\rm (km/s/Mpc)}$', fontsize=20)
axes[2].set_xlabel(r'$\sigma_{H_0}\,{\rm (km/s/Mpc)}$', fontsize=20)
axes[0].set_ylabel(r'${\rm P}(H_0|{\rm sample})$', fontsize=20)
axes[1].set_ylabel(r'${\rm P}(\hat{H}_{0,{\rm MAP}}|{\rm sample})$', fontsize=20)
axes[2].set_ylabel(r'${\rm P}(\sigma_{H_0}|{\rm sample})$', fontsize=20)
for i in range(3):
	axes[i].tick_params(axis='both', which='major', labelsize=16)
#axes[2].locator_params(nbins=6)
fig.subplots_adjust(hspace=0, wspace=0.275)
fname = base + '_h_0_summaries_' + '{:d}'.format(n_rpt) + '_rpts.pdf'
fig.savefig(fname, bbox_inches='tight')
print 'saved', fname
