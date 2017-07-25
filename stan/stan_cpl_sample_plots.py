import numpy as np
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import pystan as ps
import pickle
import scipy.stats as sps
import scipy.special as spesh
import getdist as gd
import getdist.plots as gdp
import fnmatch
import os

# plotting settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')
cols = [cm(0.15), cm(0.85), cm(0.5)]

# covariance function
def cov(d1, d2):
    n = 0
    mean1 = 0
    mean2 = 0
    cov = 0
    for x, y in zip(d1, d2):
        n += 1
        delta1 = (x - mean1) / n
        mean1 += delta1
        delta2 = (y - mean2) / n
        mean2 += delta2
        cov += (n - 1) * delta1 * delta2 - cov / n
    return n / (n - 1) * cov

def sample_plot(name, latex, value, base, limits = None, \
                logplot = False):

    # trace plots
    mp.rcParams["figure.figsize"] = [24, 10]
    fig, axes = mp.subplots(2, 3)
    n_chains = len(samples)
    n_samples = samples[0].shape[0]
    n_warmup = n_samples / 2
    if limits is None:
        mins = []
        maxes = []
        for i in range(0, n_chains):
            mins.append(np.min(samples[i][name][n_warmup:]))
            maxes.append(np.max(samples[i][name][n_warmup:]))
        limits = [np.min(mins), np.max(maxes)]
    for i in range(0, n_chains):
        j = (i / 2, np.mod(i, 2))
        axes[j].plot(samples[i][name][n_warmup:])
        axes[j].set_xlabel(r'chain {:d} iteration'.format(i + 1))
        axes[j].set_ylabel('$' + latex + '$')
        axes[j].set_ylim(limits[0], limits[1])

    # autocorrelation
    samples_par = samples[0][name][n_warmup:]
    for i in range(1, n_chains):
        samples_par = np.append(samples_par, \
                                samples[i][name][n_warmup:])
    max_lag = 100
    ac = np.zeros(max_lag + 1)
    ac[0] = cov(samples_par, samples_par)
    for i in range(1, max_lag + 1):
        ac[i] = cov(samples_par[i:], samples_par[:-i])
    axes[(0, 2)].plot(ac / ac[0])
    axes[(0, 2)].set_xlabel(r'lag $l$')
    axes[(0, 2)].set_ylabel(r'$\rho(l)$')

    # posterior
    kde_par = sps.gaussian_kde(samples_par)
    if logplot:
        par_grid = np.logspace(np.log10(limits[0]) - 0.5, \
                               np.log10(limits[1]) + 0.5, 1000)
        axes[(1, 2)].semilogx(par_grid, \
                              kde_par.evaluate(par_grid), 'b')
        axes[(1, 2)].set_xlim(10.0 ** (np.log10(limits[0]) - 0.5), \
                              10.0 ** (np.log10(limits[1]) + 0.5))
    else:
        par_grid = np.linspace(limits[0], limits[1], 1000)
        axes[(1, 2)].plot(par_grid, kde_par.evaluate(par_grid), 'b')
    if value is not None:
        axes[(1, 2)].axvline(value, color='black', ls='--')
    axes[(1, 2)].set_xlabel('$' + latex + '$')
    axes[(1, 2)].set_ylabel(r'$P(' + latex + ')$')
    mp.savefig(base + '_' + name + '_diagnostics.pdf', \
               bbox_inches = 'tight')

# convert Student's T degrees of freedom to ratio of peak height to
# corresponding Gaussian
def nu2phr(nu):
    phr = np.sqrt(2.0 / nu) * spesh.gamma((nu + 1.0) / 2.0) / \
          spesh.gamma(nu / 2.0)
    phr = sps.t.pdf(0.0, nu) / sps.norm.pdf(0.0)
    return phr

# read in data
sne_sum = False
gauss_mu_like = False
fix_redshifts = False
model_outliers = None # None, "gmm" or "ht"
plot_log_dofs = False
plot_peak_height = True
plot_d_anc = False
inc_met_dep = True
ng_maser_pdf = False
nir_sne = False
fit_cosmo_delta = None # None, 'h', 'hq'
sim = True
oplot_r16_results = False
oplot_pla = False
oplot_hists = False
plot_diagnostics = False
if sne_sum:
    base = 'stan_hh0_sne_sum'
    if gauss_mu_like:
        base += '_gauss_mu_like'
else:
    base = 'stan_hh0'
    if fit_cosmo_delta is not None:
        base += '_delta_' + fit_cosmo_delta
    if fix_redshifts:
        base += '_fixed_z'
    if model_outliers:
        base += '_' + model_outliers + '_outliers'
    if nir_sne:
        base += '_nir_sne'
if inc_met_dep:
    base += '_metals'
if ng_maser_pdf:
    base += '_ng_maser'
rfit_res = np.genfromtxt(base + '_inversion_results.csv')
samples = []
for file in os.listdir("."):
    if fnmatch.fnmatch(file, base + "_minimal_chain_*.csv"):
        print "reading " + file
        d = np.genfromtxt(file, delimiter = ",", names = True, \
                          skip_header = 4)
        samples.append(d)
n_chains = len(samples)
n_samples = samples[0].shape[0]
n_warmup = n_samples / 2
r16_h_0 = 73.24
r16_sig_h_0 = 1.74
#planck_h_0 = 66.93
#planck_sig_h_0 = 0.62
planck_h_0 = 67.81
planck_sig_h_0 = 0.92

# optionally calculate Planck posteriors
if oplot_pla:
    
    script = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(script))
    pla_dir = os.path.join(parent_dir, 'data/')
    pla_samples = gd.loadMCSamples(pla_dir + \
                                   'base_plikHM_TT_lowTEB_lensing')
    pla_om_l = pla_samples.getParams().omegal
    pla_samples.addDerived((1.0 - 3.0 * pla_om_l) / 2.0, \
                           name = 'q_0', label = 'q_0')
    # @TODO: can we change name? this is hacky
    pla_samples.addDerived(pla_samples.getParams().H0, \
                           name = 'h_0', label = 'H_0')
    pla_samples.updateBaseStatistics()

# retrieve and process results of Riess et al. method if desired
if oplot_r16_results:

    # retrieve Riess et al. means and covariance and convert to 
    # GetDist samples
    rfit_n_dim = rfit_res.shape[1]
    rfit = rfit_res[0, :]
    rfit_cov = rfit_res[1:, :]
    rfit_n_samples = 1000000
    rfit_samples = np.random.multivariate_normal(rfit, rfit_cov, \
                                                 size = rfit_n_samples)
    if sne_sum:
        rfit_pars = ['abs_mag_c_std', 'slope_p', 'slope_z', \
                     'abs_mag_s_std', 'a_x', 'log_h_0', 'h_0']
        rfit_par_names = [r'M^{\odot}', r's^p', r's^Z', r'M^{\rm s}', \
                          r'a_x', r'\log_{10}H_0', \
                          r'H_0 [{\rm kms}^{-1}{\rm Mpc}^{-1}]']
        if sim:
            rfit_par_vals = [-3.0869, -3.05, -0.25, -19.2, 0.71273, None, 71.1]
            rfit_par_vals = [-3.0869, -3.05, -0.25, -19.2, 0.71273, None, 74.6]
        else:
            #rfit_par_vals = [None, -3.23, -0.15, -19.3, None, None, 71.57]
            rfit_par_vals = [None, -3.23, -0.15, -19.3, None, None, r16_h_0]
        rfit_gd_samples = gd.MCSamples(samples = rfit_samples, \
                                       names = rfit_pars[0: -2], \
                                       labels = rfit_par_names[0: -2])
    else:
        rfit_pars = ['abs_mag_c_std', 'slope_p', 'slope_z', \
                     'abs_mag_s_std', 'a_x', 'h_0']
        rfit_par_names = [r'M^{\odot}', r's^p', r's^Z', r'M^{\rm s}', \
                          r'a_x', r'H_0 [{\rm kms}^{-1}{\rm Mpc}^{-1}]']
        if sim:
            #rfit_par_vals = [-3.0869, -3.05, -0.25, -19.2, 0.71273, 71.1]
            rfit_par_vals = [None, -3.23, -0.25, -19.3, None, 71.1]
        else:
            #rfit_par_vals = [None, -3.23, -0.15, -19.3, None, r16_h_0]
            rfit_par_vals = [None, -3.23, -0.15, -19.3, None, None]
        rfit_gd_samples = gd.MCSamples(samples = rfit_samples, \
                                       names = rfit_pars[0: -1], \
                                       labels = rfit_par_names[0: -1])
    rfit_h_0_sample = (rfit_gd_samples.getParams().abs_mag_s_std + \
                       5.0 * rfit_gd_samples.getParams().a_x + 25.0) / 5.0
    rfit_gd_samples.addDerived(10.0 ** rfit_h_0_sample, \
                               name = rfit_pars[-1], \
                               label = rfit_par_names[-1])
    rfit_gd_samples.updateBaseStatistics()

    # also analytically convert to expected distributions
    i_m_s_std = rfit_pars.index('abs_mag_s_std')
    i_a_x = rfit_pars.index('a_x')
    rfit_log_h_0 = (rfit[i_m_s_std] + 5.0 * rfit[i_a_x] + 25.0) / 5.0
    rfit_h_0 = 10.0 ** (rfit_log_h_0)
    rfit_sig_log_h_0 = np.sqrt(rfit_cov[i_m_s_std, i_m_s_std] / 25.0 + \
                               rfit_cov[i_a_x, i_a_x])
    rfit_sig_h_0 = rfit_h_0 * np.log(10.0) * rfit_sig_log_h_0

    # set up BHM parameter names, labels, etc.
    if sne_sum:
        pars = rfit_pars[:i_a_x] + rfit_pars[i_a_x+1:]
        par_names = rfit_par_names[:i_a_x] + rfit_par_names[i_a_x+1:]
        par_vals = rfit_par_vals[:i_a_x] + rfit_par_vals[i_a_x+1:]
        par_ranges = {}
    else:
        pars = rfit_pars
        par_names = rfit_par_names
        par_vals = rfit_par_vals
        pars[i_a_x] = 'q_0'
        par_names[i_a_x] = r'q_0'
        if sim:
            par_vals[i_a_x] = -0.5575
        else:
            par_vals[i_a_x] = None
        par_ranges = {}

else:

    # BHM parameter names and labels
    if nir_sne:
        pars = ['abs_mag_c_std', 'slope_p', 'slope_z', \
                'abs_mag_s_std', 'q_0', 'h_0']
        par_names = [r'M^{\odot}', r's^p', r's^Z', r'M^{\rm s}', \
                     r'q_0', r'H_0 [{\rm kms}^{-1}{\rm Mpc}^{-1}]']
        if sim:
            par_vals = [-3.08685672935, -3.05, -0.25, -18.5240, \
                        -0.5575, 72.78]
        else:
            par_vals = [None, -3.23, -0.15, -19.3, None, \
                        None]
    else:
        pars = ['abs_mag_c_std', 'slope_p', 'slope_z', \
                'abs_mag_s_std', 'alpha_s', 'beta_s', 'q_0', 'h_0']
        par_names = [r'M^{\odot}', r's^p', r's^Z', r'M^{\rm s}', \
                     r'\alpha^{\rm s}', r'\beta^{\rm s}', r'q_0', \
                     r'H_0 [{\rm kms}^{-1}{\rm Mpc}^{-1}]']
        if sim:
            par_vals = [-3.08685672935, -3.05, -0.25, -19.2, -0.14, \
                        3.1, -0.5575, 71.1]
        else:
            #par_vals = [None, -3.23, -0.15, -19.3, None, None, 71.57]
            #par_vals = [None, -3.23, -0.15, -19.3, None, None, None, \
            #            r16_h_0]
            par_vals = [None, -3.23, -0.15, -19.3, None, None, None, \
                        None]

# adjust params for BHM samples
if model_outliers == "gmm":
    for n, i in enumerate(pars):
        if i == 'abs_mag_c_std':
            pars[n] = 'intcpt_mm_c1'
        if i == 'abs_mag_s_std':
            pars[n] = 'intcpt_mm_s1'
    pars.append('f_mm_c1')
    pars.append('f_mm_c2')
    pars.append('intcpt_mm_c2')
    pars.append('sig_mm_c1')
    pars.append('sig_mm_c2')
    pars.append('f_mm_s1')
    pars.append('f_mm_s2')
    pars.append('intcpt_mm_s2')
    pars.append('sig_mm_s1')
    pars.append('sig_mm_s2')
    par_names.append(r'f^{\rm c}')
    par_names.append(r'f^{\rm co}')
    par_names.append(r'zp^{\rm co}')
    par_names.append(r'\sigma^{\rm int,\,c}')
    par_names.append(r'\sigma^{\rm int,\,co}')
    par_names.append(r'f^{\rm s}')
    par_names.append(r'f^{\rm so}')
    par_names.append(r'zp^{\rm so}')
    par_names.append(r'\sigma^{\rm int,\,s}')
    par_names.append(r'\sigma^{\rm int,\,so}')
    par_vals.append(0.8)
    par_vals.append(0.2)
    par_vals.append(-3.08685672935)
    par_vals.append(0.065)
    par_vals.append(1.0)
    par_vals.append(0.8)
    par_vals.append(0.2)
    par_vals.append(-19.2)
    par_vals.append(0.1)
    par_vals.append(1.0)
    par_ranges = {'f_mm_c1': [0, 1], 'f_mm_c2': [0, 1], \
                  'f_mm_s1': [0, 1], 'f_mm_s2': [0, 1], \
                  'sig_mm_c1': [0.01, 3.0], 'sig_mm_s1': [0.01, 3.0]}
elif model_outliers == "ht":
    pars.append('sig_int_c')
    pars.append('nu_c')
    pars.append('sig_int_s')
    pars.append('nu_s')
    par_names.append(r'\sigma^{\rm int,\,c}')
    par_names.append(r'\nu^{\rm c}')
    par_names.append(r'\sigma^{\rm int,\,s}')
    par_names.append(r'\nu^{\rm s}')
    par_vals.append(0.065)
    if sim:
        par_vals.append(2.0)
    else:
        par_vals.append(None)
    par_vals.append(0.1)
    if sim:
        par_vals.append(2.0)
    else:
        par_vals.append(None)
    par_ranges = {'sig_int_c': [0.01, 3.0], \
                  'nu_c': [0.0, None], \
                  'sig_int_s': [0.01, 3.0], \
                  'nu_s': [0.0, None]}
elif not sne_sum:
    pars.append('sig_int_c')
    pars.append('sig_int_s')
    par_names.append(r'\sigma^{\rm int,\,c}')
    par_names.append(r'\sigma^{\rm int,\,s}')
    par_vals.append(0.065)
    par_vals.append(0.1)
    par_ranges = {'sig_int_c': [0.01, 3.0], 'sig_int_s': [0.01, 3.0]}
if plot_d_anc:
    dp_anc = np.array([7.54, 49.97e-3, 2.03, 2.74, 3.26, 2.30, 3.17, \
                        2.13, 3.71, 2.64, 2.06, 2.31, \
                        2.57, 2.23, 2.19, 0.428, 0.348])
    n_anc = 0
    sample_names = samples[0].dtype.names
    for sample_name in sample_names:
        if 'true_d_anc' in sample_name:
            n_anc += 1
    for i in range(0, n_anc):
        pars.append('true_d_anc{:d}'.format(i + 1))
        par_names.append(r'd_{' + '{:d}'.format(i) + r'}')
        par_vals.append(dp_anc[i])
        '''pars.append('true_mu_h{:d}'.format(i + 1))
        par_names.append(r'\mu_{' + '{:d}'.format(i) + r'}')
        par_vals.append(None)'''
if fit_cosmo_delta is not None:
    pars.append('delta_h_0')
    par_names.append(r'\Delta H_0 [{\rm kms}^{-1}{\rm Mpc}^{-1}]')
    i_h_0 = pars.index('h_0')
    if par_vals[i_h_0] is None:
        par_vals.append(None)
    else:
        par_vals.append(par_vals[i_h_0] - 67.81)
    if fit_cosmo_delta == 'hq':
        pars.append('delta_q_0')
        par_names.append(r'\Delta q_0')
        i_q_0 = pars.index('q_0')
        if par_vals[i_q_0] is None:
            par_vals.append(None)
        else:
            par_vals.append(par_vals[i_q_0] - -0.5381)

# GetDist triangle plot
n_thin = 1
n_pars = len(pars)
np_samples = np.zeros((n_chains * n_warmup / n_thin, n_pars))
for i in range(0, n_chains):
    for j in range(0, n_pars):
        np_samples[i * n_warmup / n_thin: (i + 1) * n_warmup / n_thin, j] = \
            samples[i][pars[j]][n_warmup::n_thin]
gd_samples = gd.MCSamples(samples = np_samples, names = pars, 
                          labels = par_names, ranges = par_ranges)
if model_outliers == 'ht' and plot_log_dofs:

    # may need to do this for Gaussian distributions with Pareto
    # prior on heavy-tail distribution DOFs, as GetDist sometimes 
    # spits out an error
    gd_samples.addDerived(np.log10(gd_samples.getParams().nu_c), \
                          name='log_nu_c', label=r'\log_{10}\nu_{\rm c}')
    gd_samples.addDerived(np.log10(gd_samples.getParams().nu_s), \
                          name='log_nu_s', label=r'\log_{10}\nu_{\rm s}')
    gd_samples.setRanges({'log_nu_c':(-1,None), 'log_nu_s':(-1,None)})
    gd_samples.updateBaseStatistics()
    i_nu_c = pars.index('nu_c')
    i_nu_s = pars.index('nu_s')
    pars[i_nu_c] = 'log_nu_c'
    pars[i_nu_s] = 'log_nu_s'
    if sim:
        if plot_log_dofs:
            par_vals[i_nu_c] = np.log10(par_vals[i_nu_c])
            par_vals[i_nu_s] = np.log10(par_vals[i_nu_s])
        else:
            par_vals[i_nu_c] = par_vals[i_nu_c]
            par_vals[i_nu_s] = par_vals[i_nu_s]
    if sim:
        par_vals[i_nu_c] = None
        par_vals[i_nu_s] = None
if model_outliers == 'ht' and plot_peak_height:

    # plot kurtosis heuristic instead of nu
    gd_samples.addDerived(nu2phr(gd_samples.getParams().nu_c), \
                          name='phr_c', label=r't_{\rm c}')
    gd_samples.addDerived(nu2phr(gd_samples.getParams().nu_s), \
                          name='phr_s', label=r't_{\rm s}')
    gd_samples.setRanges({'phr_c':(0.0, 1.0), 'phr_s':(0.0, 1.0)})
    gd_samples.updateBaseStatistics()
    i_nu_c = pars.index('nu_c')
    i_nu_s = pars.index('nu_s')
    pars[i_nu_c] = 'phr_c'
    pars[i_nu_s] = 'phr_s'
    if sim:
        par_vals[i_nu_c] = nu2phr(par_vals[i_nu_c])
        par_vals[i_nu_s] = nu2phr(par_vals[i_nu_s])
        #par_vals[i_nu_c] = 0.998 # really 1, but this stands out
        #par_vals[i_nu_s] = 0.998 # really 1, but this stands out

g = gdp.getSubplotPlotter()
g.settings.lw_contour = lw # hacky hack hack: GD ignores user contour LWs
if oplot_pla:

    if oplot_r16_results:
        g.triangle_plot([gd_samples, rfit_gd_samples, pla_samples], pars, \
                        filled = [True, False, False], \
                        line_args = [{'lw': lw, \
                                      'color': mpc.rgb2hex(cols[0])}, \
                                     {'lw': lw, 'ls': '--', \
                                      'color': mpc.rgb2hex(cols[2])}, \
                                     {'lw': lw, 'ls': '--', \
                                      'color': mpc.rgb2hex(cols[1])}], \
                        contour_args = {'lws': [lw, lw, lw]}, \
                        legend_labels = ['BHM', 'MLE', 'Planck'], \
                        legend_loc = 'upper right', \
                        colors = [mpc.rgb2hex(cols[0]), \
                                  mpc.rgb2hex(cols[2]), \
                                  mpc.rgb2hex(cols[1])])
    else:
        g.triangle_plot([gd_samples, pla_samples], pars, \
                        filled = [True, False], \
                        line_args = [{'lw': lw, \
                                      'color': mpc.rgb2hex(cols[0])}, \
                                     {'lw': lw, 'ls': '--', \
                                      'color': mpc.rgb2hex(cols[2])}], \
                        contour_args = {'lws': [lw, lw]}, \
                        legend_labels = ['BHM', 'MLE'], \
                        legend_loc = 'upper right', \
                        colors = [mpc.rgb2hex(cols[0]), \
                                  mpc.rgb2hex(cols[2])])
elif oplot_r16_results:
    g.triangle_plot([gd_samples, rfit_gd_samples], pars, \
                    filled = [True, False], \
                    line_args = [{'lw': lw, \
                                  'color': mpc.rgb2hex(cols[0])}, \
                                 {'lw': lw, 'ls': '--', \
                                  'color': mpc.rgb2hex(cols[2])}], \
                    contour_args = {'lws': [lw, lw]}, \
                    legend_labels = ['BHM', 'MLE'], \
                    legend_loc = 'upper right', \
                    colors = [mpc.rgb2hex(cols[0]), \
                              mpc.rgb2hex(cols[2])])
else:
    g.triangle_plot(gd_samples, pars, filled = True, \
                    line_args = {'lw': lw, 'color': '#006FED'}, \
                    contour_args = {'lws': [lw, lw]})

# GetDist limits
# NB: GetDist limit labelling is confusing. an "upper limit" means
# the value of, say, x, below which y percent of the probability 
# density lies
print '== GetDist (KDE) Limits =='
lims = ['68%', '95%', '99%']
gd_stats = gd_samples.getMargeStats()
print gd_samples.getMeans()
for i in range(0, n_pars):
    print '* {:s}: '.format(pars[i])
    for j in range(0, 3):
        lim_type = gd_stats.parWithName(pars[i]).limits[j].limitType()
        if lim_type == 'two tail':
            lo_lim = gd_stats.parWithName(pars[i]).limits[j].lower
            up_lim = gd_stats.parWithName(pars[i]).limits[j].upper
            n_lo = (np_samples[:, i] < lo_lim).sum()
            n_hi = (np_samples[:, i] > up_lim).sum()
            print ' {:s}: '.format(lims[j]), \
                  '{:9.5f} -> {:9.5f}'.format(lo_lim, up_lim), \
                  '({:d} low, {:d} high)'.format(n_lo, n_hi)
        if lim_type == 'one tail upper limit':
            up_lim = gd_stats.parWithName(pars[i]).limits[j].upper
            n_hi = (np_samples[:, i] > up_lim).sum()
            print ' {:s}: < '.format(lims[j]), \
                  '{:9.5f}'.format(up_lim), \
                  '({:d} high)'.format(n_hi)
        if lim_type == 'one tail lower limit':
            lo_lim = gd_stats.parWithName(pars[i]).limits[j].lower
            n_lo = (np_samples[:, i] < lo_lim).sum()
            print ' {:s}: > '.format(lims[j]), \
                  '{:9.5f}'.format(lo_lim), \
                  '({:d} lo)'.format(n_lo)
    if par_vals[i] is not None:
        for ax in g.subplots[i, :i]:
            ax.axhline(par_vals[i], color='gray', ls='--')
        for ax in g.subplots[i:, i]:
            ax.axvline(par_vals[i], color='gray', ls='--')

    # optionally overplot Riess MLE Gaussian constraints on H_0
    '''if oplot_r16_results and pars[i] == 'h_0':
        h_0_min, h_0_max = (rfit_h_0 - 4.0 * rfit_sig_h_0, \
                            rfit_h_0 + 4.0 * rfit_sig_h_0)
        h_0_grid = np.linspace(h_0_min, h_0_max, 1000)
        h_0_gauss = np.exp(-0.5 * ((rfit_h_0 - h_0_grid) / \
                                   rfit_sig_h_0) ** 2)
        g.subplots[i, i].plot(h_0_grid, h_0_gauss, '--', \
                         color = mpc.rgb2hex(cols[2]))'''
mp.savefig(base + '_triangle_plot.pdf', bbox_inches = 'tight')

# check samples in tails
print '== GetDist Sample Counts =='
print 'NB: sample counts will be incorrect for derived parameters'
lims = np.array([0.683, 0.955, 0.997])
for i in range(0, n_pars):
    print '* {:s}: '.format(pars[i])
    for j in range(0, 3):
        n_extreme = (1.0 - lims[j]) / 2.0 * np_samples.shape[0]
        sig_n_extreme = np.sqrt(n_extreme)
        lim_type = gd_stats.parWithName(pars[i]).limits[j].limitType()
        if lim_type == 'two tail':
            lo_lim = gd_samples.confidence(pars[i], (1.0 - lims[j]) / 2.0,\
                                           upper=False)
            up_lim = gd_samples.confidence(pars[i], (1.0 - lims[j]) / 2.0,\
                                           upper=True)
            n_lo = (np_samples[:, i] < lo_lim).sum()
            n_hi = (np_samples[:, i] > up_lim).sum()
            print ' {:4.1f}%:'.format(lims[j] * 100), \
                  '{:9.5f} -> {:9.5f}'.format(lo_lim, up_lim), \
                  '({:d} low, {:d} high;'.format(n_lo, n_hi), \
                  'normal: {:d} +/- {:d})'.format(int(n_extreme), \
                                                  int(sig_n_extreme))
        if lim_type == 'one tail upper limit':
            up_lim = gd_samples.confidence(pars[i], (1.0 - lims[j]), \
                                           upper=True)
            n_hi = (np_samples[:, i] < up_lim).sum()
            print ' {:4.1f}%: < '.format(lims[j] * 100), \
                  '{:9.5f}'.format(up_lim), \
                  '({:d} high)'.format(n_hi)
        if lim_type == 'one tail lower limit':
            lo_lim = gd_samples.confidence(pars[i], (1.0 - lims[j]), \
                                           upper=False)
            n_lo = (np_samples[:, i] > lo_lim).sum()
            print ' {:4.1f}%: > '.format(lims[j] * 100), \
                  '{:9.5f}'.format(lo_lim), \
                  '({:d} lo)'.format(n_lo)
if model_outliers == 'ht':
    for par in ['nu_c', 'nu_s']:
        print '* {:s}: '.format(par)
        for j in range(0, 3):
            n_extreme = (1.0 - lims[j]) / 2.0 * np_samples.shape[0]
            sig_n_extreme = np.sqrt(n_extreme)
            lim_type = gd_stats.parWithName(par).limits[j].limitType()
            if lim_type == 'two tail':
                lo_lim = gd_samples.confidence(par, (1.0 - lims[j]) / 2.0,\
                                               upper=False)
                up_lim = gd_samples.confidence(par, (1.0 - lims[j]) / 2.0,\
                                               upper=True)
                print ' {:4.1f}%:'.format(lims[j] * 100), \
                      '{:9.5f} -> {:9.5f}'.format(lo_lim, up_lim)
            if lim_type == 'one tail upper limit':
                up_lim = gd_samples.confidence(par, (1.0 - lims[j]), \
                                               upper=True)
                print ' {:4.1f}%: < '.format(lims[j] * 100), \
                      '{:9.5f}'.format(up_lim)
            if lim_type == 'one tail lower limit':
                lo_lim = gd_samples.confidence(par, (1.0 - lims[j]), \
                                               upper=False)
                print ' {:4.1f}%: > '.format(lims[j] * 100), \
                      '{:9.5f}'.format(lo_lim)

# various (fractional) posteriors at Planck value
if oplot_r16_results:
    post_h_0_pdf = rfit_gd_samples.get1DDensity('h_0')
    post_h_0_planck = post_h_0_pdf.Prob(planck_h_0)[0]
    print 'GLS P(H_0_Planck|d)/P(H_0_mode|d) = {:e}'.format(post_h_0_planck)
post_h_0_pdf = gd_samples.get1DDensity('h_0')
post_h_0_planck = post_h_0_pdf.Prob(planck_h_0)[0]
print 'BHM P(H_0_Planck|d)/P(H_0_mode|d) = {:e}'.format(post_h_0_planck)
r16_log_h_0 = np.log10(r16_h_0)
r16_sig_log_h_0 = r16_sig_h_0 / np.log(10.0) / r16_h_0
r16_mode = 10.0 ** (r16_log_h_0 - np.log(10.0) * r16_sig_log_h_0 ** 2)
post_h_0_r16_mode = np.exp(-0.5 * \
                           ((r16_log_h_0 - np.log10(r16_mode)) / \
                            r16_sig_log_h_0) ** 2) / r16_mode
post_h_0_planck = np.exp(-0.5 * \
                         ((r16_log_h_0 - np.log10(planck_h_0)) / \
                          r16_sig_log_h_0) ** 2) / planck_h_0
print 'R16 P(H_0_Planck|d)/P(H_0_mode|d) = {:e}'.format(post_h_0_planck / \
                                                        post_h_0_r16_mode)

# single GetDist H_0 plot
g = gdp.getSinglePlotter()
g.settings.default_dash_styles = {'--': (5, 5), '-.': (4, 1, 1, 1)}
if oplot_r16_results:
    if oplot_pla:
        g.plot_1d([gd_samples, pla_samples], 'h_0', \
                  colors = [cols[0], 'gray'], ls = ['-', '-.'])
    else:
        g.plot_1d(gd_samples, 'h_0', \
                  colors = [cols[0]], ls = ['-'])
    h_0_min, h_0_max = (rfit_h_0 - 5.0 * rfit_sig_h_0, \
                        rfit_h_0 + 5.0 * rfit_sig_h_0)
    h_0_grid = np.linspace(h_0_min, h_0_max, 1000)
    h_0_gauss = np.exp(-0.5 * ((rfit_h_0 - h_0_grid) / \
                               rfit_sig_h_0) ** 2)
    h_0_ln = np.exp(-0.5 * ((rfit_log_h_0 - np.log10(h_0_grid)) / \
                             rfit_sig_log_h_0) ** 2) / h_0_grid
    h_0_ln /= np.max(h_0_ln)
    mp.plot(h_0_grid, h_0_ln, color = cols[1], linestyle = '--', \
            dashes = (5, 5))
    if sim:
        mp.plot(h_0_grid, h_0_gauss, color = cols[2], linestyle = '--', \
                dashes = (3, 3))
    mp.gca().set_yscale('log')
    if sim:
        mp.xlim(61., 81.)
        mp.xlim(63., 83.)
        #mp.xlim(50., 100.)
    else:
        mp.xlim(65., 80.)
    mp.ylim(1e-3, 1)
else:
    if oplot_pla:
        g.plot_1d([gd_samples, pla_samples], 'h_0')
    else:
        g.plot_1d(gd_samples, 'h_0')
if sim:
    if par_vals[pars.index('h_0')] is not None:
        mp.gca().axvline(par_vals[pars.index('h_0')], color='gray', ls = '--')
else:
    h_0_grid = np.linspace(h_0_min, h_0_max, 1000)
    r16_log_h_0 = np.log10(r16_h_0)
    r16_sig_log_h_0 = r16_sig_h_0 / np.log(10.0) / r16_h_0
    h_0_ln = np.exp(-0.5 * ((r16_log_h_0 - np.log10(h_0_grid)) / \
                             r16_sig_log_h_0) ** 2) / h_0_grid
    h_0_ln /= np.max(h_0_ln)
    mp.plot(h_0_grid, h_0_ln, color='gray', linestyle = '--', \
            dashes = (3, 3))
    '''h_0_gauss = np.exp(-0.5 * ((planck_h_0 - h_0_grid) / \
                               planck_sig_h_0) ** 2)
    mp.plot(h_0_grid, h_0_gauss, color='gray', linestyle = '--', \
            dashes = (4, 1, 1, 1))'''

    # optionally overplot histograms of BHM and R16 chains
    if oplot_hists:

        # process Planck base H_0 chains
        script = os.path.abspath(__file__)
        parent_dir = os.path.dirname(os.path.dirname(script))
        pla_stub = os.path.join(parent_dir, \
            'data/base_plikHM_TT_lowTEB_lensing_{:d}.txt')
        pla_samples = np.genfromtxt(pla_stub.format(1))[:, 23]
        for i in range(2, 5):
            pla_samples = np.concatenate((pla_samples, \
                np.genfromtxt(pla_stub.format(i))[:, 23]))

        # generate histograms
        n_bins = 100
        h_0_bin_edges = np.linspace(h_0_min, h_0_max, n_bins + 1)
        hist, edges = np.histogram(pla_samples, \
                                   bins = h_0_bin_edges)
        hist = hist / np.float(np.max(hist))
        bins = np.zeros(n_bins)
        for i in range(0, n_bins):
            bins[i] = (edges[i] + edges[i + 1]) / 2.0
        mp.bar(bins, hist, align = 'center', \
               width = edges[1] - edges[0], color = 'gray', \
               edgecolor = 'gray', alpha = 0.2)
        i_h_0 = pars.index('h_0')
        hist, edges = np.histogram(np_samples[:, i_h_0], \
                                   bins = h_0_bin_edges)
        hist = hist / np.float(np.max(hist))
        bins = np.zeros(n_bins)
        for i in range(0, n_bins):
            bins[i] = (edges[i] + edges[i + 1]) / 2.0
        mp.bar(bins, hist, align = 'center', \
               width = edges[1] - edges[0], color = cols[0], \
               edgecolor = cols[0], alpha = 0.15)
mp.savefig(base + '_h_0_plot.pdf', bbox_inches = 'tight')

# mini-triangle plot for heavy-tailed outliers
if model_outliers == "ht":
    if plot_log_dofs:
        inds = [ pars.index('log_nu_c'), pars.index('log_nu_s'), \
                 pars.index('h_0') ]
    elif plot_peak_height:
        inds = [ pars.index('phr_c'), pars.index('phr_s'), \
                 pars.index('h_0') ]
    else:
        inds = [ pars.index('nu_c'), pars.index('nu_s'), \
                 pars.index('h_0') ]
    g = gdp.getSubplotPlotter()
    g.settings.default_dash_styles = {'--': (5, 5), '-.': (4, 1, 1, 1)}
    g.settings.lw_contour = lw # hacky hack hack: GD ignores user contour LWs
    g.triangle_plot(gd_samples, [pars[i] for i in inds], filled = True, \
                    line_args = {'lw': lw, 'color': cols[0]}, \
                    contour_args = {'lws': [lw]}, \
                    contour_colors=[mpc.rgb2hex(cols[0])])
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 15
    if oplot_r16_results:
        '''mp.plot(h_0_grid, h_0_ln, color = cols[1], linestyle = '--', \
                dashes = (5, 5))'''
        g.subplots[-1, -1].plot(h_0_grid, h_0_gauss, color = cols[2], \
                                linestyle = '--', dashes = (3, 3))
    for i in range(0, len(inds)):
        if par_vals[inds[i]] is not None:
            for ax in g.subplots[i, :i]:
                ax.axhline(par_vals[inds[i]], color='gray', ls='--')
            for ax in g.subplots[i:, i]:
                ax.axvline(par_vals[inds[i]], color='gray', ls='--')
    mp.savefig(base + '_dofs_h_0_plot.pdf', bbox_inches = 'tight')

# mini-triangle plots for GMM outliers
if model_outliers == "gmm":

    inds = [ pars.index('f_mm_c1'), pars.index('f_mm_c2'), \
             pars.index('intcpt_mm_c1'), pars.index('intcpt_mm_c2'), \
             pars.index('sig_mm_c1'), pars.index('sig_mm_c2'), \
             pars.index('h_0') ]
    g = gdp.getSubplotPlotter()
    g.settings.default_dash_styles = {'--': (5, 5), '-.': (4, 1, 1, 1)}
    g.settings.lw_contour = lw # hacky hack hack: GD ignores user contour LWs
    g.triangle_plot(gd_samples, [pars[i] for i in inds], filled = True, \
                    line_args = {'lw': lw, 'color': cols[0]}, \
                    contour_args = {'lws': [lw]}, \
                    contour_colors=[mpc.rgb2hex(cols[0])])
    if oplot_r16_results:
        g.subplots[-1, -1].plot(h_0_grid, h_0_gauss, color = cols[2], \
                                linestyle = '--', dashes = (3, 3))
    for i in range(0, len(inds)):
        if par_vals[inds[i]] is not None:
            for ax in g.subplots[i, :i]:
                ax.axhline(par_vals[inds[i]], color='gray', ls='--')
            for ax in g.subplots[i:, i]:
                ax.axvline(par_vals[inds[i]], color='gray', ls='--')
    mp.savefig(base + '_gmm_c_h_0_plot.pdf', bbox_inches = 'tight')

    inds = [ pars.index('f_mm_s1'), pars.index('f_mm_s2'), \
             pars.index('intcpt_mm_s1'), pars.index('intcpt_mm_s2'), \
             pars.index('sig_mm_s1'), pars.index('sig_mm_s2'), \
             pars.index('h_0') ]
    g = gdp.getSubplotPlotter()
    g.settings.default_dash_styles = {'--': (5, 5), '-.': (4, 1, 1, 1)}
    g.settings.lw_contour = lw # hacky hack hack: GD ignores user contour LWs
    g.triangle_plot(gd_samples, [pars[i] for i in inds], filled = True, \
                    line_args = {'lw': lw, 'color': cols[0]}, \
                    contour_args = {'lws': [lw]}, \
                    contour_colors=[mpc.rgb2hex(cols[0])])
    if oplot_r16_results:
        g.subplots[-1, -1].plot(h_0_grid, h_0_gauss, color = cols[2], \
                                linestyle = '--', dashes = (3, 3))
    for i in range(0, len(inds)):
        if par_vals[inds[i]] is not None:
            for ax in g.subplots[i, :i]:
                ax.axhline(par_vals[inds[i]], color='gray', ls='--')
            for ax in g.subplots[i:, i]:
                ax.axvline(par_vals[inds[i]], color='gray', ls='--')
    mp.savefig(base + '_gmm_s_h_0_plot.pdf', bbox_inches = 'tight')

# outputs for extended cosmological models
if fit_cosmo_delta is not None:

    # first determine evidence ratio via SDDR!
    import scipy.stats as ss
    print '== Evidence Ratio =='
    if fit_cosmo_delta == 'h':

        # define delta_h_0 posterior and prior. getting normalized 1D
        # densities from GetDist is a little involved but is possible
        # (by normalized I mean integrating to 1)
        post_d_h_0_pdf = gd_samples.get1DDensity('delta_h_0')
        post_d_h_0_pdf.normalize(in_place=True)
        post_d_h_0 = post_d_h_0_pdf.Prob
        prior_d_h_0 = ss.norm(scale = 6).pdf
        evr = post_d_h_0(0.0)[0] / prior_d_h_0(0.0)
        print 'P(M_1|d)/P(M_2|d) = {:e}'.format(evr)

    else:

        # define joint delta_h_0-delta_q_0 posterior and priors. 
        # GetDist has a "normalized" option for 2D posteriors, 
        # apparently
        post_d_hq = gd_samples.get2DDensity('delta_h_0', \
                                            'delta_q_0', \
                                            normalized = True).Prob
        prior_d_hq = ss.multivariate_normal(cov = [[6.0 ** 2, 0.0], \
                                                   [0.0, 0.5 ** 2]]).pdf
        evr = post_d_hq(0.0, 0.0) / prior_d_hq([0.0, 0.0])
        print 'P(M_1|d)/P(M_2|d) = {:e}'.format(evr)

        # separate estimates directly from unsmoothed samples
        # first pick out sampled ranges of params of interest
        i_dh_0 = pars.index('delta_h_0')
        i_dq_0 = pars.index('delta_q_0')
        dh_0_min = np.min(np_samples[:, i_dh_0])
        dh_0_max = np.max(np_samples[:, i_dh_0])
        dq_0_min = np.min(np_samples[:, i_dq_0])
        dq_0_max = np.max(np_samples[:, i_dq_0])
        delta_dh_0 = dh_0_max - dh_0_min
        delta_dq_0 = dq_0_max - dq_0_min

        # first version of plot: bin by fractional radii
        fracs = np.linspace(0.01, 0.1, 20)
        sddr = []
        sddr_err = []
        n_sddr = []
        for frac in fracs:

            # determine distance from nested value and bin
            n_nest = 0
            for i in range(n_chains * n_warmup / n_thin):
                dist = np.sqrt((np_samples[i, i_dh_0] / delta_dh_0) ** 2 + \
                               (np_samples[i, i_dq_0] / delta_dq_0) ** 2)
                if dist < frac:
                    n_nest += 1
            patch_area = np.pi * frac * delta_dh_0 * frac * delta_dq_0

            # calculate normalized posterior in nested bin (such that
            # sum = 1), and its error
            post_dh_0_dq_0 = float(n_nest) / \
                             float(n_chains * n_warmup / n_thin) / \
                             patch_area
            err_post_dh_0_dq_0 = np.sqrt(n_nest) / \
                                 float(n_chains * n_warmup / n_thin) / \
                                 patch_area

            # calculate prior: specifically the normalized prior in 
            # the nested bin
            prior_dh_0_dq_0 = 1.0 / (2.0 * np.pi * 6.0 * 0.5)

            # calculate SDDR and retain
            n_sddr.append(n_nest)
            sddr.append(post_dh_0_dq_0 / prior_dh_0_dq_0)
            sddr_err.append(err_post_dh_0_dq_0 / prior_dh_0_dq_0)

        sddr = np.array(sddr)
        sddr_err = np.array(sddr_err)
        sddr_var = 1.0 / np.sum(1.0 / sddr_err[sddr > 0] ** 2)
        sddr_mean = np.sum(sddr[sddr > 0] / sddr_err[sddr > 0] ** 2) * \
                    sddr_var
        print np.mean(sddr)
        print 'P(M_1|d)/P(M_2|d) = ' + \
              '{:e} +/- {:e}'.format(sddr_mean, np.sqrt(sddr_var))

        # plot estimates as a function of bin count; overlay GetDist
        # estimate
        fig, ax_l = mp.subplots()
        ax_l.plot(fracs, sddr, color = cols[0])
        ax_l.fill_between(fracs, sddr - sddr_err, \
                          sddr + sddr_err, color = cols[0], \
                          alpha = 0.5)
        ax_l.axhline(sddr_mean, color = cols[0], ls = '--', \
                     dashes = (5, 5))
        ax_l.axhline(evr, color = cols[2], ls = '--', \
                     dashes = (5, 5))
        ax_l.set_xlabel(r'$r_{\rm bin,\,frac}$')
        ax_l.set_ylabel(r'${\rm Pr}(\Lambda|d)/' + \
                        r'{\rm Pr}(\bar{\Lambda}|d)$')
        ax_r = ax_l.twinx()
        ax_l.set_xlim(fracs[0], fracs[-1])
        ax_r.set_xlim(fracs[0], fracs[-1])
        ax_l.set_ylim(0.0, 0.25)
        ax_r.plot(fracs, n_sddr, color = cols[1], ls = '--', \
                  dashes = (3, 3))
        ax_r.set_ylabel('samples in bin')
        mp.savefig(base + '_sddr_plot.pdf', bbox_inches = 'tight')
        
        # second version: use different numbers of rectangular bins,
        # centering one bin on the nested value each time
        n_bins_list = range(10, 26)
        sddr = []
        sddr_err = []
        n_sddr = []
        for n_bins in n_bins_list:

            # define dh and dq bins, and find bins including the 
            # nested values of the simple model (dh = dq = 0)
            dh_0_edges = np.linspace(dh_0_min, dh_0_max, n_bins + 1)
            dq_0_edges = np.linspace(dq_0_min, dq_0_max, n_bins + 1)
            i_nest_dh_0 = np.argmin(np.abs(dh_0_edges))
            i_nest_dq_0 = np.argmin(np.abs(dq_0_edges))
            if i_nest_dh_0 == 0 or dh_0_edges[i_nest_dh_0] < 0:
                bin_nest_dh_0 = np.array([dh_0_edges[i_nest_dh_0], \
                                          dh_0_edges[i_nest_dh_0 + 1]])
            else:
                bin_nest_dh_0 = np.array([dh_0_edges[i_nest_dh_0 - 1], \
                                          dh_0_edges[i_nest_dh_0]])
            if i_nest_dq_0 == 0 or dq_0_edges[i_nest_dq_0] < 0:
                bin_nest_dq_0 = np.array([dq_0_edges[i_nest_dq_0], \
                                          dq_0_edges[i_nest_dq_0 + 1]])
            else:
                bin_nest_dq_0 = np.array([dq_0_edges[i_nest_dq_0 - 1], \
                                          dq_0_edges[i_nest_dq_0]])
            
            # shift bins so nested value always @ center of one bin
            dh_0_edges -= np.sum(bin_nest_dh_0) / 2.0
            dq_0_edges -= np.sum(bin_nest_dq_0) / 2.0
            bin_nest_dh_0 -= np.sum(bin_nest_dh_0) / 2.0
            bin_nest_dq_0 -= np.sum(bin_nest_dq_0) / 2.0

            # count number of samples in the nested bin
            n_nest = 0
            for i in range(n_chains * n_warmup / n_thin):
                if np_samples[i, i_dh_0] >= bin_nest_dh_0[0] and \
                   np_samples[i, i_dh_0] < bin_nest_dh_0[1] and \
                   np_samples[i, i_dq_0] >= bin_nest_dq_0[0] and \
                   np_samples[i, i_dq_0] < bin_nest_dq_0[1]:
                    n_nest += 1

            # count number of samples in all bins!
            n_tot = 0
            for i in range(n_chains * n_warmup / n_thin):
                if np_samples[i, i_dh_0] >= dh_0_edges[0] and \
                   np_samples[i, i_dh_0] < dh_0_edges[-1] and \
                   np_samples[i, i_dq_0] >= dq_0_edges[0] and \
                   np_samples[i, i_dq_0] < dq_0_edges[-1]:
                    n_tot += 1

            # calculate normalized posterior density in nested bin (such that
            # it integrates to 1), and its error
            post_dh_0_dq_0 = float(n_nest) / float(n_tot) / \
                             (dh_0_edges[1] - dh_0_edges[0]) / \
                             (dq_0_edges[1] - dq_0_edges[0])
            err_post_dh_0_dq_0 = np.sqrt(n_nest) / float(n_tot) / \
                                 (dh_0_edges[1] - dh_0_edges[0]) / \
                                 (dq_0_edges[1] - dq_0_edges[0])

            # calculate prior: specifically the normalized prior density in 
            # the nested bin
            prior_dh_0_dq_0 = 1.0 / (2.0 * np.pi * 6.0 * 0.5)

            # calculate SDDR and error
            n_sddr.append(n_nest)
            sddr.append(post_dh_0_dq_0 / prior_dh_0_dq_0)
            sddr_err.append(err_post_dh_0_dq_0 / prior_dh_0_dq_0)

        sddr = np.array(sddr)
        sddr_err = np.array(sddr_err)
        sddr_var = 1.0 / np.sum(1.0 / sddr_err[sddr > 0] ** 2)
        sddr_mean = np.sum(sddr[sddr > 0] / sddr_err[sddr > 0] ** 2) * \
                    sddr_var
        print np.mean(sddr)
        print 'P(M_1|d)/P(M_2|d) = ' + \
              '{:e} +/- {:e}'.format(sddr_mean, np.sqrt(sddr_var))

        # plot estimates as a function of bin count; overlay GetDist
        # estimate
        fig, ax_l = mp.subplots()
        ax_l.plot(n_bins_list, sddr, color = cols[0])
        ax_l.fill_between(n_bins_list, sddr - sddr_err, \
                          sddr + sddr_err, color = cols[0], \
                          alpha = 0.5)
        ax_l.axhline(sddr_mean, color = cols[0], ls = '--', \
                     dashes = (5, 5))
        ax_l.axhline(evr, color = cols[2], ls = '--', \
                     dashes = (5, 5))
        ax_l.set_xlabel(r'$N_{\rm bins}$')
        ax_l.set_ylabel(r'${\rm Pr}(\Lambda|d)/' + \
                        r'{\rm Pr}(\bar{\Lambda}|d)$')
        ax_r = ax_l.twinx()
        ax_l.set_xlim(n_bins_list[0], n_bins_list[-1])
        ax_r.set_xlim(n_bins_list[0], n_bins_list[-1])
        ax_l.set_ylim(0.0, 0.25)
        ax_r.plot(n_bins_list, n_sddr, color = cols[1], ls = '--', \
                  dashes = (3, 3))
        ax_r.set_ylabel('samples in bin')
        mp.savefig(base + '_sddr_plot_rect.pdf', bbox_inches = 'tight')

    # next plot the parameters of interest
    h_0_plot_min = 64
    h_0_plot_max = 80
    h_0_ticks = [68, 72, 76, 80]
    d_h_0_plot_min = -12
    d_h_0_plot_max = 2
    d_h_0_ticks = [-10, -5, 0]
    inds = [ pars.index('delta_h_0'), pars.index('h_0') ]
    plot_min = [ d_h_0_plot_min, h_0_plot_min ]
    plot_max = [ d_h_0_plot_max, h_0_plot_max ]
    ticks = [ d_h_0_ticks, h_0_ticks ]
    if fit_cosmo_delta == 'hq':
        q_0_plot_min = -1.5
        q_0_plot_max = -0.2
        q_0_ticks = [-1.2, -0.8, -0.4]
        d_q_0_plot_min = -0.3
        d_q_0_plot_max = 0.8
        d_q_0_ticks = [-0.25, 0.0, 0.25, 0.5, 0.75]
        inds = [ pars.index('delta_q_0'), pars.index('q_0') ] + inds
        plot_min = [ d_q_0_plot_min, q_0_plot_min ] + plot_min
        plot_max = [ d_q_0_plot_max, q_0_plot_max ] + plot_max
        ticks = [ d_q_0_ticks, q_0_ticks ] + ticks
    if oplot_pla:
        if oplot_r16_results:
            gdtp_samples = [gd_samples, rfit_gd_samples, pla_samples]
            gdtp_filled = [True, False, False]
            gdtp_line_args = [{'lw': lw, \
                               'color': mpc.rgb2hex(cols[0])}, \
                              {'lw': lw, 'ls': '--', \
                               'color': mpc.rgb2hex(cols[2])}, \
                              {'lw': lw, 'ls': '-.', \
                               'color': mpc.rgb2hex(cols[1])}]
            gdtp_contour_args = {'lws': [lw, lw, lw]}
            gdtp_labels = ['BHM', 'MLE', 'Planck']
            gdtp_colors = [mpc.rgb2hex(cols[0]), mpc.rgb2hex(cols[2]), \
                           mpc.rgb2hex(cols[1])]
        else:
            gdtp_samples = [gd_samples, pla_samples]
            gdtp_filled = [True, False]
            gdtp_line_args = [{'lw': lw, \
                               'color': mpc.rgb2hex(cols[0])}, \
                              {'lw': lw, 'ls': '--', \
                               'color': mpc.rgb2hex(cols[2])}]
            gdtp_contour_args = {'lws': [lw, lw]}
            gdtp_labels = ['BHM', 'MLE']
            gdtp_colors = [mpc.rgb2hex(cols[0]), mpc.rgb2hex(cols[2])]
    elif oplot_r16_results:
        gdtp_samples = [gd_samples, rfit_gd_samples]
        gdtp_filled = [True, False]
        gdtp_line_args = [{'lw': lw, 'color': mpc.rgb2hex(cols[0])}, \
                          {'lw': lw, 'ls': '--', \
                           'color': mpc.rgb2hex(cols[2])}]
        gdtp_contour_args = {'lws': [lw, lw]}
        gdtp_labels = ['BHM', 'MLE']
        gdtp_colors = [mpc.rgb2hex(cols[0]), mpc.rgb2hex(cols[2])]
    else:
        gdtp_samples = gd_samples
        gdtp_filled = True
        gdtp_line_args = {'lw': lw, 'color': mpc.rgb2hex(cols[0])}
        gdtp_contour_args = {'lws': [lw, lw]}
        gdtp_labels = 'BHM'
        gdtp_colors = [mpc.rgb2hex(cols[0])]
    g = gdp.getSubplotPlotter()
    g.settings.lw_contour = lw # hacky hack hack: GD ignores user contour LWs
    g.settings.line_labels = False
    g.settings.default_dash_styles = {'--': (3, 3), '-.': (4, 1, 1, 1)}
    g.triangle_plot(gdtp_samples, [pars[i] for i in inds], \
                    filled = gdtp_filled, line_args = gdtp_line_args, \
                    contour_args = gdtp_contour_args, \
                    colors = gdtp_colors, settings = {'fine_bins_2D': 512})
    for i in range(0, len(inds)):
        for ax in g.subplots[i, :i]:
            if par_vals[inds[i]] is not None:
                ax.axhline(par_vals[inds[i]], color='gray', ls='--')
            ax.set_ylim(plot_min[i], plot_max[i])
            ax.set_yticks(ticks[i])
        for ax in g.subplots[i:, i]:
            if par_vals[inds[i]] is not None:
                ax.axvline(par_vals[inds[i]], color='gray', ls='--')
            ax.set_xlim(plot_min[i], plot_max[i])
            ax.set_xticks(ticks[i])
        '''if pars[inds[i]] == 'delta_h_0':
            d_h_0_grid = np.linspace(d_h_0_plot_min, \
                                     d_h_0_plot_max, 1000)
            g.subplots[i, i].plot(d_h_0_grid, \
                                  prior_d_h_0(d_h_0_grid) * \
                                  post_d_h_0_norm / prior_d_h_0_norm, \
                                  color = 'gray', ls = ':')
        if pars[inds[i]] == 'delta_q_0':
            d_q_0_grid = np.linspace(d_q_0_plot_min, \
                                     d_q_0_plot_max, 1000)
            g.subplots[i, i].plot(d_q_0_grid, \
                                  prior_d_q_0(d_q_0_grid) * \
                                  post_d_q_0_norm / prior_d_q_0_norm, \
                                  color = 'gray', ls = ':')'''
    if fit_cosmo_delta == 'hq':
        mp.savefig(base + '_delta_h_0_q_0_plot.pdf', \
                   bbox_inches = 'tight')
    else:
        mp.savefig(base + '_delta_h_0_plot.pdf', bbox_inches = 'tight')

# optionally call plotting code for individual parameters
if plot_diagnostics:
    for i in range(0, n_pars):
        sample_plot(pars[i], par_names[i], par_vals[i], base)

# mini triangle plot for H_0, q_0 and dofs
h_0_plot_min = 64
h_0_plot_max = 80#81
h_0_ticks = [68, 72, 76, 80]
q_0_plot_min = -1.5
q_0_plot_max = -0.2
q_0_ticks = [-1.2, -0.8, -0.4]
if model_outliers == 'ht':
    if sim:
        inds = [ pars.index('h_0') ]
        ticks = [ h_0_ticks ]
        plot_min = [ h_0_plot_min ]
        plot_max = [ h_0_plot_max ]
    else:
        inds = [ pars.index('q_0'), pars.index('h_0') ]
        ticks = [ q_0_ticks, h_0_ticks ]
        plot_min = [ q_0_plot_min, h_0_plot_min ]
        plot_max = [ q_0_plot_max, h_0_plot_max ]
    if plot_log_dofs:
        inds = [pars.index('log_nu_c'), pars.index('log_nu_s')] + inds
        dof_plot_min = 0.1
        dof_plot_max = 1.5
        dof_ticks = [0.2, 0.6, 1.0, 1.4]
    elif plot_peak_height:
        inds = [pars.index('phr_c'), pars.index('phr_s')] + inds
        dof_plot_min = 0.8
        dof_plot_max = 1.0
        dof_ticks = [0.82, 0.86, 0.90, 0.94, 0.98]
        #dof_ticks = [0.84, 0.89, 0.94, 0.99]
        dof_ticks = [0.83, 0.88, 0.93, 0.98]
        dof_plot_min = 0.8
        dof_plot_max = 1.0
        dof_ticks = [0.8, 0.85, 0.9, 0.95]

        dof_plot_min = 0.8
        dof_plot_max = 1.0
        dof_ticks = [0.85, 0.9, 0.95, 1.0]
    else:
        inds = [pars.index('nu_c'), pars.index('nu_s')] + inds
        dof_plot_min = 0.8
        dof_plot_max = 3.2
        dof_ticks = [1.0, 1.5, 2.0, 2.5, 3.0]
    ticks = [ dof_ticks, dof_ticks ] + ticks
    plot_min = [ dof_plot_min, dof_plot_min ] + plot_min
    plot_max = [ dof_plot_max, dof_plot_max ] + plot_max
else:
    inds = [ pars.index('q_0'), pars.index('h_0') ]
    plot_min = [ q_0_plot_min, h_0_plot_min ]
    plot_max = [ q_0_plot_max, h_0_plot_max ]
    if sim:
        exit()
    else:
        ticks = [ q_0_ticks, h_0_ticks ]
if oplot_pla:
    if oplot_r16_results:
        gdtp_samples = [gd_samples, rfit_gd_samples, pla_samples]
        gdtp_filled = [True, False, False]
        gdtp_line_args = [{'lw': lw, \
                           'color': mpc.rgb2hex(cols[0])}, \
                          {'lw': lw, 'ls': '--', \
                           'color': mpc.rgb2hex(cols[2])}, \
                          {'lw': lw, 'ls': '-.', \
                           'color': mpc.rgb2hex(cols[1])}]
        gdtp_contour_args = {'lws': [lw, lw, lw]}
        gdtp_labels = ['BHM', 'MLE', 'Planck']
        gdtp_colors = [mpc.rgb2hex(cols[0]), mpc.rgb2hex(cols[2]), \
                       mpc.rgb2hex(cols[1])]
    else:
        gdtp_samples = [gd_samples, pla_samples]
        gdtp_filled = [True, False]
        gdtp_line_args = [{'lw': lw, \
                           'color': mpc.rgb2hex(cols[0])}, \
                          {'lw': lw, 'ls': '--', \
                           'color': mpc.rgb2hex(cols[2])}]
        gdtp_contour_args = {'lws': [lw, lw]}
        gdtp_labels = ['BHM', 'MLE']
        gdtp_colors = [mpc.rgb2hex(cols[0]), mpc.rgb2hex(cols[2])]
elif oplot_r16_results:
    gdtp_samples = [gd_samples, rfit_gd_samples]
    gdtp_filled = [True, False]
    gdtp_line_args = [{'lw': lw, 'color': mpc.rgb2hex(cols[0])}, \
                      {'lw': lw, 'ls': '--', \
                       'color': mpc.rgb2hex(cols[2])}]
    gdtp_contour_args = {'lws': [lw, lw]}
    gdtp_labels = ['BHM', 'MLE']
    gdtp_colors = [mpc.rgb2hex(cols[0]), mpc.rgb2hex(cols[2])]
else:
    gdtp_samples = gd_samples
    gdtp_filled = True
    gdtp_line_args = {'lw': lw, 'color': mpc.rgb2hex(cols[0])}
    gdtp_contour_args = {'lws': [lw, lw]}
    gdtp_labels = 'BHM'
    gdtp_colors = [mpc.rgb2hex(cols[0])]
g = gdp.getSubplotPlotter()
g.settings.lw_contour = lw # hacky hack hack: GD ignores user contour LWs
g.settings.line_labels = False
if model_outliers == 'ht':
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 15
g.settings.default_dash_styles = {'--': (3, 3), '-.': (4, 1, 1, 1)}
'''g.triangle_plot(gdtp_samples, [pars[i] for i in inds], \
                filled = gdtp_filled, line_args = gdtp_line_args, \
                contour_args = gdtp_contour_args, \
                legend_labels = gdtp_labels, \
                legend_loc = 'upper right', colors = gdtp_colors)'''
g.triangle_plot(gdtp_samples, [pars[i] for i in inds], \
                filled = gdtp_filled, line_args = gdtp_line_args, \
                contour_args = gdtp_contour_args, \
                colors = gdtp_colors, settings = {'fine_bins_2D': 512})
for i in range(0, len(inds)):
    for ax in g.subplots[i, :i]:
        if par_vals[inds[i]] is not None:
            ax.axhline(par_vals[inds[i]], color='gray', ls='--')
        ax.set_ylim(plot_min[i], plot_max[i])
        ax.set_yticks(ticks[i])
    for ax in g.subplots[i:, i]:
        if par_vals[inds[i]] is not None:
            ax.axvline(par_vals[inds[i]], color='gray', ls='--')
        ax.set_xlim(plot_min[i], plot_max[i])
        ax.set_xticks(ticks[i])
    '''if not sim and oplot_r16_results and pars[inds[i]] == 'q_0':
        g.subplots[i, i].axvline(-0.5575, \
                                 color = mpc.rgb2hex(cols[2]), \
                                 ls = '--', dashes = [3, 3])'''
if model_outliers == 'ht':
    if sim:
        mp.savefig(base + '_dofs_h_0_plot.pdf', bbox_inches = 'tight')
    else:
        mp.savefig(base + '_dofs_q_0_h_0_plot.pdf', bbox_inches = 'tight')
else:
    if not sim:
        mp.savefig(base + '_q_0_h_0_plot.pdf', bbox_inches = 'tight')

