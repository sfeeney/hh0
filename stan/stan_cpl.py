import numpy as np
import matplotlib.pyplot as mp
import matplotlib.ticker as mpt
import pystan as ps
import numpy.random as npr
import pickle
import scipy.stats as sps
import scipy.special as spesh
import copy as cp
import sim_hh0_data as sh0
import parse_hh0_data as ph0

mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = 1.5
mp.rcParams['lines.linewidth'] = 1.5

# define Riess et al fitting and rejection functions
def riess_fit(n_ch_d, n_ch_p, n_ch_c, n_ch_s, n_c_ch, app_mag_c, \
              app_mag_err_c, p_c, sig_int_c, mu_anc, sig_mu_anc, \
              zp_off_mask, sig_zp_off, app_mag_s, app_mag_err_s, \
              sig_int_s, log_z_c=None, prior_s_p=None, \
              prior_s_z=None, period_break=0.0, break_at_intcpt=False):

    # helpful parameters
    # n_obs is one magnitude per Cepheid and SN, plus one constraint
    # per anchor and one more for the zero-point offset
    # n_par is one delta-mu per distance anchor (not parallax), one 
    # mu per calibrator/C+SN host, 2+1+1 CPL params, 1 SN param and the
    # zero-point offset
    # ORDER is one mu per calibrator/C+SN host, M^c, s^p, d_zp, M^s
    # one d_mu per distance (not parallax) anchor, s^z
    # NB: prior_s_p is not integrated with period_break
    n_ch = len(n_c_ch)
    n_ch_g = n_ch_d + n_ch_p
    n_c_tot = np.sum(n_c_ch)
    n_obs = n_c_tot + n_ch_s + n_ch_d + 1
    n_par = n_ch_d + n_ch_c + n_ch_s + 4
    if log_z_c is not None:
        n_par += 1
    if prior_s_p is not None:
        n_obs += 1
    if prior_s_z is not None:
        n_obs += 1
    if period_break:
        n_par += 1
        log_period_break = np.log10(period_break)
    y_vec = np.zeros(n_obs)
    l_mat = np.zeros((n_obs, n_par))
    c_mat_inv = np.zeros((n_obs, n_obs))

    # Cepheids: build vectors and matrices
    k = 0
    for i in range(0, n_ch):
        for j in range(0, n_c_ch[i]):

            # build obs array: subtract off anchor distance moduli
            y_vec[k] = app_mag_c[i, j]
            if i < n_ch_g:
                y_vec[k] -= mu_anc[i]

            # build L matrix
            if i >= n_ch_g:
                l_mat[k, i - n_ch_g] = 1.0
            l_mat[k, n_ch - n_ch_g] = 1.0
            if period_break:
                if p_c[i, j] >= period_break:
                    l_mat[k, n_ch - n_ch_g + 1] = np.log10(p_c[i, j]) - log_period_break * int(break_at_intcpt)
            else:
                l_mat[k, n_ch - n_ch_g + 1] = np.log10(p_c[i, j])
            l_mat[k, n_ch - n_ch_g + 2] = zp_off_mask[i, j]
            if i < n_ch_d:
                l_mat[k, n_ch - n_ch_g + 3 + i] = 1.0
            if log_z_c is not None:
                l_mat[k, n_ch - n_ch_p + 4] = log_z_c[i, j]
                if period_break:
                    if p_c[i, j] < period_break:
                        l_mat[k, n_ch - n_ch_p + 5] = np.log10(p_c[i, j]) - log_period_break * int(break_at_intcpt)
            else:
                if period_break:
                    if p_c[i, j] < period_break:
                        l_mat[k, n_ch - n_ch_p + 4] = np.log10(p_c[i, j]) - log_period_break * int(break_at_intcpt)

            # build covariance matrix
            if i >= n_ch_d and i < n_ch_g:
                c_mat_inv[k, k] = 1.0 / (app_mag_err_c[i, j] ** 2 + \
                                         sig_mu_anc[i] ** 2 + \
                                         sig_int_c ** 2)
            else:
                c_mat_inv[k, k] = 1.0 / (app_mag_err_c[i, j] ** 2 + \
                                         sig_int_c ** 2)
            k += 1

    # now the supernova contributions
    for i in range(0, n_ch_s):
        y_vec[k] = app_mag_s[i]
        l_mat[k, n_ch_c + i] = 1.0
        l_mat[k, n_ch - n_ch_p + 3] = 1.0
        c_mat_inv[k, k] = 1.0 / (app_mag_err_s[i] ** 2 + sig_int_s ** 2)
        k += 1

    # and the constraints
    l_mat[k, n_ch - n_ch_g + 2] = 1.0
    c_mat_inv[k, k] = 1.0 / sig_zp_off ** 2
    k += 1
    for i in range(0, n_ch_d):
        l_mat[k, n_ch - n_ch_g + 3 + i] = 1.0
        c_mat_inv[k, k] = 1.0 / sig_mu_anc[i] ** 2
        k += 1

    # and, finally, any slope priors desired
    if prior_s_p is not None:
        y_vec[k] = prior_s_p[0]
        l_mat[k, n_ch - n_ch_g + 1] = 1.0
        c_mat_inv[k, k] = 1.0 / prior_s_p[1] ** 2
        k += 1
    if prior_s_z is not None:
        y_vec[k] = prior_s_z[0]
        l_mat[k, n_ch - n_ch_p + 4] = 1.0
        c_mat_inv[k, k] = 1.0 / prior_s_z[1] ** 2
        k += 1
        
    # fit, calculate residuals in useable form and return
    ltci = np.dot(l_mat.transpose(), c_mat_inv)
    q_hat_cov = np.linalg.inv(np.dot(ltci, l_mat))
    q_hat = np.dot(np.dot(q_hat_cov, ltci), y_vec)
    res = y_vec - np.dot(l_mat, q_hat)
    cpl_res = np.zeros(app_mag_c.shape)
    k = 0
    for i in range(0, n_ch):
        for j in range(0, n_c_ch[i]):
            cpl_res[i, j] = res[k]
            k += 1
    dof = n_obs - n_par
    chisq_dof = np.dot(res.transpose(), np.dot(c_mat_inv, res)) / dof
    return q_hat, q_hat_cov, cpl_res, chisq_dof

def riess_reject(n_c_ch, app_mag_err_c, sig_int_c, res, threshold = 2.7):

    res_scaled = np.zeros(res.shape)
    for i in range(0, len(n_c_ch)):
        res_scaled[i, 0: n_c_ch[i]] = np.abs(res[i, 0: n_c_ch[i]]) / \
                                      np.sqrt(app_mag_err_c[i, 0: n_c_ch[i]] ** 2 + 
                                              sig_int_c ** 2)
    to_rej = np.unravel_index(np.argmax(res_scaled), res.shape)
    if res_scaled[to_rej] > threshold:
        hosts = ["N4258", "N3021", "N3370", "N1309", "N3982", "N4639", \
                 "N5584", "N4038", "N4536", "N1015", "N1365", \
                 "N1448", "N3447", "N7250", "N5917", "N4424", \
                 "U9391", "N3972", "N2442", "M101"]
        print(hosts[to_rej[0]], 'Cepheid', to_rej[1] + 1, \
              'rejected')#, res_scaled[to_rej]
        return to_rej
    else:
        return None

def riess_delete(i_del, target):
    for i in range(len(target)):
        if target is not None:
            target[i][i_del[0], i_del[1]: -1] = target[i][i_del[0], i_del[1] + 1:]
            target[i][i_del[0], -1] = 0.0
    return target


# settings and cuts to match previous analyses
n_chains = 4
n_samples = 1000 # 10000
recompile = False
use_riess_rejection = False
ceph_only = True
sne_sum = False
gauss_mu_like = False
fix_redshifts = False
model_outliers = None # None, "gmm", "ht"
inc_met_dep = True
period_break = 10.0 # 0.0
break_at_intcpt = False
ng_maser_pdf = False
nir_sne = False
inc_zp_off = True
round_data = False
fit_cosmo_delta = None # None, 'h', 'hq'
v_pla = 2015 # 2015, 2016
save_full_fit = False
save_d_anc = True
save_host_mus = False
constrain = True
stan_constrain = True
setup = "rd19_one_anc"
sim = False
max_col_c = None # None or maximum V-I colour to include

# determine basis of filename
if ceph_only:
    base = 'stan_cpl'
elif sne_sum:
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

# simulate or read data
if sim:
    outputs = sh0.hh0_sim(setup = setup, \
                          fix_redshifts = fix_redshifts, \
                          model_outliers = model_outliers, \
                          inc_met_dep = inc_met_dep, \
                          inc_zp_off = inc_zp_off, \
                          constrain = constrain, \
                          round_data = round_data)
    n_ch_d, n_ch_p, n_ch_c, n_ch_s, n_c_ch, n_s, dis_anc, \
    sig_dis_anc, est_app_mag_c, sig_app_mag_c, est_p_c, sig_int_c, \
    est_app_mag_s_ch, sig_app_mag_s_ch, est_app_mag_s, \
    sig_app_mag_s, est_z_s, est_x_1_s, sig_x_1_s, est_c_s, sig_c_s, \
    cov_x_1_app_mag_s, cov_c_app_mag_s, cov_x_1_c_s, sig_int_s, \
    est_q_0, sig_q_0, sig_zp_off, zp_off_mask, par_anc_lkc, \
    sim_info = outputs[0: 31]
    if not fix_redshifts:
        sig_z_s = outputs[31]
        sig_v_pec = outputs[32]
    if inc_met_dep:
        est_z_c = outputs[-1]
else:
    outputs = ph0.hh0_parse(dataset = setup, \
                            fix_redshifts = fix_redshifts, \
                            inc_met_dep = inc_met_dep, \
                            model_outliers = model_outliers, \
                            max_col_c = max_col_c)
    n_ch_d, n_ch_p, n_ch_c, n_ch_s, n_c_ch, n_s, dis_anc, \
    sig_dis_anc, est_app_mag_c, sig_app_mag_c, est_p_c, sig_int_c, \
    est_app_mag_s_ch, sig_app_mag_s_ch, est_app_mag_s, \
    sig_app_mag_s, est_z_s, est_x_1_s, sig_x_1_s, est_c_s, sig_c_s, \
    cov_x_1_app_mag_s, cov_c_app_mag_s, cov_x_1_c_s, sig_int_s, \
    est_q_0, sig_q_0, sig_zp_off, zp_off_mask, \
    par_anc_lkc = outputs[0: 30]
    if not fix_redshifts:
        sig_z_s = outputs[30]
        sig_v_pec = outputs[31]
    if inc_met_dep:
        est_z_c = outputs[-1]
n_ch = n_ch_d + n_ch_p + n_ch_c + n_ch_s
n_ch_g = n_ch_d + n_ch_p

# if we're going to allow the cosmological expansion parameters to 
# differ from the local values (and fit for them separately), specify
# the cosmological estimates
if fit_cosmo_delta == 'h':
    if v_pla == 2015:
        est_h_0 = 67.81
        sig_h_0 = 0.92
    elif v_pla == 2016:
        est_h_0 = 66.93
        sig_h_0 = 0.62
elif fit_cosmo_delta == 'hq':
    if v_pla == 2015:
        est_h_0 = 67.81
        sig_h_0 = 0.92
        est_q_0 = -0.5381
        sig_q_0 = 0.0184
        rho = -0.99 # -0.994 probably better: -0.99 used in paper
    elif v_pla == 2016: # my rerun: 2015 low-p+TTTEEE + tau prior
        est_h_0 = 66.745
        sig_h_0 = 0.61525
        est_q_0 = -0.51552
        sig_q_0 = 0.013234
        rho = -0.994
    mu_exp_c = np.array((est_h_0, est_q_0))
    cov_exp_c = np.array(((sig_h_0 ** 2, rho * sig_h_0 * sig_q_0),\
                          (rho * sig_h_0 * sig_q_0, sig_q_0 ** 2)))
    cov_l_exp_c = np.array([[np.sqrt(cov_exp_c[0, 0]), 0.0], \
                            [cov_exp_c[0, 1] / \
                             np.sqrt(cov_exp_c[0, 0]), \
                             np.sqrt(cov_exp_c[1, 1] - \
                                     cov_exp_c[0, 1] ** 2 / \
                                     cov_exp_c[0, 0])]])

# perform Riess et al. iterative fitting
# NB: this only covers the Cepheid part of the analysis, taking 
# an independent constraint on the Hubble-flow SNe from R11. there
# aren't enough details of the fitting procedure used for those 
# SNe to reproduce, and I don't want to guess, given it's technically
# a non-linear fit to quadratic data...
# first thing is to convert distances to distance moduli, recalling
# that distances are Mpc and parallaxes are mas
if setup == "d17":
    riess_a = 2.834 / 5.0
    sig_riess_a = 0.023 / 5.0
else:
    riess_a = 0.71273
    sig_riess_a = 0.00176
mu_anc = np.zeros(n_ch_g)
sig_mu_anc = np.zeros(n_ch_g)
for i in range(0, n_ch_d):
    mu_anc[i] = 5.0 * np.log10(dis_anc[i]) + 25.0 # Mpc!
    sig_mu_anc[i] = 5.0 / np.log(10.0) * sig_dis_anc[i] / dis_anc[i]
for i in range(n_ch_d, n_ch_g):
    mu_anc[i] = -5.0 * np.log10(dis_anc[i]) + 10.0 - \
                par_anc_lkc[i - n_ch_d] # mas!
    sig_mu_anc[i] = 5.0 / np.log(10.0) * sig_dis_anc[i] / dis_anc[i]
if round_data:
    for i in range(0, n_ch_d):
        mu_anc[i] = np.float('{:5.5g}'.format(mu_anc[i]))
        sig_mu_anc[i] = np.float('{:4.4g}'.format(sig_mu_anc[i]))
    for i in range(n_ch_d, n_ch_g):
        mu_anc[i] = np.float('{:3.3g}'.format(mu_anc[i]))
        sig_mu_anc[i] = np.float('{:2.2g}'.format(sig_mu_anc[i]))
rfit_n_c_ch = cp.deepcopy(n_c_ch)
rfit_est_app_mag_c = cp.deepcopy(est_app_mag_c)
rfit_sig_app_mag_c = cp.deepcopy(sig_app_mag_c)
rfit_est_p_c = cp.deepcopy(est_p_c)
rfit_zp_off_mask = cp.deepcopy(zp_off_mask)
if inc_met_dep:
    rfit_est_z_c = cp.deepcopy(est_z_c)
else:
    rfit_est_z_c = None
n_rej = 0
while (True):
    rfit, rfit_cov, \
    rfit_res, rfit_chisq = riess_fit(n_ch_d, n_ch_p, n_ch_c, n_ch_s, \
                                     rfit_n_c_ch, rfit_est_app_mag_c, \
                                     rfit_sig_app_mag_c, rfit_est_p_c, \
                                     sig_int_c, mu_anc, sig_mu_anc, \
                                     rfit_zp_off_mask, sig_zp_off, \
                                     est_app_mag_s_ch, \
                                     sig_app_mag_s_ch, \
                                     sig_int_s, rfit_est_z_c, \
                                     period_break=period_break, \
                                     break_at_intcpt=break_at_intcpt)
    to_rej = riess_reject(rfit_n_c_ch, rfit_sig_app_mag_c, \
                          sig_int_c, rfit_res)
    if (to_rej is None):
        break
    else:
        n_rej += 1
        rfit_n_c_ch[to_rej[0]] -= 1
        rfit_est_app_mag_c, rfit_sig_app_mag_c, rfit_est_p_c, rfit_est_z_c = \
            riess_delete(to_rej, (rfit_est_app_mag_c, rfit_sig_app_mag_c, \
                                  rfit_est_p_c, rfit_est_z_c))
rfit_err = np.sqrt(np.diag(rfit_cov))

# calculate H_0 fit and its uncertainty
rfit_log_h_0 = (rfit[n_ch - n_ch_p + 3] + 5.0 * riess_a + 25.0) / 5.0
rfit_h_0 = 10.0 ** (rfit_log_h_0)
rfit_sig_log_h_0 = np.sqrt(rfit_err[n_ch - n_ch_p + 3] ** 2 / 25.0 + \
                           sig_riess_a ** 2)
rfit_sig_h_0 = rfit_h_0 * np.log(10.0) * rfit_sig_log_h_0

# report results, H_0 fit and its uncertainty
print('Riess iterative fitting: ')
print(' {:} rejected, chi^2/dof = {:5.3f}'.format(n_rej, rfit_chisq))
if sim:
    for i in range(0, n_ch_d):
        print(' mu_{0:d}: {1:7.5f} +/- {2:7.5f} ({3:7.5f})'.format(i + 1, \
            rfit[n_ch - n_ch_g + 3 + i] + sim_info['true_mu_ch'][i], \
            rfit_err[n_ch - n_ch_g + 3 + i], \
            sim_info['true_mu_ch'][i]))
    for i in range(0, n_ch_p):
        print(' mu_{0:d}: not estimated by Riess et al.'.format(i + n_ch_d + 1))
    for i in range(0, n_ch - n_ch_g):
        print(' mu_{0:d}: {1:7.5f} +/- {2:7.5f} ({3:7.5f})'.format(i + n_ch_g + 1, \
            rfit[i], rfit_err[i], sim_info['true_mu_ch'][i + n_ch_g]))
    print(' intcpt: {0:7.4f} +/- {1:7.5f} ({2:7.4f})'.format(rfit[n_ch_c + n_ch_s], \
        rfit_err[n_ch_c + n_ch_s], sim_info['abs_mag_c_std']))
    print(' slope_p: {0:8.5f} +/- {1:7.5f} ({2:8.5f})'.format(rfit[n_ch_c + n_ch_s + 1], \
        rfit_err[n_ch_c + n_ch_s + 1], sim_info['slope_p']))
    if inc_met_dep:
        print(' slope_z: {0:8.5f} +/- {1:7.5f} ({2:8.5f})'.format(rfit[n_ch_d + n_ch_c + n_ch_s + 4], \
            rfit_err[n_ch_d + n_ch_c + n_ch_s + 4], sim_info['slope_z']))
    if period_break:
        print(' slope_p_low: {0:8.5f} +/- {1:7.5f}'.format(rfit[-1], \
            rfit_err[-1], sim_info['slope_p']))
    print(' m_0^SN: {0:7.4f} +/- {1:7.5f} ({2:7.4f})'.format(rfit[n_ch - n_ch_p + 3], \
        rfit_err[n_ch - n_ch_p + 3], sim_info['abs_mag_s_std']))
    print(' zp_off: {0:7.4f} +/- {1:7.5f} ({2:7.4f})'.format(rfit[n_ch - n_ch_g + 2], \
        rfit_err[n_ch - n_ch_g + 2], sim_info['zp_off']))
    print(' H_0: {0:8.5f} +/- {1:7.5f} ({2:8.5f})'.format(rfit_h_0, \
        rfit_sig_h_0, sim_info['h_0']))
else:
    for i in range(0, n_ch_d):
        print(' mu_{0:d}: {1:7.5f} +/- {2:7.5f}'.format(i + 1, \
            rfit[n_ch - n_ch_g + 3 + i] + mu_anc[i], \
            rfit_err[n_ch - n_ch_g + 3 + i]))
    for i in range(0, n_ch_p):
        print(' mu_{0:d}: not estimated by Riess et al.'.format(i + n_ch_d + 1))
    for i in range(0, n_ch - n_ch_g):
        print(' mu_{0:d}: {1:7.5f} +/- {2:7.5f}'.format(i + n_ch_g + 1, \
            rfit[i], rfit_err[i]))
    print(' intcpt: {0:7.4f} +/- {1:7.5f}'.format(rfit[n_ch_c + n_ch_s], \
        rfit_err[n_ch_c + n_ch_s]))
    print(' slope_p: {0:8.5f} +/- {1:7.5f}'.format(rfit[n_ch_c + n_ch_s + 1], \
        rfit_err[n_ch_c + n_ch_s + 1]))
    if inc_met_dep:
        print(' slope_z: {0:8.5f} +/- {1:7.5f}'.format(rfit[n_ch_d + n_ch_c + n_ch_s + 4], \
            rfit_err[n_ch_d + n_ch_c + n_ch_s + 4]))
    if period_break:
        print(' slope_p_low: {0:8.5f} +/- {1:7.5f}'.format(rfit[-1], \
            rfit_err[-1]))
    print(' m_0^SN: {0:7.4f} +/- {1:7.5f}'.format(rfit[n_ch - n_ch_p + 3], \
        rfit_err[n_ch - n_ch_p + 3]))
    print(' zp_off: {0:7.4f} +/- {1:7.5f}'.format(rfit[n_ch - n_ch_g + 2], \
        rfit_err[n_ch - n_ch_g + 2]))
    print(' H_0: {0:8.5f} +/- {1:7.5f}'.format(rfit_h_0, \
          rfit_sig_h_0))

# save results (trimmed parameter covariance matrix) to file.
# order is: M^c, s^p, [s^Z,] M^s. append independent a_x constraint
if inc_met_dep:
    inds = [n_ch_c + n_ch_s, n_ch_c + n_ch_s + 1, -1, \
            n_ch - n_ch_p + 3]
else:
    inds = [n_ch_c + n_ch_s, n_ch_c + n_ch_s + 1, n_ch - n_ch_p + 3]
rfit_n_par_red = len(inds) + 1
rfit_results = np.zeros((rfit_n_par_red + 1, rfit_n_par_red))
rfit_results[0, 0: rfit_n_par_red - 1] = rfit[inds]
rfit_results[0, -1] = riess_a
for i in range(0, rfit_n_par_red - 1):
    for j in range(0, rfit_n_par_red - 1):
        rfit_results[i + 1, j] = rfit_cov[inds[i], inds[j]]
        rfit_results[j + 1, i] = rfit_cov[inds[i], inds[j]]
rfit_results[-1, -1] = sig_riess_a ** 2
np.savetxt(base + '_inversion_results.csv', rfit_results)

# plot rejections
if sim:
    fig, ax = mp.subplots()
    for i in range(n_ch):
        ax.set_xscale('log')
        mp.errorbar(est_p_c[i, 0: n_c_ch[i]], \
                    est_app_mag_c[i, 0: n_c_ch[i]] - \
                    sim_info['true_mu_ch'][i], \
                    yerr = sig_app_mag_c[i, 0: n_c_ch[i]], \
                    fmt='o', markerfacecolor = "white", \
                    markeredgecolor = 'k', ecolor = 'k', \
                    markeredgewidth = 1.0, zorder = 1)
        mp.semilogx(rfit_est_p_c[i, 0: rfit_n_c_ch[i]], \
                    rfit_est_app_mag_c[i, 0: rfit_n_c_ch[i]] - \
                    sim_info['true_mu_ch'][i], 'ro', zorder = 2)
        ax.invert_yaxis()
        mp.xlabel(r'$\hat{p}_{ij}^{\rm c}$ [days]', fontsize = 16)
        mp.ylabel(r'$M_{ij}^{\rm c}$ [mags]', fontsize = 16)
        mp.xlim(5.0, 60.0)
        ax.set_xticks([5, 10, 20, 40, 60])
        ax.get_xaxis().set_major_formatter(mpt.ScalarFormatter())
    mp.savefig(base + '_cepheids.pdf', bbox_inches = 'tight')
    mp.show()

# optionally use outlier-rejected data as inputs to BHM
if use_riess_rejection:
    n_c_ch = rfit_n_c_ch
    est_app_mag_c = rfit_est_app_mag_c
    sig_app_mag_c = rfit_sig_app_mag_c
    est_p_c = rfit_est_p_c
    zp_off_mask = rfit_zp_off_mask
    if inc_met_dep:
        est_z_c = rfit_est_z_c

# adjust distance anchor constraints if MASER distance should be 
# sampled from a non-Gaussian distribution (fitted as a three-
# component Gaussian mixture). the MASER should always be the first
# anchor. bit hacky for now but, you know, wevs. mixture components
# are fitted in data/maser_skew_estimate.py
if ng_maser_pdf:
    dis_anc = dis_anc[1:]
    sig_dis_anc = sig_dis_anc[1:]
    n_comp_m = 3
    amp_dis_m = [0.35071, 0.03688, 2.34142]
    mu_dis_m = [7.72166, 7.95486, 7.50804]
    sig_dis_m = [0.16624, 0.14978, 0.18508]

# compile Stan model
if recompile:
    stan_model = ps.StanModel(file = base + '.stan')
    with open(base + '_model.pkl', 'wb') as f:
        pickle.dump(stan_model, f)
else:
    try:
        with open(base + '_model.pkl', 'rb') as f:
            stan_model = pickle.load(f)
    except EnvironmentError:
        print('ERROR: pickled Stan model (' + base + '_model.pkl) not found. ' + \
              'Please set recompile = True')
        exit()

# set up Stan data
n_c_tot = np.sum(n_c_ch)
stan_c_ch = np.zeros(n_c_tot, dtype = np.int)
stan_est_app_mag_c = np.zeros(n_c_tot)
stan_sig_app_mag_c = np.zeros(n_c_tot)
stan_est_p_c = np.zeros(n_c_tot)
stan_est_z_c = np.zeros(n_c_tot)
stan_zp_off_mask = np.zeros(n_c_tot)
j = 0
for i in range(n_ch):
    stan_c_ch[j: j + n_c_ch[i]] = i + 1
    stan_est_app_mag_c[j: j + n_c_ch[i]] = est_app_mag_c[i, 0: n_c_ch[i]]
    stan_sig_app_mag_c[j: j + n_c_ch[i]] = sig_app_mag_c[i, 0: n_c_ch[i]]
    stan_est_p_c[j: j + n_c_ch[i]] = est_p_c[i, 0: n_c_ch[i]]
    if inc_met_dep:
        stan_est_z_c[j: j + n_c_ch[i]] = est_z_c[i, 0: n_c_ch[i]]
    stan_zp_off_mask[j: j + n_c_ch[i]] = zp_off_mask[i, 0: n_c_ch[i]]
    j += n_c_ch[i]
if not nir_sne:
    data_s_hi_z = np.zeros((n_s, 3))
    for i in range(0, n_s):
        data_s_hi_z[i, :] = [est_app_mag_s[i], est_x_1_s[i], est_c_s[i]]
    cov_l_s_hi_z = np.zeros((n_s, 3, 3))
    cov_l_s_hi_z[:, 0, 0] = sig_app_mag_s
    cov_l_s_hi_z[:, 1, 0] = cov_x_1_app_mag_s / cov_l_s_hi_z[:, 0, 0]
    cov_l_s_hi_z[:, 1, 1] = np.sqrt(sig_x_1_s ** 2 - \
                                    cov_l_s_hi_z[:, 1, 0] ** 2)
    cov_l_s_hi_z[:, 2, 0] = cov_c_app_mag_s / cov_l_s_hi_z[:, 0, 0]
    cov_l_s_hi_z[:, 2, 1] = (cov_x_1_c_s - \
                             cov_l_s_hi_z[:, 2, 0] * \
                             cov_l_s_hi_z[:, 1, 0]) / \
                            cov_l_s_hi_z[:, 1, 1]
    cov_l_s_hi_z[:, 2, 2] = np.sqrt(sig_c_s ** 2 - \
                                    cov_l_s_hi_z[:, 2, 0] ** 2 - \
                                    cov_l_s_hi_z[:, 2, 1] ** 2)
stan_data = {'n_ch': n_ch, 'n_ch_d': n_ch_d, 'n_ch_p': n_ch_p, \
             'n_ch_c': n_ch_c, 'n_ch_s': n_ch_s, \
             'n_c_tot': n_c_tot, 'c_ch': stan_c_ch, \
             'est_app_mag_c': stan_est_app_mag_c, \
             'sig_app_mag_c': stan_sig_app_mag_c, \
             'log_p_c': np.log10(stan_est_p_c), \
             'est_app_mag_s': est_app_mag_s_ch, \
             'sig_app_mag_s': sig_app_mag_s_ch, \
             'est_z_s_hi_z': est_z_s, \
             'sig_zp_off': sig_zp_off, \
             'zp_off_mask': stan_zp_off_mask, \
             'period_break': period_break, \
             'break_at_intcpt': int(break_at_intcpt)}
if sne_sum:
    if gauss_mu_like:
        stan_data['est_mu_anc'] = mu_anc
        stan_data['sig_mu_anc'] = sig_mu_anc
        stan_data['sig_int_c'] = sig_int_c
        stan_data['sig_int_s'] = sig_int_s
        stan_data['est_a_x'] = riess_a
        stan_data['sig_a_x'] = sig_riess_a
        stan_pars = ['abs_mag_c_std', 'slope_p', 'abs_mag_s_std', \
                     'log_h_0', 'h_0']
    else:
        stan_data['est_d_anc'] = dis_anc
        stan_data['sig_d_anc'] = sig_dis_anc
        stan_data['lk_corr'] = par_anc_lkc
        stan_data['est_a_x'] = riess_a
        stan_data['sig_a_x'] = sig_riess_a
        stan_pars = ['abs_mag_c_std', 'slope_p', 'sig_int_c', \
                     'abs_mag_s_std', 'sig_int_s', 'log_h_0', 'h_0']
else:
    stan_data['n_s_hi_z'] = n_s
    stan_data['est_d_anc'] = dis_anc
    stan_data['sig_d_anc'] = sig_dis_anc
    stan_data['est_z_s_hi_z'] = est_z_s
    if nir_sne:
        stan_data['est_app_mag_s_hi_z'] = est_app_mag_s
        stan_data['sig_app_mag_s_hi_z'] = sig_app_mag_s
    else:
        stan_data['data_s_hi_z'] = data_s_hi_z
        stan_data['cov_l_s_hi_z'] = cov_l_s_hi_z
    stan_data['est_q_0'] = est_q_0
    stan_data['sig_q_0'] = sig_q_0
    stan_data['lk_corr'] = par_anc_lkc
    if ceph_only:
        stan_pars = ['abs_mag_c_std', 'slope_p', 'zp_off']
    elif nir_sne:
        stan_pars = ['abs_mag_c_std', 'slope_p', 'zp_off', \
                     'abs_mag_s_std', 'h_0', 'q_0']
    else:
        stan_pars = ['abs_mag_c_std', 'slope_p', 'zp_off', \
                     'abs_mag_s_std', 'alpha_s', 'beta_s', \
                     'h_0', 'q_0']
    if not fix_redshifts:
        stan_data['sig_z_s_hi_z'] = np.ones(n_s) * sig_z_s
        stan_data['sig_v_pec'] = sig_v_pec
    if model_outliers == "gmm":
        stan_data['n_mm_c'] = 2
        stan_data['n_mm_s'] = 2
        stan_pars.remove('abs_mag_c_std')
        stan_pars.append('f_mm_c')
        stan_pars.append('intcpt_mm_c')
        stan_pars.append('sig_mm_c')
        if not ceph_only:
            stan_pars.remove('abs_mag_s_std')
            stan_pars.append('f_mm_s')
            stan_pars.append('intcpt_mm_s')
            stan_pars.append('sig_mm_s')
    else:
        stan_pars.append('sig_int_c')
        if not ceph_only:
            stan_pars.append('sig_int_s')
        if model_outliers == "ht":
            stan_pars.append('nu_c')
            if not ceph_only:
                stan_pars.append('nu_s')
    if fit_cosmo_delta == 'h':
        stan_pars.append('delta_h_0')
        stan_data['est_h_0'] = est_h_0
        stan_data['sig_h_0'] = sig_h_0
    elif fit_cosmo_delta == 'hq':
        stan_pars.append('delta_h_0')
        stan_pars.append('delta_q_0')
        stan_data['data_exp_c'] = mu_exp_c
        stan_data['cov_l_exp_c'] = cov_l_exp_c
if ng_maser_pdf:
    stan_data['n_comp_m'] = n_comp_m
    stan_data['amp_d_m'] = amp_dis_m
    stan_data['mu_d_m'] = mu_dis_m
    stan_data['sig_d_m'] = sig_dis_m
if inc_met_dep:
    stan_data['log_z_c'] = stan_est_z_c
    stan_pars.append('slope_z')
if period_break:
    stan_pars.append('slope_p_low')
    base = base + '_period_break_{:.1f}'.format(period_break).replace('.', 'p')
    if not break_at_intcpt:
        base = base + '_1_d_intcpt'
if save_d_anc:
    stan_pars.append('true_d_anc')
if save_host_mus:
    stan_pars.append('true_mu_h')
if stan_constrain:
    stan_seed = 23102014
else:
    stan_seed = None
if save_full_fit:
    fit = stan_model.sampling(data = stan_data, iter = n_samples, \
                              chains = n_chains, seed = stan_seed, \
                              sample_file = './' + base + '_chain')
else:
    fit = stan_model.sampling(data = stan_data, iter = n_samples, \
                              chains = n_chains, seed = stan_seed, \
                              pars = stan_pars)
    samples = fit.extract(permuted = False, inc_warmup = True)
    stan_version = ps.__version__.split('.')
    hdr_str = '# Sample generated by Stan\n'
    hdr_str += '# stan_version_major={:s}\n'.format(stan_version[0])
    hdr_str += '# stan_version_minor={:s}\n'.format(stan_version[1])
    hdr_str += '# stan_version_patch={:s}\n'.format(stan_version[2])
    if model_outliers == "gmm":
        i = stan_pars.index('f_mm_c')
        stan_pars[i] = 'f_mm_c1'
        stan_pars.insert(i + 1, 'f_mm_c2')
        i = stan_pars.index('intcpt_mm_c')
        stan_pars[i] = 'intcpt_mm_c1'
        stan_pars.insert(i + 1, 'intcpt_mm_c2')
        i = stan_pars.index('sig_mm_c')
        stan_pars[i] = 'sig_mm_c1'
        stan_pars.insert(i + 1, 'sig_mm_c2')
        i = stan_pars.index('f_mm_s')
        stan_pars[i] = 'f_mm_s1'
        stan_pars.insert(i + 1, 'f_mm_s2')
        i = stan_pars.index('intcpt_mm_s')
        stan_pars[i] = 'intcpt_mm_s1'
        stan_pars.insert(i + 1, 'intcpt_mm_s2')
        i = stan_pars.index('sig_mm_s')
        stan_pars[i] = 'sig_mm_s1'
        stan_pars.insert(i + 1, 'sig_mm_s2')
    if period_break:
        i = stan_pars.index('slope_p_low')
        stan_pars[i] = 'slope_p_low1'
    if save_d_anc:
        i = stan_pars.index('true_d_anc')
        stan_pars[i] = 'true_d_anc.1'
        for j in range(1, n_ch_d + n_ch_p):
            stan_pars.insert(i + j, 'true_d_anc.{:d}'.format(j + 1))
    if save_host_mus:
        i = stan_pars.index('true_mu_h')
        stan_pars[i] = 'true_mu_h.1'
        for j in range(1, n_ch):
            stan_pars.insert(i + j, 'true_mu_h.{:d}'.format(j + 1))
    hdr_str += 'lp__,' + ','.join(stan_pars)
    idx = len(stan_pars) + np.arange(0, len(stan_pars) + 1)
    idx = np.mod(idx, len(stan_pars) + 1)
    for i in range(0, n_chains):
        np.savetxt(base + '_minimal_chain_{:d}.csv'.format(i), \
                   samples[:, i, idx], delimiter = ',', \
                   header = hdr_str, comments = '')
print(fit)

# plot some of the fits and report percentiles
if not ceph_only:
    samples = fit.extract(permuted = True)
    print('{:6.3f}'.format(np.percentile(samples['h_0'], 50.0 - 68.27 / 2.0)) + \
          ' < H_0 < ' + \
          '{:6.3f}'.format(np.percentile(samples['h_0'], 50.0 + 68.27 / 2.0)) + \
          ' (68.3%)')
    print('{:6.3f}'.format(np.percentile(samples['h_0'], 50.0 - 95.45 / 2.0)) + \
          ' < H_0 < ' + \
          '{:6.3f}'.format(np.percentile(samples['h_0'], 50.0 + 95.45 / 2.0)) + \
          ' (95.5%)')
    print('{:6.3f}'.format(np.percentile(samples['h_0'], 50.0 - 99.73 / 2.0)) + \
          ' < H_0 < ' + \
          '{:6.3f}'.format(np.percentile(samples['h_0'], 50.0 + 99.73 / 2.0)) + \
          ' (99.7%)')
    if model_outliers:
        mp.rcParams["figure.figsize"] = [24, 5]
        fig, (ax_h_0, ax_intcpt, ax_sig) = mp.subplots(1, 3)
    else:
        fig, ax_h_0 = mp.subplots()
    h_0_grid = np.linspace(55.0, 90.0, 1000)
    kde_h_0 = sps.gaussian_kde(samples['h_0'])
    ax_h_0.plot(h_0_grid, kde_h_0.evaluate(h_0_grid), 'b')
    if sim:
        ax_h_0.plot([sim_info['h_0'], sim_info['h_0']], \
                    ax_h_0.get_ylim(), 'k--')
    rfit_h_0_pdf = np.exp(-0.5 * ((h_0_grid - rfit_h_0) / \
                                  rfit_sig_h_0) ** 2) / \
                   np.sqrt(2.0 * np.pi) / rfit_sig_h_0
    rfit_log_h_0_pdf = np.exp(-0.5 * ((np.log10(h_0_grid) - rfit_log_h_0) / \
                                      rfit_sig_log_h_0) ** 2) / \
                   np.sqrt(2.0 * np.pi) / rfit_sig_log_h_0 / h_0_grid / \
                   np.log(10.0)
    ax_h_0.plot(h_0_grid, rfit_h_0_pdf, 'r--')
    ax_h_0.plot(h_0_grid, rfit_log_h_0_pdf, 'g--')
    ax_h_0.set_xlabel(r'$H_0$')
    ax_h_0.set_ylabel(r'$P(H_0)$')
    if model_outliers == "gmm":
        intcpt_grid = np.linspace(-5.0, 0.0, 1000)
        sig_grid = np.logspace(-1.5, 0.5, 1000)
        kde_i0 = sps.gaussian_kde(samples['intcpt_mm_c'][:, 0])
        kde_i1 = sps.gaussian_kde(samples['intcpt_mm_c'][:, 1])
        kde_s0 = sps.gaussian_kde(samples['sig_mm_c'][:, 0])
        kde_s1 = sps.gaussian_kde(samples['sig_mm_c'][:, 1])
        ax_intcpt.plot(intcpt_grid, kde_i0.evaluate(intcpt_grid), 'r')
        ax_intcpt.plot(intcpt_grid, kde_i1.evaluate(intcpt_grid), 'g')
        ax_sig.semilogx(sig_grid, kde_s0.evaluate(sig_grid), 'r')
        ax_sig.semilogx(sig_grid, kde_s1.evaluate(sig_grid), 'g')
        if sim:
            ax_intcpt.plot([sim_info['abs_mag_c_std'], \
                            sim_info['abs_mag_c_std']], \
                           ax_intcpt.get_ylim(), 'k--')
            ax_sig.plot([sim_info['sig_int_c'], sim_info['sig_int_c']], \
                        ax_sig.get_ylim(), 'k--')
            ax_sig.plot([sim_info['sig_out_c'], sim_info['sig_out_c']], \
                        ax_sig.get_ylim(), 'k--')
            ax_intcpt.plot([sim_info['abs_mag_c_std'] + sim_info['dmag_out_c'], \
                            sim_info['abs_mag_c_std'] + sim_info['dmag_out_c']], \
                           ax_intcpt.get_ylim(), 'k:')
        ax_intcpt.set_xlabel(r'$M^{1\odot}$')
        ax_intcpt.set_ylabel(r'$P(M^{1\odot})$')
        ax_sig.set_xlabel(r'$\sigma_{i/o}$')
        ax_sig.set_ylabel(r'$P(\sigma_{i/o})$')
    elif model_outliers == "ht":
        intcpt_grid = np.linspace(-5.0, 0.0, 1000)
        sig_grid = np.logspace(-1.5, 0.5, 1000)
        kde_i0 = sps.gaussian_kde(samples['abs_mag_c_std'])
        kde_s0 = sps.gaussian_kde(samples['sig_int_c'])
        ax_intcpt.plot(intcpt_grid, kde_i0.evaluate(intcpt_grid), 'r')
        ax_sig.semilogx(sig_grid, kde_s0.evaluate(sig_grid), 'r')
        if sim:
            ax_intcpt.plot([sim_info['abs_mag_c_std'], sim_info['abs_mag_c_std']], \
                           ax_intcpt.get_ylim(), 'k--')
            ax_sig.plot([sim_info['sig_int_c'], sim_info['sig_int_c']], \
                        ax_sig.get_ylim(), 'k--')
        ax_intcpt.set_xlabel(r'$M^{1\odot}$')
        ax_intcpt.set_ylabel(r'$P(M^{1\odot})$')
        ax_sig.set_xlabel(r'$\sigma_{i/o}$')
        ax_sig.set_ylabel(r'$P(\sigma_{i/o})$')
    mp.savefig(base + '_constraints.pdf', bbox_inches = 'tight')
