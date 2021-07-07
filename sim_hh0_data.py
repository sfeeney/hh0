import numpy as np
import matplotlib.pyplot as mp
import pystan as ps
import numpy.random as npr
import pickle
import scipy.stats as sps
import scipy.special as spesh
import copy as cp
import os

c = 299792.458 # km s^-1

mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = 1.5
mp.rcParams['lines.linewidth'] = 1.5

def hh0_sim(setup="not_so_simple", fix_redshifts=True, \
            model_outliers=None, inc_met_dep=True, inc_zp_off=True, \
            constrain=True, round_data=False):
    
    """
    model_outliers = None / "gmm" / "ht"
    """

    # settings and cuts to match previous analyses
    p_c_min = 5.0 # 2.5 # lower limit ~ 2.5 but for local hosts
    p_c_max = 60.0 # tail beyond 60 but not many
    if setup == "d17":
        z_s_min = 0.011631
        z_s_max = 0.0762
    else:
        z_s_min = 0.023
        z_s_max = 0.1
    # @TODO: update metallicity stats to R16
    z_c_mean = 8.86101933216 # derived from R11. true dist bimodal!
    z_c_sigma = 0.15312221469
    ff_s_mean = np.array([ -0.2, 0.0 ])  # derived from R16
    ff_s_sigma = np.array([ 1.2, 0.89 ])
    if setup == "simple":
        n_ch_d = 1
        n_ch_p = 0
        n_ch_c = 0
        n_ch_s = 1
        n_c_ch = np.array([40, 40])
        n_c_ch = np.concatenate((40 * np.ones(n_ch_d, dtype=int), \
                                 np.ones(n_ch_p, dtype=int), \
                                 40 * np.ones(n_ch_c, dtype=int), \
                                 40 * np.ones(n_ch_s, dtype=int)))
        zp_off_mask = np.zeros(len(n_c_ch))
        n_s = 80
    elif setup == "not_so_simple":
        n_ch_d = 2
        n_ch_p = 2
        n_ch_c = 1
        n_ch_s = 1
        n_c_ch = np.concatenate((20 * np.ones(n_ch_d, dtype=int), \
                                 np.ones(n_ch_p, dtype=int), \
                                 20 * np.ones(n_ch_c, dtype=int), \
                                 20 * np.ones(n_ch_s, dtype=int)))
        zp_off_mask = np.zeros(len(n_c_ch))
        n_s = 80
    elif setup == "r11":
        n_ch_d = 1
        n_ch_p = 0
        n_ch_c = 0
        n_ch_s = 8
        n_c_ch = np.array([69, 32, 79, 29, 26, 36, 95, 39, 164])
        zp_off_mask = np.zeros(len(n_c_ch))
        n_s = 253
    elif setup == "r16_one_anc":
        # R16 single-anchor fit
        # anchor: NGC4258; cal: M31; 19 C/SN hosts
        n_ch_d = 1
        n_ch_p = 0
        n_ch_c = 1
        n_ch_s = 19
        n_c_ch = np.array([139, 372, 251, 14, 44, 32, 54, 141, 18, 63, \
                           80, 42, 16, 13, 3, 33, 25, 83, 13, 22, 28])
        zp_off_mask = np.zeros(len(n_c_ch))
        n_s = 217
        z_s_max = 0.15
    elif setup == "d17":
        # D17/R16 hybrid fit
        # anchors: NGC4258, LMC, MW C
        # cal: M31, N3021, N3370, N3982, N4639, N4038, N4536, N1015, 
        #      N1365, N3447, N7250; 
        # C/SN hosts: N1448, N1309, U9391, N5917, N5584, N3972, M101,
        #             N4424, N2442
        n_ch_d = 2
        n_ch_p = 15
        n_ch_c = 11
        n_ch_s = 9
        n_c_ch = np.concatenate((np.array([139, 775]), \
                                 np.ones(n_ch_p, dtype=int), \
                                 np.array([372, 18, 63, 16, 25, 13, \
                                           33, 14, 32, 80, 22, 54, \
                                           44, 28, 13, 83, 42, 251, \
                                           3, 141])))
        zp_off_mask = np.zeros(len(n_c_ch))
        zp_off_mask[1: n_ch_p + 2] = 1.0
        n_s = 27
    else:
        # R16 preferred fit
        # anchors: NGC4258, LMC, MW C; cal: M31; 19 C/SN hosts
        n_ch_d = 2
        n_ch_p = 15
        n_ch_c = 1
        n_ch_s = 19
        n_c_ch = np.concatenate((np.array([139, 775]), \
                                 np.ones(n_ch_p, dtype=int), \
                                 np.array([372, 251, 14, 44, 32, 54, \
                                           141, 18, 63, 80, 42, 16, \
                                           13, 3, 33, 25, 83, 13, 22, \
                                           28])))
        zp_off_mask = np.zeros(len(n_c_ch))
        zp_off_mask[1: n_ch_p + 2] = 1.0
        n_s = 217#281
        z_s_max = 0.15
    n_ch = n_ch_d + n_ch_p + n_ch_c + n_ch_s
    n_ch_g = n_ch_d + n_ch_p

    # read in Riess Cepheid data to estimate magnitude error
    # distribution (and eyeball metallicities if desired)
    riess_app_mag_err = np.zeros(569)
    riess_metals = np.zeros(569)
    pardir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pardir, 'data/Riess2.txt')) as f:
        for i, l in enumerate(f):
            if (i > 2):
                vals = [val for val in l.split()]
                riess_app_mag_err[i-3] = float(vals[7])
                riess_metals[i-3] = float(vals[11])
    sig_app_mag_c_shape = np.mean(riess_app_mag_err) ** 2 / \
                          np.var(riess_app_mag_err)
    sig_app_mag_c_scale = np.var(riess_app_mag_err) / \
                          np.mean(riess_app_mag_err)
    '''
    z_grid = np.linspace(8.0, 9.5, 1000)
    kde_z = sps.gaussian_kde(riess_metals)
    mp.plot(z_grid, kde_z.evaluate(z_grid), 'b')
    mp.xlabel(r'$\Delta\log_{10}[O/H]$')
    mp.ylabel(r'$P(\Delta\log_{10}[O/H])$')
    mp.show()
    '''

    # constants
    c = 299792.458 # km s^-1

    # dimension observable arrays:
    #  - apparent magnitudes of Cepheids in each SH0ES host
    #  - measured periods of Cepheids in each SH0ES host
    #  - apparent magnitudes of supernovae
    #  - measured redshifts of supernovae
    est_app_mag_c = np.zeros((n_ch, np.max(n_c_ch)))
    est_p_c = np.zeros((n_ch, np.max(n_c_ch)))
    est_app_mag_s_ch = np.zeros(n_ch_s)
    est_app_mag_s = np.zeros(n_s)
    est_z_s = np.zeros(n_s)

    # dimension true underlying arrays:
    #  - true absolute magnitudes of Cepheids in each SH0ES host
    #  - true periods of Cepheids in each SH0ES host
    #  - true distances of SH0ES hosts
    #  - true distances of supernovae
    true_abs_mag_c = np.zeros((n_ch, np.max(n_c_ch)))
    true_p_c = np.zeros((n_ch, np.max(n_c_ch)))
    true_z_c = np.zeros((n_ch, np.max(n_c_ch)))
    sig_app_mag_c = np.zeros((n_ch, np.max(n_c_ch)))
    io_c = np.zeros((n_ch, np.max(n_c_ch)), dtype = np.int)
    true_dis_ch = np.zeros(n_ch)
    true_dis_s = np.zeros(n_s)

    # seed random number generator if desired
    if constrain:
        npr.seed(0)

    # "sample" hyperparameters
    # LMC distance from http://www.nature.com/nature/journal/v495/n7439/full/nature11878.html
    dis_anc = np.array([7.54e6, 49.97e3])
    sig_dis_anc = np.array([np.sqrt(0.17e6 ** 2 + 0.10e6 ** 2), \
                            np.sqrt(0.19e3 ** 2 + 1.11e3 ** 2)])
    par_anc = np.array([2.03, 2.74, 3.26, 2.30, 3.17, \
                        2.13, 3.71, 2.64, 2.06, 2.31, \
                        2.57, 2.23, 2.19, 0.428, 0.348]) * 1.0e-3
    sig_par_anc = np.array([0.16, 0.12, 0.14, 0.19, 0.14, \
                            0.29, 0.12, 0.16, 0.22, 0.19, \
                            0.33, 0.30, 0.33, 0.054, 0.038]) * 1.0e-3
    par_anc_lkc = np.array([-0.05, -0.02, -0.02, -0.06, -0.02, \
                            -0.15, -0.01, -0.03, -0.09, -0.06, \
                            -0.13, -0.15, -0.18, -0.04, -0.04])
    mu_dis_anc = 5.0 * np.log10(dis_anc) - 5.0
    sig_mu_dis_anc = 5.0 / np.log(10.0) / dis_anc * sig_dis_anc
    mu_par_anc = -5.0 * np.log10(par_anc) - 5.0 - par_anc_lkc
    sig_mu_par_anc = 5.0 / np.log(10.0) / par_anc * sig_par_anc
    if n_ch_p == 0:
        dis_anc = dis_anc[0: n_ch_d] / 1.0e6
        sig_dis_anc = sig_dis_anc[0: n_ch_d] / 1.0e6
        mu_anc = mu_dis_anc[0: n_ch_d]
        sig_mu_anc = sig_mu_dis_anc[0: n_ch_d]
        par_anc_lkc = []
    else:
        dis_anc = np.concatenate([dis_anc[0: n_ch_d] / 1.0e6, \
                                  par_anc[0: n_ch_p] / 1.0e-3])
        sig_dis_anc = np.concatenate([sig_dis_anc[0: n_ch_d] / 1.0e6, \
                                      sig_par_anc[0: n_ch_p] / 1.0e-3])
        mu_anc = np.concatenate((mu_dis_anc[0: n_ch_d], \
                                 mu_par_anc[0: n_ch_p]))
        sig_mu_anc = np.concatenate((sig_mu_dis_anc[0: n_ch_d], \
                                     sig_mu_par_anc[0: n_ch_p]))
        par_anc_lkc = par_anc_lkc[0: n_ch_p]
    abs_mag_c_std = 26.3 - mu_anc[0]
    slope_p = -3.05
    slope_z = -0.25
    if "r16" in setup or setup == "not_so_simple" or setup == "d17":
        sig_app_mag_c_mean = 0.276
        sig_int_c = 0.065
        if setup == "d17":
            sig_app_mag_s_ch_mean = 0.02769
        else:
            sig_app_mag_s_ch_mean = 0.064
    else:
        sig_app_mag_c_mean = 0.3
        sig_int_c = 0.21
        sig_app_mag_s_ch_mean = 0.1
    sig_int_s = 0.1
    if setup == "d17":
        sig_z_s = 0.001
    else:
        sig_z_s = 0.00001
    sig_v_pec = 250.0 # km s^-1
    sig_z_s_tot = np.sqrt(sig_z_s ** 2 + (sig_v_pec / c) ** 2)
    if setup == "d17":
        abs_mag_s_std = -18.524
        sig_app_mag_s_mean = 0.05192
    else:
        abs_mag_s_std = -19.2
    alpha_s = -0.14
    beta_s = 3.1
    cov_s = np.array([[0.00396, 0.00186, 0.00163],
                      [0.00186, 0.06566, 0.00100],
                      [0.00163, 0.00100, 0.00123]])
    if model_outliers == "ht":
        st_nu_c = 2.0
        st_nu_s = 2.0
    elif model_outliers == "gmm":
        f_out_c = 0.3
        dmag_out_c = 0.0#0.7
        sig_out_c = 1.0
        f_out_s = 0.3
        dmag_out_s = 0.0#0.7
        sig_out_s = 1.0
    else:
        f_out_c = 0.0
        dmag_out_c = 0.0
        sig_out_c = 1.0
        f_out_s = 0.0
        dmag_out_s = 0.0
        sig_out_s = 1.0
    if setup == "d17":
        h_0 = 72.78
    else:
        h_0 = 71.10
    est_q_0 = -0.5575 # Betoule et al. 2014
    sig_q_0 =  0.0510 # Betoule et al. 2014
    j_0 = 1.0         # FIXED by assumption of flat LCDM universe
    zp_off = 0.01
    sig_zp_off = 0.03

    # distance-redshift conversion functions
    z2d_p_0 = -(1.0 - est_q_0 - 3.0 * est_q_0 ** 2 + j_0) * c / 6.0
    z2d_p_1 = (1.0 - est_q_0) * c / 2.0
    z2d_p_2 = c
    temp_0 = (3.0 * z2d_p_0 * z2d_p_2 - z2d_p_1 ** 2) / (3.0 * z2d_p_0 ** 2)
    temp_1 = (2.0 * z2d_p_1 ** 3 - 9.0 * z2d_p_0 * z2d_p_1 * z2d_p_2) / \
             (27.0 * z2d_p_0 ** 3)
    d2z_p_0 = 2.0 * np.sqrt(-temp_0 / 3.0)
    d2z_p_1 = -z2d_p_1 / (3.0 * z2d_p_0)
    d2z_p_2 = 3.0 * temp_1 / (d2z_p_0 * temp_0)
    d2z_p_3 = 3.0 / (d2z_p_0 * temp_0 * z2d_p_0)
    def d2z(d):
        phi = np.arccos(d2z_p_2 - d2z_p_3 * h_0 * d / 1.0e6)
        return d2z_p_0 * np.cos((phi + 4.0 * np.pi) / 3.0) + d2z_p_1
    def z2d(z):
        return (z2d_p_0 * z ** 3 + z2d_p_1 * z ** 2 + z2d_p_2 * z) / h_0 * 1.0e6

    # "sample" distances
    r16_cal_m31_mu = np.array([24.36])
    r16_sh0es_mu = np.array([29.135, 32.497, 32.523, \
                             31.307, 31.311, 31.511, \
                             32.498, 32.072, 31.908, \
                             31.587, 31.737, 31.290, \
                             31.080, 30.906, 31.532, \
                             31.786, 32.263, 31.499, \
                             32.919])
    if setup == "r11":
        true_mu_ch = np.concatenate((mu_anc, 
                                     np.array([30.91, 31.67, 32.13, \
                                               31.70, 32.27, 32.59, \
                                               31.72, 31.66])))
    elif setup == "r16_one_anc":
        true_mu_ch = np.concatenate((mu_anc, r16_cal_m31_mu, \
                                     r16_sh0es_mu[0: n_ch - n_ch_g]))
    elif setup == "r16":
        true_mu_ch = np.concatenate((mu_anc, r16_cal_m31_mu, r16_sh0es_mu))
    elif setup == "d17":
        ordering = [0, 1, 3, 4, 6, 7, 8, 9, 11, 12] + \
                   [10, 2, 15, 13, 5, 16, 18, 14, 17]
        true_mu_ch = np.concatenate((mu_anc, r16_cal_m31_mu, \
                                     r16_sh0es_mu[ordering]))
    else:
        #true_mu_ch = np.concatenate((mu_anc, np.array([31.83])))
        true_mu_ch = np.concatenate((mu_anc, \
                                     r16_sh0es_mu[0: n_ch - n_ch_g]))
    true_dis_ch = 10.0 ** ((true_mu_ch + 5.0) / 5.0)
    print('simulating {:d} Cepheids'.format(np.sum(n_c_ch)))

    # loop over SH0ES hosts
    i_res = 0
    res_to_plot = np.zeros(np.sum(n_c_ch))
    for i in range(0, n_ch):

        # optionally include outliers
        if model_outliers != "ht":
            outliers = npr.uniform(0.0, 1.0, n_c_ch[i]) < f_out_c
            io_c[i, 0: n_c_ch[i]] = np.array(outliers, dtype = np.int)
            sig_extra = np.ones(n_c_ch[i]) * sig_int_c
            sig_extra[outliers] = sig_out_c
            offset = np.zeros(n_c_ch[i])
            offset[outliers] = dmag_out_c

        # simulate Cepheids: uniformly distributed periods within limits of
        # P-L relation, plus intrinsic Gaussian scatter following Niccolo,
        # draw measurement errors from Gamma distribution with appropriate
        # parameters. include metallicity if desired; note that true 
        # metallicity dependence is (at least) bimodal and asymmetric
        #true_p_c[i, 0: n_c_ch[i]] = npr.uniform(p_c_min, p_c_max, n_c_ch[i])
        true_p_c[i, 0: n_c_ch[i]] = 10.0 ** npr.uniform(np.log10(p_c_min), \
                                                        np.log10(p_c_max), \
                                                        n_c_ch[i])
        true_abs_mag_c[i, 0: n_c_ch[i]] = abs_mag_c_std + \
                                          slope_p * np.log10(true_p_c[i, 0: n_c_ch[i]])
        if inc_met_dep:
            true_z_c[i, 0: n_c_ch[i]] = npr.normal(z_c_mean, z_c_sigma, \
                                                   n_c_ch[i])
            true_abs_mag_c[i, 0: n_c_ch[i]] += slope_z * \
                                               true_z_c[i, 0: n_c_ch[i]]
            est_z_c = true_z_c
        if model_outliers == "ht":
            true_abs_mag_c[i, 0: n_c_ch[i]] += npr.standard_t(st_nu_c, n_c_ch[i]) * \
                                               sig_int_c
        else:
            true_abs_mag_c[i, 0: n_c_ch[i]] += npr.normal(0.0, 1.0, n_c_ch[i]) * \
                                               sig_extra + offset
        #sig_app_mag_c[i, 0: n_c_ch[i]] = npr.gamma(sig_app_mag_c_shape, \
        #                                           sig_app_mag_c_scale, n_c_ch[i])
        sig_app_mag_c[i, 0: n_c_ch[i]] = sig_app_mag_c_mean
        est_p_c = true_p_c
        est_app_mag_c[i, 0: n_c_ch[i]] = true_abs_mag_c[i, 0: n_c_ch[i]] + \
                                         true_mu_ch[i] + \
                                         npr.normal(0.0, 1.0, n_c_ch[i]) * \
                                         sig_app_mag_c[i, 0: n_c_ch[i]]

        # plots
        '''colors = ['r' if int(j) else 'g' for j in outliers]
        mp.scatter(true_p_c[i, 0: n_c_ch[i]], \
                   true_abs_mag_c[i, 0: n_c_ch[i]] - \
                   (abs_mag_c_std + \
                    slope_p * \
                    np.log10(true_p_c[i, 0: n_c_ch[i]])), \
                   c = colors)'''
        res_to_plot[i_res: i_res + n_c_ch[i]] = true_abs_mag_c[i, 0: n_c_ch[i]] - \
                                                (abs_mag_c_std + \
                                                 slope_p * \
                                                 np.log10(true_p_c[i, 0: n_c_ch[i]]))
        i_res += n_c_ch[i]
    mp.hist(res_to_plot, bins = 30)
    mp.show()

    # simulate SH0ES SNe: already have their true distances. no 
    # intrinsic scatter in r16 sims, though there probably should be
    print('simulating {:d} supernovae'.format(n_ch_s + n_s))
    true_app_mag_s_ch = abs_mag_s_std + \
                        true_mu_ch[n_ch_g + n_ch_c:]
    if setup == "d17":
        if model_outliers == "ht":
            true_app_mag_s_ch += npr.standard_t(st_nu_s, n_ch_s) * \
                                 sig_int_s
        else:
            outliers = npr.uniform(0.0, 1.0, n_ch_s) < f_out_s
            io_ch_s = np.array(outliers, dtype = np.int)
            sig_extra = np.ones(n_ch_s) * sig_int_s
            sig_extra[outliers] = sig_out_s
            offset = np.zeros(n_ch_s)
            offset[outliers] = dmag_out_s
            true_app_mag_s_ch += npr.normal(0.0, 1.0, n_ch_s) * \
                                 sig_extra + offset
    est_app_mag_s_ch = true_app_mag_s_ch + \
                       npr.normal(0.0, sig_app_mag_s_ch_mean, n_ch_s)
    sig_app_mag_s_ch = np.ones(n_ch_s) * sig_app_mag_s_ch_mean

    # add in zero-point offset if desired
    if inc_zp_off:
        est_app_mag_s_ch += zp_off_mask[n_ch_g + n_ch_c:] * zp_off
        for i in range(0, n_ch):
            est_app_mag_c[i, 0: n_c_ch[i]] += zp_off_mask[i] * zp_off

    # simulate high-z SNe. need to sample true distances,
    # then generate observed apparent magnitudes and redshifts
    # optionally include SNe outliers
    true_z_s = npr.uniform(z_s_min, z_s_max, n_s)
    true_dis_s = z2d(true_z_s)
    true_app_mag_s = abs_mag_s_std + \
                     5.0 * np.log10(true_dis_s) - 5.0
    if setup != "d17":
        true_ff_s = npr.multivariate_normal(ff_s_mean, \
                                            np.diag(ff_s_sigma ** 2), \
                                            n_s)
        true_app_mag_s += alpha_s * true_ff_s[:, 0] + \
                          beta_s * true_ff_s[:, 1]
    if model_outliers == "ht":
        true_app_mag_s += npr.standard_t(st_nu_s, n_s) * sig_int_s
    else:
        outliers = npr.uniform(0.0, 1.0, n_s) < f_out_s
        io_s = np.array(outliers, dtype = np.int)
        sig_extra = np.ones(n_s) * sig_int_s
        sig_extra[outliers] = sig_out_s
        offset = np.zeros(n_s)
        offset[outliers] = dmag_out_s
        true_app_mag_s += npr.normal(0.0, 1.0, n_s) * sig_extra + offset
    if not fix_redshifts:
        est_z_s = true_z_s + npr.normal(0.0, sig_z_s_tot, n_s)
    else:
        est_z_s = true_z_s
    if setup == "d17":
        est_app_mag_s = true_app_mag_s + npr.normal(0.0, 1.0, n_s) * \
                        sig_app_mag_s_mean
        sig_app_mag_s = np.ones(n_s) * sig_app_mag_s_mean
        est_ff_s = np.array([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
        sig_x_1_s = None
        sig_c_s = None
        cov_x_1_app_mag_s = None
        cov_c_app_mag_s = None
        cov_x_1_c_s = None
    else:
        corr_noise_s = npr.multivariate_normal([0, 0, 0], cov_s, n_s)
        est_ff_s = true_ff_s + corr_noise_s[:, 1:]
        est_app_mag_s = true_app_mag_s + corr_noise_s[:, 0]
        sig_app_mag_s = np.ones(n_s) * np.sqrt(cov_s[0, 0])
        sig_x_1_s = np.ones(n_s) * np.sqrt(cov_s[1, 1])
        sig_c_s = np.ones(n_s) * np.sqrt(cov_s[2, 2])
        cov_x_1_app_mag_s = np.ones(n_s) * cov_s[0, 1]
        cov_c_app_mag_s = np.ones(n_s) * cov_s[0, 2]
        cov_x_1_c_s = np.ones(n_s) * cov_s[1, 2]
        print('input high-z SN parameter observation covariance:')
        print(cov_s)
        print('sample high-z SN parameter observation covariance:')
        print(np.cov(corr_noise_s.transpose()))
    res_to_plot = est_app_mag_s - \
                  (abs_mag_s_std + 5.0 * np.log10(true_dis_s) - 5.0)
    fig, axes = mp.subplots(1, 2)
    axes[0].hist(res_to_plot, bins = 30, normed = True)
    axes[1].plot(5.0 * np.log10(true_dis_s) - 5.0, est_app_mag_s, 'ro')
    if model_outliers != "ht":
        axes[1].plot(5.0 * np.log10(true_dis_s[io_s == 0]) - 5.0, \
                     est_app_mag_s[io_s == 0], 'go')
    mp.suptitle("Supernovae")
    mp.show()

    # simulate writing to R16 file and reading in
    if round_data:
        for i in range(0, n_ch):
            for j in range(0, n_c_ch[i]):
                # NB: only Cepheid data rounded for now. app mags
                #     really drawn from colour and H mag so rounding
                #     could be a little worse
                est_p_c[i, j] = np.float('{:4.4g}'.format(est_p_c[i, j]))
                est_app_mag_c[i, j] = np.float('{:4.4g}'.format(est_app_mag_c[i, j]))
                if inc_met_dep:
                    est_z_c[i, j] = np.float('{:4.4g}'.format(est_z_c[i, j]))
        for i in range(0, n_ch_s):
            est_app_mag_s_ch[i] = np.float('{:5.5g}'.format(est_app_mag_s_ch[i]))

    # sim info
    print('true abs_mag_c_std: ', abs_mag_c_std)
    print('true slope_p:       ', slope_p)
    if inc_met_dep:
        print('true slope_z:       ', slope_z)
    print('true sig_int:       ', sig_int_c)
    if model_outliers == "gmm":
        print('true f_out_c:       ', f_out_c)
        print('true dmag_out_c:    ', dmag_out_c)
        print('true sig_out_c:     ', sig_out_c)
        print('true f_out_s:       ', f_out_s)
        print('true dmag_out_s:    ', dmag_out_s)
        print('true sig_out_s:     ', sig_out_s)
    print('true abs_mag_s_std: ', abs_mag_s_std)
    print('true true_mu_h:     ', np.array_str(true_mu_ch, precision = 2))
    print('true h_0:           ', h_0)
    sim_info = {'abs_mag_c_std': abs_mag_c_std, \
                'slope_p': slope_p, 'sig_int_c': sig_int_c, \
                'abs_mag_s_std': abs_mag_s_std, \
                'sig_int_s': sig_int_s, \
                'true_mu_ch': true_mu_ch, 'h_0': h_0, 'q_0': est_q_0, \
                'alpha_s': alpha_s, 'beta_s': beta_s}
    if inc_met_dep:
        sim_info['slope_z'] = slope_z
    if model_outliers == "gmm":
        sim_info['f_out_c'] = f_out_c
        sim_info['dmag_out_c'] = dmag_out_c
        sim_info['sig_out_c'] = sig_out_c
        sim_info['f_out_s'] = f_out_s
        sim_info['dmag_out_s'] = dmag_out_s
        sim_info['sig_out_s'] = sig_out_s
    elif model_outliers == "ht":
        sim_info['st_nu_c'] = st_nu_c
        sim_info['st_nu_s'] = st_nu_s
    if inc_zp_off:
        sim_info['zp_off'] = zp_off

    # return simulated data
    to_return = [n_ch_d, n_ch_p, n_ch_c, n_ch_s, n_c_ch, n_s, \
                 dis_anc, sig_dis_anc, est_app_mag_c, \
                 sig_app_mag_c, est_p_c, sig_int_c, \
                 est_app_mag_s_ch, sig_app_mag_s_ch, est_app_mag_s, \
                 sig_app_mag_s, est_z_s, est_ff_s[:, 0], sig_x_1_s, \
                 est_ff_s[:, 1], sig_c_s, cov_x_1_app_mag_s, \
                 cov_c_app_mag_s, cov_x_1_c_s, sig_int_s, \
                 est_q_0, sig_q_0, sig_zp_off, zp_off_mask, \
                 par_anc_lkc, sim_info]
    if not fix_redshifts:
        to_return.append(np.ones(n_s) * sig_z_s)
        to_return.append(sig_v_pec)
    if inc_met_dep:
        to_return.append(est_z_c)
    return to_return

