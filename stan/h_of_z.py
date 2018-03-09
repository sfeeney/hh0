import numpy as np
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import matplotlib.patches as mpp
import matplotlib.gridspec as mpg
import os
import sys
import fnmatch
import numpy.random as npr
import sim_hh0_data as sh0
import parse_hh0_data as ph0
import pickle
import scipy.special as spesh
import scipy.stats as sps
import scipy.integrate as si
import scipy.optimize as so
import getdist as gd

######################################################################

def mu_expansion(z, h_0, q_0, j_0, abs_mag):
    # NB: this isn't really mu, as the abs mag is in there! @TODO: FIX
    z_dep = z + (1.0 - q_0) * z ** 2 / 2.0 - \
            (1.0 - q_0 - 3.0 * q_0 ** 2 + j_0) * z ** 3 / 6.0
    return 5.0 * np.log10(z_dep * c / h_0) + abs_mag + 25.0

def mu_expansion_alt(z, intcpt, q_0, j_0):
    # here, intcpt = 5 * log10(h_0) - abs_mag - 10 for h_0 in 
    # km/s/Mpc and c in km/s
    # NB: this isn't really mu, as the abs mag is in there! @TODO: FIX
    z_dep = z + (1.0 - q_0) * z ** 2 / 2.0 - \
            (1.0 - q_0 - 3.0 * q_0 ** 2 + j_0) * z ** 3 / 6.0
    return 5.0 * np.log10(z_dep * c * 1.0e3) - intcpt

def h_z_expansion(z, h_0, q_0, j_0):
    term_1 = 1.0 + (1.0 - q_0) * z - \
             (1.0 - q_0 - 3.0 * q_0 ** 2 + j_0) * z ** 2 / 2.0
    term_2 = z + (1.0 - q_0) * z ** 2 / 2.0 - \
             (1.0 - q_0 - 3.0 * q_0 ** 2 + j_0) * z ** 3 / 6.0
    return (1.0 + z) * h_0 / (term_1 - term_2 / (1.0 + z))

def h_z(z, h_0, om_l, om_m, om_r):
    return h_0 * np.sqrt(om_l + om_m * (1.0 + z) ** 3 + \
                         om_r * (1.0 + z) ** 4)

def dr_dz(z, om_m, h_0):
    #expects h_0 in km/s/Mpc; c is in km/s
    #return c / h_z(z, h_0, 1.0 - om_m, om_m, 0.0)
    return c / h_0 / \
           np.sqrt((1.0 - om_m) + om_m * (1.0 + z) ** 3)

def z2mu(z, om_m, h_0):
    # d_l here is in Mpc
    # @TODO: should check for success
    n_sn = len(z)
    d_l = np.zeros(n_sn)
    for i in range(n_sn):
       d_l[i] = (1.0 + z[i]) * \
                si.quad(dr_dz, 0.0, z[i], args=(om_m, h_0))[0]
    mu = 5.0 * np.log10(d_l) + 25.0
    return mu

def z2dc(z, om_m, h_0):
    # d_c here is in Mpc
    # @TODO: should check for success
    n_sn = len(z)
    d_c = np.zeros(n_sn)
    for i in range(n_sn):
       d_c[i] = si.quad(dr_dz, 0.0, z[i], args=(om_m, h_0))[0]
    return d_c

def z2mu_exp(z, h_0, q_0, j_0):
    z_dep = z + (1.0 - q_0) * z ** 2 / 2.0 - \
            (1.0 - q_0 - 3.0 * q_0 ** 2 + j_0) * z ** 3 / 6.0
    return 5.0 * np.log10(z_dep * c / h_0) + 25.0

def z2dc_exp(z, h_0, q_0, j_0):
    z_dep = z + (1.0 - q_0) * z ** 2 / 2.0 - \
            (1.0 - q_0 - 3.0 * q_0 ** 2 + j_0) * z ** 3 / 6.0
    return c / h_0 * z_dep / (1.0 + z)

def om_to_q_0(om_l):
    return (1.0 - 3.0 * om_l) / 2.0

def nu2phr(nu):
    phr = np.sqrt(2.0 / nu) * spesh.gamma((nu + 1.0) / 2.0) / \
          spesh.gamma(nu / 2.0)
    phr = sps.t.pdf(0.0, nu) / sps.norm.pdf(0.0)
    return phr

def approx_chi_sq(z_s, z_err_tot_s, data_s, cov_s, alpha_s, beta_s, \
                  intcpt, q_0, j_0):
    m_theory = mu_expansion_alt(z_s, intcpt, q_0, j_0) + \
               alpha_s * data_s[:, 1] + beta_s * data_s[:, 2]
    a = np.array([1.0, alpha_s, beta_s])
    m_var = np.einsum('j,ij', a, np.einsum('ijk,k', cov_s, a))
    m_var += (5.0 * z_err_tot_s / z_s / np.log(10.0)) ** 2
    m_var += 0.12 ** 2 # 0.18 ** 2
    chi_sq = np.sum((data_s[:, 0] - m_theory) ** 2 / m_var)
    return chi_sq

def emcee_ln_p(pars, z_s, z_err_tot_s, data_s, cov_s):
    if -5.0 < pars[0] < 5.0 and -10.0 < pars[1] < 10.0 and \
       0.0 < pars[2] < 40.0 and -5.0 < pars[3] < 5.0 and \
       -5.0 < pars[4] < 5.0:
        return -approx_chi_sq(z_s, z_err_tot_s, data_s, cov_s, \
                              *pars) / 2.0
    else:
        return -np.inf

def sc_parse(z_s_min=0.0233, z_s_max=0.4, verbose=False):

    # read in Hubble Flow SN data
    print "= = = = = = = = = ="
    print "= Hubble Flow SNe ="

    # Dan Scolnic's comments: 
    # User note: This list does not have duplicate SNe removed - most
    # of the SNe have the same name, but a handful of SDSS that have 
    # duplicates don't:
    dupe_sdss = ['16314', '16392', '16333', '14318', '17186', '17784', \
                 '7876']
    dupe_true = ['2006oa', '2006ob', '2006on', '2006py', '2007hx', \
                 '2007jg','2005ir']
    #z_s_min = 0.0233 #0.01
    #z_s_max = 0.15 # 0.4 # 5.0 # 0.15
    n_skip = 14
    n_s = 0
    sne = []
    sne_dupes = []
    sne_sel = []
    est_app_mag_s = []
    sig_app_mag_s = []
    est_z_s = []
    sig_z_s = []
    est_x_0_s = []
    sig_x_0_s = []
    est_x_1_s = []
    sig_x_1_s = []
    est_c_s = []
    sig_c_s = []
    cov_x_1_c_s = []
    cov_x_1_x_0_s = []
    cov_c_x_0_s = []
    pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(pardir, 'data/supercal_vH0.fitres.txt')) as f:

        # loop through full file
        for i, l in enumerate(f):
            if (i > n_skip - 1):
                vals = [val for val in l.split()]
                if len(vals) == 0:
                    break
                sn = vals[1]

                # check SN is on R16 list and is not a duplicate
                #if model_outliers is None:
                #    if sn not in r16_sn_hosts:
                #        continue
                sne.append(sn)
                if sn in sne[:-1] or sn in dupe_sdss:
                    if verbose:
                        print "dupe!", sn
                    sne_dupes.append(sn)
                    continue

                # read in raw data (and add host-mass step)
                z_temp = float(vals[6])
                z_err_temp = float(vals[7])
                lhm_temp = float(vals[10])
                peak_time_temp = float(vals[16])
                x_1_temp = float(vals[17])
                x_1_err_temp = float(vals[18])
                c_temp = float(vals[19])
                c_err_temp = float(vals[20])
                m_temp = float(vals[21])
                if lhm_temp > 10:
                    m_temp += 0.03
                else:
                    m_temp -= 0.03
                m_err_temp = float(vals[22])
                x_0_temp = float(vals[23])
                x_0_err_temp = float(vals[24])
                x_1_c_cov_temp = float(vals[25])
                x_1_x_0_cov_temp = float(vals[26])
                c_x_0_cov_temp = float(vals[27])
                prob_temp = float(vals[30])

                # check that covariance matrix is pos def
                c_x_0_rho = c_x_0_cov_temp / c_err_temp / \
                            x_0_err_temp
                if np.abs(c_x_0_rho) > 0.95:
                    s = '* SN {:s} corr coeff extreme: {:6.3f}'
                    print s.format(sn, c_x_0_rho)
                cov = np.zeros((3, 3))
                cov[0, 0] = x_0_err_temp ** 2
                cov[1, 1] = x_1_err_temp ** 2
                cov[2, 2] = c_err_temp ** 2
                cov[0, 1] = x_1_x_0_cov_temp
                cov[0, 2] = c_x_0_cov_temp
                cov[1, 2] = x_1_c_cov_temp
                cov[1, 0] = cov[0, 1]
                cov[2, 0] = cov[0, 2]
                cov[2, 1] = cov[1, 2]
                try:
                    l_cov = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    s = '* SN {:s} x_0/x_1/c cov not pos def:'
                    print s.format(sn)
                    print cov
                    print 'has eigenvalues: ', np.linalg.eigvals(cov)
                    print '=> rejected'
                    continue

                # R16 data quality cuts
                if (z_temp >= z_s_min and z_temp <= z_s_max and \
                    np.abs(c_temp) < 0.3 and np.abs(x_1_temp) < 3.0 and \
                    x_1_err_temp < 1.0 and prob_temp > 0.0001 and \
                    peak_time_temp < 2.0):
                    sne_sel.append(sn)
                    est_z_s.append(z_temp)
                    sig_z_s.append(z_err_temp)
                    est_app_mag_s.append(m_temp)
                    sig_app_mag_s.append(m_err_temp)
                    est_x_1_s.append(x_1_temp)
                    sig_x_1_s.append(x_1_err_temp)
                    est_c_s.append(c_temp)
                    sig_c_s.append(c_err_temp)
                    est_x_0_s.append(x_0_temp)
                    sig_x_0_s.append(x_0_err_temp)
                    cov_x_1_c_s.append(x_1_c_cov_temp)
                    cov_x_1_x_0_s.append(x_1_x_0_cov_temp)
                    cov_c_x_0_s.append(c_x_0_cov_temp)
                    n_s += 1
    
    # convert lists to arrays
    est_z_s = np.array(est_z_s)
    sig_z_s = np.array(sig_z_s)
    est_app_mag_s = np.array(est_app_mag_s)
    sig_app_mag_s = np.array(sig_app_mag_s)
    est_x_1_s = np.array(est_x_1_s)
    sig_x_1_s = np.array(sig_x_1_s)
    est_c_s = np.array(est_c_s)
    sig_c_s = np.array(sig_c_s)
    est_x_0_s = np.array(est_x_0_s)
    sig_x_0_s = np.array(sig_x_0_s)
    cov_x_1_c_s = np.array(cov_x_1_c_s)
    print "duplicate SNe: ", sne_dupes
    s = '{} SNe after cuts'
    print s.format(n_s)
    s = 'RMS SN app mag err: {:7.5f}'
    print s.format(np.sqrt(np.mean(sig_app_mag_s ** 2)))
    '''rms_sig_app_mag_tot = np.sqrt(np.mean(sig_app_mag_s ** 2 + \
                                          sig_int_s_r16 ** 2))
    s = 'RMS SN total app mag err: {:7.5f}'
    print s.format(rms_sig_app_mag_tot)'''
    s = 'RMS SN z err: {:7.5f}'
    print s.format(np.sqrt(np.mean(sig_z_s ** 2)))
    s = 'mean / sigma of SN shape: {:7.5f}+/-{:7.5f}'
    print s.format(np.mean(est_x_1_s), np.std(est_x_1_s))
    s = 'mean / sigma of SN colour: {:7.5f}+/-{:7.5f}'
    print s.format(np.mean(est_c_s), np.std(est_c_s))

    # convert salt-2 covariance from (x0, x1, c) to (m_B, x1, c)
    sf = -2.5 / (est_x_0_s * np.log(10.0))
    sig_app_mag_s = np.abs(sig_x_0_s * sf)
    cov_x_1_app_mag_s = np.array(cov_x_1_x_0_s) * sf
    cov_c_app_mag_s = np.array(cov_c_x_0_s) * sf
    s = '"average" stretch-colour obs cov: ' + \
        '[[{:7.5f}, {:7.5f}, {:7.5f}],\n' + \
        '                                  ' + \
        ' [{:7.5f}, {:7.5f}, {:7.5f}],\n' + \
        '                                  ' + \
        ' [{:7.5f}, {:7.5f}, {:7.5f}]]'
    print s.format(np.mean(sig_app_mag_s ** 2), \
                   np.mean(cov_x_1_app_mag_s), \
                   np.mean(cov_c_app_mag_s), \
                   np.mean(cov_x_1_app_mag_s), \
                   np.mean(sig_x_1_s ** 2), np.mean(cov_x_1_c_s), \
                   np.mean(cov_c_app_mag_s), np.mean(cov_x_1_c_s), \
                   np.mean(sig_c_s ** 2))
    s = '"average" mag-stretch correlation: {:7.5f}'
    print s.format(np.mean(cov_x_1_app_mag_s / sig_x_1_s / \
                           sig_app_mag_s))
    s = '"average" mag-colour correlation: {:7.5f}'
    print s.format(np.mean(cov_c_app_mag_s / sig_c_s / \
                           sig_app_mag_s))
    s = '"average" stretch-colour correlation: {:7.5f}'
    print s.format(np.mean(cov_x_1_c_s / sig_x_1_s / sig_c_s))
    
    # return parsed data
    to_return = [n_s, est_app_mag_s, sig_app_mag_s, est_z_s, \
    			 sig_z_s, est_x_1_s, sig_x_1_s, est_c_s, sig_c_s, \
    			 cov_x_1_app_mag_s, cov_c_app_mag_s, cov_x_1_c_s]
    return to_return

def b14_parse(z_min=None, z_max=None, qual_cut=False, \
        jla_path='/Users/sfeeney/Software_Packages/jla_v6/jla_likelihood_v6/data/'):

    # read lightcurve data
    print '* reading B14 inputs'
    data = np.genfromtxt(jla_path + 'jla_lcparams.txt', \
                         dtype = None, names = True)
    n_sn_in = len(data)

    # cut if desired
    inds = (np.arange(n_sn_in),)
    if z_min is not None:
        if z_max is not None:
            inds = np.where((data['zcmb'] > z_min) & \
                            (data['zcmb'] < z_max))
        else:
            inds = np.where(z_min < data['zcmb'])
    elif z_max is not None:
        inds = np.where(data['zcmb'] < z_max)
    data = data[inds]
    n_sn = len(data)

    # read V (non-diagonal) covariance matrices
    cmats = {'v0': np.zeros((n_sn, n_sn)), \
             'va': np.zeros((n_sn, n_sn)), \
             'vb': np.zeros((n_sn, n_sn)), \
             'v0a': np.zeros((n_sn, n_sn)), \
             'v0b': np.zeros((n_sn, n_sn)), \
             'vab': np.zeros((n_sn, n_sn))}
    for cmat in cmats:
        d = np.genfromtxt(jla_path + 'jla_' + cmat + '_covmatrix.dat')
        #for i in range(n_sn):
        #    cmats[cmat][i, :] = d[i * n_sn + 1: (i + 1) * n_sn + 1]
        for i in range(n_sn):
            cmats[cmat][i, :] = d[inds[0][i] * n_sn_in + 1 + inds[0]]
        #print np.allclose(cmats[cmat], cmats[cmat].T)
    print '* B14 inputs read'

    return data, cmats

def b14_cov_mat(data, cmats, alpha, beta):
    n_sn = len(data)
    c_mat = cmats['v0'] + alpha ** 2 * cmats['va'] + \
            beta ** 2 * cmats['vb'] + 2 * alpha * cmats['v0a'] - \
            2 * beta * cmats['v0b'] - \
            2 * alpha * beta * cmats['vab']
    d_mat = data['dmb'] ** 2 + (alpha * data['dx1']) ** 2 + \
            (beta * data['dcolor']) ** 2 + \
            2 * alpha * data['cov_m_s'] - \
            2 * beta * data['cov_m_c'] - \
            2 * alpha * beta * data['cov_s_c']# + \
    #        sig_pec^2 + sig_lens^2 + sig_coh^2
    # @TODO: are these contributions already included?
    #        check vs the C matrix approach?
    return c_mat + np.diag(d_mat)

def s17_parse(z_min=None, z_max=None, qual_cut=True):

    # find data directory
    pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pardir += '/data/pantheon/'

    # read lightcurve data
    print '* reading Pantheon inputs'
    if qual_cut:
        fname = 'lcparam_full_long.txt'
    else:
        fname = 'lcparam_full_long_nocut.txt'
    data = np.genfromtxt(pardir + fname, dtype = None, names = True)
    n_sn_in = len(data)

    # cut if desired
    inds = (np.arange(n_sn_in),)
    if z_min is not None:
        if z_max is not None:
            inds = np.where((data['zcmb'] > z_min) & \
                            (data['zcmb'] < z_max))
        else:
            inds = np.where(z_min < data['zcmb'])
    elif z_max is not None:
        inds = np.where(data['zcmb'] < z_max)
    data = data[inds]
    n_sn = len(data)
    #mp.errorbar(data['zcmb'], data['mb'], data['dmb'], ls='None')
    #mp.show()
    
    # read single systematics covariance matrix and add diagonal 
    # statistical variance
    if qual_cut:
        fname = 'sys_full_long.txt'
    else:
        fname = 'sys_full_long_nocut.txt'
    cmat = np.zeros((n_sn, n_sn))
    cmat_raw = np.genfromtxt(pardir + fname)
    for i in range(n_sn):
        cmat[i, :] = cmat_raw[inds[0][i] * n_sn_in + 1 + inds[0]]
    cmat = cmat + np.diag(data['dmb'] ** 2)
    print '* Pantheon inputs read'
    #print np.sum(np.isclose(cmat, cmat.T)), n_sn ** 2
    #mp.imshow(cmat, interpolation='nearest', norm=mpc.LogNorm())
    #mp.show()
    #exit()

    return data, cmat

def sdss_parse(inc_full_shape=True, h_z_only=False):

    # dimension data
    n_bao = 3
    if h_z_only:
        n_type = 1
    else:
        n_type = 2
    data = np.zeros(n_bao, dtype=[('zbao', np.float32), \
                                  ('xbao', np.float32)])
    data = {'zbao': np.zeros(n_bao * n_type), \
            'xbao': np.zeros(n_bao * n_type), \
            'type': []}

    # find data directory
    pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pardir += '/data/bao/sdss/'

    # read data, depending on data combination
    if inc_full_shape:

        # read redshifts and means
        pardir += 'COMBINEDDR12_final_consensus_dM_Hz/'
        fname = 'final_consensus_results_dM_Hz_fsig.txt'
        j = 0
        with open(pardir + fname) as f:
            for i, l in enumerate(f):
                if (np.mod(i, 3) == 1 and not h_z_only):
                    vals = [val for val in l.split()]
                    data['zbao'][j] = float(vals[0])
                    data['xbao'][j] = float(vals[2])
                    data['type'].append('dm/rd')
                    j += 1
                elif (np.mod(i, 3) == 2):
                    vals = [val for val in l.split()]
                    data['zbao'][j] = float(vals[0])
                    data['xbao'][j] = float(vals[2])
                    data['type'].append('hzrd')
                    j += 1

        # read and Cholesky decompose covmat
        fname = 'final_consensus_covtot_dM_Hz_fsig.txt'
        cmat_tot = np.genfromtxt(pardir + fname)
        cmat = np.zeros((n_bao * n_type, n_bao * n_type))
        if h_z_only:
            for i in range(n_bao):
                cmat[i, :] = cmat_tot[3 * i + 1, 1::3]
        else:
            for i in range(n_bao):
                cmat[2 * i, 0::2] = cmat_tot[3 * i, 0::3]
                cmat[2 * i, 1::2] = cmat_tot[3 * i, 1::3]
                cmat[2 * i + 1, 0::2] = cmat_tot[3 * i + 1, 0::3]
                cmat[2 * i + 1, 1::2] = cmat_tot[3 * i + 1, 1::3]
        
    else:

        # read redshifts and means
        pardir += 'COMBINEDDR12_BAO_consensus_dM_Hz/'
        fname = 'BAO_consensus_results_dM_Hz.txt'
        j = 0
        with open(pardir + fname) as f:
            for i, l in enumerate(f):
                if (i > 0):
                    vals = [val for val in l.split()]
                    if (np.mod(i, 2) == 0):
                        data['zbao'][j] = float(vals[0])
                        data['xbao'][j] = float(vals[2])
                        data['type'].append('hzrd')
                        j += 1
                    elif (np.mod(i, 2) == 1) and not h_z_only:
                        data['zbao'][j] = float(vals[0])
                        data['xbao'][j] = float(vals[2])
                        data['type'].append('dm/rd')
                        j += 1

        # read and Cholesky decompose covmat
        fname = 'BAO_consensus_covtot_dM_Hz.txt'
        cmat_tot = np.genfromtxt(pardir + fname)
        if h_z_only:
            cmat = np.zeros((n_bao * n_type, n_bao * n_type))
            for i in range(n_bao):
                cmat[i, :] = cmat_tot[2 * i + 1, 1::2]
        else:
            cmat = cmat_tot

    data['type'] = np.array(data['type'])
    return data, cmat

def b14_chi_sq(pars, data, cmats, h_0 = 70.0, delta_m_cut = 10.0):
    alpha, beta, abs_mag, delta_m, om_m = pars
    mu = z2mu(data['zcmb'], om_m, h_0)
    res = data['mb'] - (abs_mag - alpha * data['x1'] + \
                        beta * data['color'] + \
                        delta_m * (data['3rdvar'] > delta_m_cut)) - mu
    cov_mat = b14_cov_mat(data, cmats, alpha, beta)
    cov_mat_chol = np.linalg.cholesky(cov_mat)
    white_res = np.linalg.solve(cov_mat_chol, res)
    chi_sq = np.dot(white_res, white_res)
    return chi_sq

def emcee_b14_ln_p(pars, data, cmats, h_0 = 70.0, delta_m_cut = 10.0):
    if 0.0 < pars[0] < 0.5 and 0.0 < pars[1] < 6.0 and \
       -25.0 < pars[2] < -15.0 and -0.5 < pars[3] < 0.5 and \
       0.2 < pars[4] < 0.4:
        return -b14_chi_sq(pars, data, cmats, h_0, delta_m_cut) / 2.0
    else:
        return -np.inf

def b14_chi_sq_exp(pars, data, cmats, h_0 = 70.0, delta_m_cut = 10.0):
    alpha, beta, abs_mag, delta_m, q_0, j_0 = pars
    mu = z2mu_exp(data['zcmb'], h_0, q_0, j_0)
    res = data['mb'] - (abs_mag - alpha * data['x1'] + \
                        beta * data['color'] + \
                        delta_m * (data['3rdvar'] > delta_m_cut)) - mu
    cov_mat = b14_cov_mat(data, cmats, alpha, beta)
    cov_mat_chol = np.linalg.cholesky(cov_mat)
    white_res = np.linalg.solve(cov_mat_chol, res)
    chi_sq = np.dot(white_res, white_res)
    return chi_sq

def emcee_b14_ln_p_exp(pars, data, cmats, h_0 = 70.0, \
                       delta_m_cut = 10.0):
    if 0.0 < pars[0] < 0.5 and 0.0 < pars[1] < 6.0 and \
       -25.0 < pars[2] < -15.0 and -0.5 < pars[3] < 0.5 and \
       -5.0 < pars[4] < 5.0 and -10.0 < pars[5] < 10.0:
        return -b14_chi_sq_exp(pars, data, cmats, h_0, \
                               delta_m_cut) / 2.0
    else:
        return -np.inf

def s17_chi_sq(pars, data, cmat_chol, h_0 = 70.0):
    abs_mag, om_m = pars
    mu = z2mu(data['zcmb'], om_m, h_0)
    res = data['mb'] - abs_mag - mu
    white_res = np.linalg.solve(cmat_chol, res)
    chi_sq = np.dot(white_res, white_res)
    return chi_sq

def emcee_s17_ln_p(pars, data, cmat_chol, h_0 = 70.0):
    if -25.0 < pars[0] < -15.0 and 0.2 < pars[1] < 0.4:
        return -s17_chi_sq(pars, data, cmat_chol, h_0) / 2.0
    else:
        return -np.inf

def s17_chi_sq_exp(pars, data, cmat_chol, h_0 = 70.0):
    abs_mag, q_0, j_0 = pars
    mu = z2mu_exp(data['zcmb'], h_0, q_0, j_0)
    res = data['mb'] - abs_mag - mu
    white_res = np.linalg.solve(cmat_chol, res)
    chi_sq = np.dot(white_res, white_res)
    return chi_sq

def emcee_s17_ln_p_exp(pars, data, cmat_chol, h_0 = 70.0):
    if -25.0 < pars[0] < -15.0 and -5.0 < pars[1] < 5.0 and \
       -5.0 < pars[2] < 5.0:
        return -s17_chi_sq_exp(pars, data, cmat_chol, h_0) / 2.0
    else:
        return -np.inf

def bao_chi_sq(pars, data, cmat_chol, r_d_fid):

    # figure out components of measurement
    n_bao = len(data['type'])
    i_h_z = (data['type'] == 'hzrd')
    n_h_z = int(np.sum(i_h_z))
    n_d_m = n_bao - n_h_z

    # construct theoretical values from pars
    om_m, r_d, h_0 = pars
    model = np.zeros(n_bao)
    if n_h_z > 0:
        h_z_bao = h_z(data['zbao'][i_h_z], h_0, 1.0 - om_m, om_m, 0.0)
        model[i_h_z] = h_z_bao * r_d / r_d_fid
    if n_d_m > 0:
        # defined near Eq. 1 of 1607.03155
        d_m_z_bao = z2dc(data['zbao'][~i_h_z], om_m, h_0)
        model[~i_h_z] = d_m_z_bao * r_d_fid / r_d

    # evaluate chi-square
    res = data['xbao'] - model
    white_res = np.linalg.solve(cmat_chol, res)
    chi_sq = np.dot(white_res, white_res)
    return chi_sq

def emcee_b14_bao_ln_p(pars, mu_r_d, sig_r_d, data_sn, cmats_sn, \
                       data_bao, cmat_bao, r_d_fid, \
                       delta_m_cut = 10.0):
    # pars = alpha, beta, abs_mag, delta_m, om_m, r_d, h_0
    if 0.0 < pars[0] < 0.5 and 0.0 < pars[1] < 6.0 and \
       -25.0 < pars[2] < -15.0 and -0.5 < pars[3] < 0.5 and \
       0.2 < pars[4] < 0.4 and 40.0 < pars[6] < 100.0:
        ln_prior = -((pars[5] - mu_r_d) / sig_r_d) ** 2 / 2.0
        ln_like_sn = -b14_chi_sq(pars[:5], data_sn, cmats_sn, \
                                 pars[6], delta_m_cut) / 2.0
        ln_like_bao = -bao_chi_sq(pars[4:], data_bao, \
                                  cmat_bao, r_d_fid) / 2.0
        return ln_prior + ln_like_sn + ln_like_bao
    else:
        return -np.inf

def emcee_s17_bao_ln_p(pars, mu_r_d, sig_r_d, data_sn, cmat_chol_sn, \
                       data_bao, cmat_bao, r_d_fid):
    # pars = abs_mag, om_m, r_d, h_0
    if -25.0 < pars[0] < -15.0 and 0.2 < pars[1] < 0.4 and  \
       40.0 < pars[3] < 100.0:
        ln_prior = -((pars[2] - mu_r_d) / sig_r_d) ** 2 / 2.0
        ln_like_sn = -b14_chi_sq(pars[:2], data_sn, cmat_chol_sn, \
                                 pars[3]) / 2.0
        ln_like_bao = -bao_chi_sq(pars[1:], data_bao, \
                                  cmat_bao, r_d_fid) / 2.0
        return ln_prior + ln_like_sn + ln_like_bao
    else:
        return -np.inf

def bao_chi_sq_exp(pars, data, cmat_chol, r_d_fid):

    # figure out components of measurement
    n_bao = len(data['type'])
    i_h_z = (data['type'] == 'hzrd')
    n_h_z = int(np.sum(i_h_z))
    n_d_m = n_bao - n_h_z

    # construct theoretical values from pars
    q_0, j_0, r_d, h_0 = pars
    model = np.zeros(n_bao)
    if n_h_z > 0:
        h_z_bao = h_z_expansion(data['zbao'][i_h_z], h_0, q_0, j_0)
        model[i_h_z] = h_z_bao * r_d / r_d_fid
    if n_d_m > 0:
        # defined near Eq. 1 of 1607.03155
        d_m_z_bao = z2dc_exp(data['zbao'][~i_h_z], h_0, q_0, j_0)
        model[~i_h_z] = d_m_z_bao * r_d_fid / r_d

    # evaluate chi-square
    res = data['xbao'] - model
    white_res = np.linalg.solve(cmat_chol, res)
    chi_sq = np.dot(white_res, white_res)
    return chi_sq

def emcee_b14_bao_ln_p_exp(pars, mu_r_d, sig_r_d, data_sn, cmats_sn, \
                           data_bao, cmat_bao, r_d_fid, \
                           delta_m_cut = 10.0):
    # pars = alpha, beta, abs_mag, delta_m, q_0, j_0, r_d, h_0
    if 0.0 < pars[0] < 0.5 and 0.0 < pars[1] < 6.0 and \
       -25.0 < pars[2] < -15.0 and -0.5 < pars[3] < 0.5 and \
       -5.0 < pars[4] < 5.0 and -10.0 < pars[5] < 10.0 and \
       40.0 < pars[7] < 100.0:
        ln_prior = -((pars[6] - mu_r_d) / sig_r_d) ** 2 / 2.0
        ln_like_sn = -b14_chi_sq_exp(pars[:6], data_sn, cmats_sn, \
                                     pars[7], delta_m_cut) / 2.0
        ln_like_bao = -bao_chi_sq_exp(pars[4:], data_bao, \
                                      cmat_bao, r_d_fid) / 2.0
        return ln_prior + ln_like_sn + ln_like_bao
    else:
        return -np.inf

def emcee_s17_bao_ln_p_exp(pars, mu_r_d, sig_r_d, data_sn, \
                           cmat_chol_sn, data_bao, cmat_bao, \
                           r_d_fid):
    # pars = abs_mag, q_0, j_0, r_d, h_0
    if -25.0 < pars[0] < -15.0 and -5.0 < pars[1] < 5.0 and \
       -10.0 < pars[2] < 10.0 and 40.0 < pars[4] < 100.0:
        ln_prior = -((pars[3] - mu_r_d) / sig_r_d) ** 2 / 2.0
        ln_like_sn = -s17_chi_sq_exp(pars[:3], data_sn, \
                                     cmat_chol_sn, pars[4]) / 2.0
        ln_like_bao = -bao_chi_sq_exp(pars[1:], data_bao, \
                                      cmat_bao, r_d_fid) / 2.0
        return ln_prior + ln_like_sn + ln_like_bao
    else:
        return -np.inf

def gd_opt_wrapper(h_0, gd_1d_density):
    return -gd_1d_density.Prob(h_0)[0]

######################################################################
######################################################################
######################################################################


# settings
recompile = False
sample = False
sampler = 'emcee' # 'opt', 'emcee' or 'stan'
if sampler == 'stan':
    import pystan as ps
    n_samples = 5000
    n_chains = 4
    stan_constrain = False
elif sampler == 'emcee':
    import emcee as mc
    import emcee.utils as mcu
    n_samples = 10000
    n_walk = 50
sim = False
dataset = 's17' # 'r16' (actually Scolnic supercal) or 'b14' or 's17'
r_d_dataset = 'planck' # 'wmap' or 'planck' or 'planck_no_lens'
z_min = None#0.023
z_max = None#0.8
bao_inc_fs = False
bao_h_z_only = False
inv_ladder = True
mu_exp = True
model_outliers = None # 'ht' or None
plotter = 'getdist' # 'getdist' or 'corner'
if plotter == 'getdist':
    import getdist.plots as gdp
if plotter == 'corner':
    import corner
ol_h_z_samples = False
box_not_vio = False
ol_cmb_h_0_ppd = True
ol_wmap_r_d = False
cf_planck_w_lens = False

# plotting settings
lw = 1.5
mp.rc('font', family = 'serif')
#mp.rc('text', usetex=True)
#mp.rcParams['text.latex.preamble']=[r'\usepackage{bm}']
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
mp.rcParams['lines.markeredgewidth'] = lw
cm = mpcm.get_cmap('plasma')
cols = [cm(0.15), cm(0.85), cm(0.5)]

# define colourblind-friendly colours. order is dark blue, orange,
# light blue, orange, then black and grey
cbf_cols = [ (0.0 / 255.0, 107.0 / 255.0, 164.0 / 255.0), \
             (200.0 / 255.0, 82.0 / 255.0, 0.0 / 255.0), \
             (95.0 / 255.0, 158.0 / 255.0, 209.0 / 255.0), \
             (255.0 / 255.0, 128.0 / 255.0, 14.0 / 255.0), \
             (0.0, 0.0, 0.0), \
             (89.0 / 255.0, 89.0 / 255.0, 89.0 / 255.0) ]

# constants Planck/WMAP CMB measurements / chains
base = 'h_of_z'
c = 299792.458 # km s^-1
h_0_local = 73.24
sig_h_0_local = 1.74
j_0 = 1.0
script = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(script))
data_dir = os.path.join(parent_dir, 'data/')
if r_d_dataset == 'wmap':
    cmb_base = 'base_WMAP'
elif r_d_dataset == 'planck_no_lens':
    cmb_base = 'base_plikHM_TT_lowTEB'
else:
    cmb_base = 'base_plikHM_TT_lowTEB_lensing'
cmb_samples = gd.loadMCSamples(data_dir + cmb_base)
h_0_cmb = np.mean(cmb_samples.getParams().H0)
sig_h_0_cmb = np.std(cmb_samples.getParams().H0)
om_r_cmb = 0.0 # 9.24e-5
om_m_cmb = np.mean(cmb_samples.getParams().omegam)
om_l_cmb = np.mean(cmb_samples.getParams().omegal)
q_0_cmb = om_to_q_0(om_l_cmb)
r_d_cmb = np.mean(cmb_samples.getParams().rdrag)
sig_r_d_cmb = np.std(cmb_samples.getParams().rdrag)
if dataset == 'b14':
    sig_v_pec = 150.0
elif dataset == 'r16':
    sig_v_pec = 250.0

# quick plot of d_L expansion performance
# what does this small percentage error mean in terms of q_0?
if False:
    z = np.linspace(0.0, 1.0, 10000)
    mp.plot(z, 1.0 - h_z_expansion(z, h_0_cmb, q_0_cmb, j_0) / \
                     h_z(z, h_0_cmb, om_l_cmb, om_m_cmb, \
                         om_r_cmb), 'r')
    mp.show()

# BAO measurements
# first three are x = H(z) * r_d / r_d^fid
#  - H(z) = x / r_d * r_d^fid
#  - sig(H(z))/H(z) = sqrt((sig(x)/x)**2 + (sig(r_d)/r_d)**2)
# last two are x = c / H(z) / r_d
#  - H(z) = c / x / r_d
#  - sig(H(z))/H(z) = sqrt((sig(x)/x)**2 + (sig(r_d)/r_d)**2)
z_bao = np.array([0.38, 0.51, 0.61, 2.34, 2.36])
x_bao = np.array([81.2, 90.9, 99.0, 9.18, 9.0])
sig_x_bao = np.array([2.4, 2.3, 2.5, 0.28, 0.30])
n_bao = len(z_bao)
r_d_fid_boss_dr12 = 147.78 # Mpc
#z_bao = np.array([0.106, 0.15, 0.38, 0.51, 0.61, 2.34, 2.36])
#h_z_bao = np.array([0.336, 664.0, ])
#sig_h_z_bao = np.array([])

# convert BAO x measurements and uncertainties into H(z) equivalents 
# assuming the CMB r_d constraint is correct
h_z_bao = np.zeros(n_bao)
sig_h_z_bao = np.zeros(n_bao)
h_z_bao[0: 3] = x_bao[0: 3] * r_d_fid_boss_dr12 / r_d_cmb
sig_h_z_bao[0: 3] = h_z_bao[0: 3] * \
					np.sqrt((sig_x_bao[0: 3] / x_bao[0: 3]) ** 2 + \
							(sig_r_d_cmb / r_d_cmb) ** 2)
h_z_bao[3:] = c / x_bao[3:] / r_d_cmb
sig_h_z_bao[3:] = h_z_bao[3:] * \
				  np.sqrt((sig_x_bao[3:] / x_bao[3:]) ** 2 + \
						  (sig_r_d_cmb / r_d_cmb) ** 2)

# overwrites the BAO measurements above. read BOSS BAO measurements
# from Alam et al. 2016 and their covariance. again, convert any H(z)
# based measurements into H(z) assuming the CMB r_d is correct, 
# propagating uncertainties approximately
d_bao_boss_dr12, cmat_bao_boss_dr12 = sdss_parse(inc_full_shape=bao_inc_fs, \
                                                 h_z_only=bao_h_z_only)
cmat_chol_bao_boss_dr12 = np.linalg.cholesky(cmat_bao_boss_dr12)
i_h_z = (d_bao_boss_dr12['type'] == 'hzrd')
z_bao = d_bao_boss_dr12['zbao'][i_h_z]
x_bao = d_bao_boss_dr12['xbao'][i_h_z]
h_z_bao = x_bao * r_d_fid_boss_dr12 / r_d_cmb
sig_x_bao = np.sqrt(np.diag(cmat_bao_boss_dr12)[i_h_z])
sig_h_z_bao = h_z_bao * np.sqrt((sig_x_bao / x_bao) ** 2 + \
                                (sig_r_d_cmb / r_d_cmb) ** 2)

# choose dataset
if dataset == 'b14':

    # read in data
    base += '_' + dataset
    if r_d_dataset == 'wmap':
        base += '_wmap_r_d'
    elif r_d_dataset == 'planck_no_lens':
        base += '_planck_no_lens_r_d'
    if bao_inc_fs:
        base += '_bao_fs'
    if bao_h_z_only:
        base += '_h_z_only'
    if inv_ladder:
        base += '_inv_lad'
    if mu_exp:
        base += '_mu_exp'
    b14_data, b14_cmats = b14_parse(z_min = z_min, z_max = z_max)
    print '* using {:d} SNe'.format(len(b14_data))

    # standardize redshifts for Hubble diagram
    # these errors are very approximate btw
    # @TODO: should really only show diagonal of magnitude cov
    #        with no redshift errors in this case
    est_z_s = b14_data['zcmb']
    sig_z_s = b14_data['dz']
    est_app_mag_s = b14_data['mb']
    sig_app_mag_s = b14_data['dmb']
    sig_tot_z_s = np.sqrt(sig_z_s ** 2 + (sig_v_pec / c) ** 2)

    # choose sampler
    if sampler == 'stan':

        exit('Sampler not yet supported for this dataset!')

    elif sampler == 'opt':

        # quick check of reference chi^2 (should be 682.896 if no 
        # z_min/z_max set)
        #b14_data, b14_cmats = b14_parse()
        pars = [0.141, 3.101, -19.05, -0.070, 0.295]
        chi_sq = b14_chi_sq(pars, b14_data, b14_cmats)
        print '* LCDM ref chi^2: ', chi_sq
        pars = [0.141, 3.101, -19.05, -0.070, -0.575, 1.0]
        chi_sq = b14_chi_sq_exp(pars, b14_data, b14_cmats)
        print '* expansion ref chi^2: ', chi_sq

        # quick check of BAO chi^2
        pars = [om_m_cmb, r_d_cmb, h_0_cmb]
        chi_sq = bao_chi_sq(pars, d_bao_boss_dr12, \
                            cmat_chol_bao_boss_dr12, \
                            r_d_fid_boss_dr12)
        print '* LCDM ref BAO chi^2: ', chi_sq
        pars = [-0.575, 1.0, r_d_cmb, h_0_cmb]
        chi_sq = bao_chi_sq_exp(pars, d_bao_boss_dr12, \
                                cmat_chol_bao_boss_dr12, \
                                r_d_fid_boss_dr12)
        print '* expansion ref BAO chi^2: ', chi_sq

        # optimize! first using LCDM d_L
        print '* optimizing LCDM'
        pini = [0.1, 3.1, -20, 0.0, 0.3]
        opt_res = so.minimize(b14_chi_sq, pini, \
                              args=(b14_data, b14_cmats), \
                              options={'gtol': 1e-03})
        if not opt_res['success']:
            print 'chi-sq minimization failed'
            exit()
        else:
            for i in range(len(pini)):
                print opt_res['x'][i], ' +/- ', \
                      np.sqrt(opt_res['hess_inv'][i, i])

        # next using the expansion
        print '* optimizing expansion'
        pini = [0.141, 3.101, -19.05, -0.070, -0.575, 1.0]
        opt_res_exp = so.minimize(b14_chi_sq_exp, pini, \
                                  args=(b14_data, b14_cmats), \
                                  options={'gtol': 1e-03})
        if not opt_res_exp['success']:
            print 'chi-sq minimization failed'
            exit()
        else:
            for i in range(len(pini)):
                print opt_res_exp['x'][i], ' +/- ', \
                      np.sqrt(opt_res_exp['hess_inv'][i, i])

        # and the BAOs using LCDM, then expansion
        print '* optimizing BAO LCDM'
        pini = [om_m_cmb, r_d_cmb, h_0_cmb]
        opt_res_bao = so.minimize(bao_chi_sq, pini, \
                                  args=(d_bao_boss_dr12, \
                                        cmat_chol_bao_boss_dr12, \
                                        r_d_fid_boss_dr12), \
                                  options={'gtol': 1e-03})
        if not opt_res_bao['success']:
            print 'chi-sq minimization failed'
            exit()
        else:
            for i in range(len(pini)):
                print opt_res_bao['x'][i], ' +/- ', \
                      np.sqrt(opt_res_bao['hess_inv'][i, i])
        print '* optimizing BAO expansion'
        pini = [-0.575, 1.0, r_d_cmb, h_0_cmb]
        opt_res_bao_exp = so.minimize(bao_chi_sq_exp, pini, \
                                      args=(d_bao_boss_dr12, \
                                            cmat_chol_bao_boss_dr12, \
                                            r_d_fid_boss_dr12), \
                                      options={'gtol': 1e-03})
        if not opt_res_bao_exp['success']:
            print 'chi-sq minimization failed'
            exit()
        else:
            for i in range(len(pini)):
                print opt_res_bao_exp['x'][i], ' +/- ', \
                      np.sqrt(opt_res_bao_exp['hess_inv'][i, i])

        # plot hubble diagram
        map_abs_mag, map_om_m = opt_res['x'][2], opt_res['x'][4]
        map_alpha_exp, map_beta_0_exp, map_abs_mag_exp = opt_res_exp['x'][0: 3]
        map_q_0_exp, map_j_0_exp = opt_res_exp['x'][4:]
        print map_abs_mag, map_om_m
        print map_alpha_exp, map_beta_0_exp, map_abs_mag_exp
        print map_q_0_exp, map_j_0_exp
        map_intcpt = 5 * np.log10(70.0) - map_abs_mag - 10
        map_intcpt_exp = 5 * np.log10(70.0) - map_abs_mag_exp - 10
        z_plot = np.linspace(np.min(est_z_s), np.max(est_z_s))
        map_m = z2mu(z_plot, map_om_m, 70.0) + map_abs_mag_exp
        map_m_exp = mu_expansion_alt(z_plot, map_intcpt_exp, \
                                     map_q_0_exp, map_j_0_exp)
        mp.plot(z_plot, map_m, ls='-', color='black')
        mp.plot(z_plot, map_m_exp, \
                ls='--', color='red')
        mp.errorbar(est_z_s, est_app_mag_s, xerr = sig_tot_z_s, \
                    yerr = sig_app_mag_s, color = 'k', ls = 'None', \
                    marker = '_')
        mp.xlabel(r'$z$')
        mp.ylabel(r'$m$')
        mp.savefig(base + '_hubble_fit_opt.pdf', bbox_inches = 'tight')
        fig, axes = mp.subplots(1, 2, figsize=(16, 5))
        map_m = z2mu(est_z_s, map_om_m, 70.0) + map_abs_mag_exp
        map_m_exp = mu_expansion_alt(est_z_s, map_intcpt_exp, \
                                     map_q_0_exp, map_j_0_exp)
        axes[0].errorbar(est_z_s, est_app_mag_s - map_m_exp, \
                    yerr = sig_app_mag_s, color = 'k', ls = 'None', \
                    marker = '_')
        axes[1].errorbar(est_z_s, est_app_mag_s - map_m, \
                    yerr = sig_app_mag_s, color = 'k', ls = 'None', \
                    marker = '_')
        axes[0].set_xlabel(r'$z$')
        axes[0].set_ylabel(r'$\Delta m$')
        axes[1].set_xlabel(r'$z$')
        axes[1].set_ylabel(r'$\Delta m$')
        mp.savefig(base + '_hubble_fit_opt_res.pdf', bbox_inches = 'tight')
        exit()

    elif sampler == 'emcee':

        # sample using emcee. first, some unused MPI shenanigans
        # NB: apparently MPI doesn't increase number of samples
        '''
        pool = mcu.MPIPool()
        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)

        '''

        # sample! or read in samples...
        if inv_ladder:
            if mu_exp:
                print '* using mu expansion'
                n_pars = 8
                pars = ['alpha', 'beta', 'M', 'Delta_M', 'q_0', 'j_0', \
                        'r_d', 'h_0']
                par_names = [r'$\alpha$', r'$\beta$', r'$M$', \
                             r'$\Delta M$', r'$q_0$', r'$j_0$', \
                             r'$r_d$', r'$H_0$']
            else:
                print '* using LCDM mu'
                n_pars = 7
                pars = ['alpha', 'beta', 'M', 'Delta_M', 'Omega_m', \
                        'r_d', 'h_0']
                par_names = [r'$\alpha$', r'$\beta$', r'$M$', \
                             r'$\Delta M$', r'$\Omega_{\rm m}$', \
                             r'$r_d$', r'$H_0$']
        else:
            if mu_exp:
                print '* using mu expansion'
                n_pars = 6
                pars = ['alpha', 'beta', 'M', 'Delta_M', 'q_0', 'j_0']
                par_names = [r'$\alpha$', r'$\beta$', r'$M$', \
                             r'$\Delta M$', r'$q_0$', r'$j_0$']
            else:
                print '* using LCDM mu'
                n_pars = 5
                pars = ['alpha', 'beta', 'M', 'Delta_M', 'Omega_m']
                par_names = [r'$\alpha$', r'$\beta$', r'$M$', \
                             r'$\Delta M$', r'$\Omega_{\rm m}$']
        if sample:

            # generate initial conditions. don't need to be 
            # particularly well spread out, according to 
            # documentation: DFM suggests initializing close to the ML
            # point. choose between LCDM mu or q_0/j_0 expansion
            if inv_ladder:
                if mu_exp:
                    ig_mu = np.array([0.1, 3.0, -20.0, 0.0, -0.5, -0.5, 146.0, 70.0])
                    ig_sig = np.array([0.05, 0.3, 0.5, 0.05, 0.5, 0.5, 0.5, 1.0])
                    while True:
                        ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                                       for i in xrange(n_walk)])
                        ig = np.array(ig)
                        if (0.0 < ig[:, 0]).all() and \
                           (ig[:, 0] < 0.5).all() and \
                           (0.0 < ig[:, 1]).all() and \
                           (ig[:, 1] < 6.0).all() and \
                           (-25.0 < ig[:, 2]).all() and \
                           (ig[:, 2] < -15.0).all() and \
                           (-0.5 < ig[:, 3]).all() and \
                           (ig[:, 3] < 0.5).all() and \
                           (-5.0 < ig[:, 4]).all() and \
                           (ig[:, 4] < 5.0).all() and \
                           (-10.0 < ig[:, 5]).all() and \
                           (ig[:, 5] < 10.0).all():
                            break
                else:
                    ig_mu = np.array([0.1, 3.0, -20.0, 0.0, 0.3, 146.0, 70.0])
                    ig_sig = np.array([0.05, 0.3, 0.5, 0.05, 0.01, 0.5, 1.0])
                    while True:
                        ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                                       for i in xrange(n_walk)])
                        ig = np.array(ig)
                        if (0.0 < ig[:, 0]).all() and \
                           (ig[:, 0] < 0.5).all() and \
                           (0.0 < ig[:, 1]).all() and \
                           (ig[:, 1] < 6.0).all() and \
                           (-25.0 < ig[:, 2]).all() and \
                           (ig[:, 2] < -15.0).all() and \
                           (-0.5 < ig[:, 3]).all() and \
                           (ig[:, 3] < 0.5).all() and \
                           (0.2 < ig[:, 4]).all() and \
                           (ig[:, 4] < 0.4).all():
                            break
            else:
                if mu_exp:
                    ig_mu = np.array([0.1, 3.0, -20.0, 0.0, -0.5, -0.5])
                    ig_sig = np.array([0.05, 0.3, 0.5, 0.05, 0.5, 0.5])
                    while True:
                        ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                                       for i in xrange(n_walk)])
                        ig = np.array(ig)
                        if (0.0 < ig[:, 0]).all() and \
                           (ig[:, 0] < 0.5).all() and \
                           (0.0 < ig[:, 1]).all() and \
                           (ig[:, 1] < 6.0).all() and \
                           (-25.0 < ig[:, 2]).all() and \
                           (ig[:, 2] < -15.0).all() and \
                           (-0.5 < ig[:, 3]).all() and \
                           (ig[:, 3] < 0.5).all() and \
                           (-5.0 < ig[:, 4]).all() and \
                           (ig[:, 4] < 5.0).all() and \
                           (-10.0 < ig[:, 5]).all() and \
                           (ig[:, 5] < 10.0).all():
                            break
                else:
                    ig_mu = np.array([0.1, 3.0, -20.0, 0.0, 0.3])
                    ig_sig = np.array([0.05, 0.3, 0.5, 0.05, 0.01])
                    while True:
                        ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                                       for i in xrange(n_walk)])
                        ig = np.array(ig)
                        if (0.0 < ig[:, 0]).all() and \
                           (ig[:, 0] < 0.5).all() and \
                           (0.0 < ig[:, 1]).all() and \
                           (ig[:, 1] < 6.0).all() and \
                           (-25.0 < ig[:, 2]).all() and \
                           (ig[:, 2] < -15.0).all() and \
                           (-0.5 < ig[:, 3]).all() and \
                           (ig[:, 3] < 0.5).all() and \
                           (0.2 < ig[:, 4]).all() and \
                           (ig[:, 4] < 0.4).all():
                            break

            # sample!
            # NB: using "threads = 20" results in seg faults
            print '* starting sampling'
            if inv_ladder:
                if mu_exp:
                    sampler = mc.EnsembleSampler(n_walk, n_pars, \
                                                 emcee_b14_bao_ln_p_exp, \
                                                 args = [r_d_cmb, \
                                                         sig_r_d_cmb, 
                                                         b14_data, \
                                                         b14_cmats, \
                                                         d_bao_boss_dr12, \
                                                         cmat_chol_bao_boss_dr12, \
                                                         r_d_fid_boss_dr12])
                else:
                    sampler = mc.EnsembleSampler(n_walk, n_pars, \
                                                 emcee_b14_bao_ln_p, \
                                                 args = [r_d_cmb, \
                                                         sig_r_d_cmb, 
                                                         b14_data, \
                                                         b14_cmats, \
                                                         d_bao_boss_dr12, \
                                                         cmat_chol_bao_boss_dr12, \
                                                         r_d_fid_boss_dr12])
            else:
                if mu_exp:
                    sampler = mc.EnsembleSampler(n_walk, n_pars, \
                                                 emcee_b14_ln_p_exp, \
                                                 args = [b14_data, \
                                                         b14_cmats])
                else:
                    sampler = mc.EnsembleSampler(n_walk, n_pars, \
                                                 emcee_b14_ln_p, \
                                                 args = [b14_data, \
                                                         b14_cmats])
            pos, prob, state = sampler.run_mcmc(ig, n_samples / 2)
            print '* burned in'
            sampler.reset()
            sampler.run_mcmc(pos, n_samples, rstate0 = state)
            #pool.close()

            # report completion and save samples
            print '* sampling complete'
            print "* mean acceptance fraction: ", \
                  np.mean(sampler.acceptance_fraction)
            samples = sampler.chain.reshape((-1, n_pars))
            hdr_str = '# Sample generated by emcee\n'
            hdr_str += ','.join(pars)
            #hdr_str += 'lp__,' + ','.join(pars)
            # @TODO: need to add log-prob to this array?
            np.savetxt(base + '_chain.csv', samples, delimiter=',', \
                       header=hdr_str, comments='')

        else:

            print '* reading samples'
            samples = np.genfromtxt(base + '_chain.csv', \
                                    delimiter=",", skip_header=2)
        n_samples = len(samples)

elif dataset == 's17':

    # read in data
    base += '_' + dataset
    if r_d_dataset == 'wmap':
        base += '_wmap_r_d'
    elif r_d_dataset == 'planck_no_lens':
        base += '_planck_no_lens_r_d'
    if bao_inc_fs:
        base += '_bao_fs'
    if bao_h_z_only:
        base += '_h_z_only'
    if inv_ladder:
        base += '_inv_lad'
    if mu_exp:
        base += '_mu_exp'
    s17_data, s17_cmat = s17_parse(z_min = z_min, z_max = z_max)
    s17_cmat_chol = np.linalg.cholesky(s17_cmat)
    print '* using {:d} SNe'.format(len(s17_data))

    # standardize redshifts for Hubble diagram
    # these errors are very approximate btw
    # @TODO: should really only show diagonal of magnitude cov
    #        with no redshift errors in this case
    est_z_s = s17_data['zcmb']
    est_app_mag_s = s17_data['mb']
    sig_app_mag_s = s17_data['dmb']

    # choose sampler
    if sampler == 'stan':

        exit('Sampler not yet supported for this dataset!')

    elif sampler == 'opt':

        # quick check of reference chi^2
        # @TODO: ask Dan S for a reference check?
        pars = [-19.3, 0.295]
        chi_sq = s17_chi_sq(pars, s17_data, s17_cmat_chol)
        print '* LCDM ref chi^2: ', chi_sq
        pars = [-19.3, -0.575, 1.0]
        chi_sq = s17_chi_sq_exp(pars, s17_data, s17_cmat_chol)
        print '* expansion ref chi^2: ', chi_sq

        # optimize! first using LCDM d_L
        print '* optimizing LCDM'
        pini = [-19.3, 0.4]
        opt_res = so.minimize(s17_chi_sq, pini, \
                              args=(s17_data, s17_cmat_chol), \
                              options={'gtol': 1e-03})
        if not opt_res['success']:
            print 'chi-sq minimization failed'
            exit()
        else:
            for i in range(len(pini)):
                print opt_res['x'][i], ' +/- ', \
                      np.sqrt(opt_res['hess_inv'][i, i])

        # next using the expansion
        print '* optimizing expansion'
        pini = [-19.3, -0.575, 1.0]
        opt_res_exp = so.minimize(s17_chi_sq_exp, pini, \
                                  args=(s17_data, s17_cmat_chol), \
                                  options={'gtol': 1e-03})
        if not opt_res_exp['success']:
            print 'chi-sq minimization failed'
            exit()
        else:
            for i in range(len(pini)):
                print opt_res_exp['x'][i], ' +/- ', \
                      np.sqrt(opt_res_exp['hess_inv'][i, i])

        # plot hubble diagram
        map_abs_mag, map_om_m = opt_res['x']
        map_abs_mag_exp, map_q_0_exp, map_j_0_exp = opt_res_exp['x']
        print map_abs_mag, map_om_m
        print map_abs_mag_exp, map_q_0_exp, map_j_0_exp
        map_intcpt = 5 * np.log10(70.0) - map_abs_mag - 10
        map_intcpt_exp = 5 * np.log10(70.0) - map_abs_mag_exp - 10
        z_plot = np.linspace(np.min(est_z_s), np.max(est_z_s))
        map_m = z2mu(z_plot, map_om_m, 70.0) + map_abs_mag_exp
        map_m_exp = mu_expansion_alt(z_plot, map_intcpt_exp, \
                                     map_q_0_exp, map_j_0_exp)
        mp.plot(z_plot, map_m, ls='-', color='black')
        mp.plot(z_plot, map_m_exp, \
                ls='--', color='red')
        mp.errorbar(est_z_s, est_app_mag_s, sig_app_mag_s, \
                    color = 'k', ls = 'None', marker = '_')
        mp.xlabel(r'$z$')
        mp.ylabel(r'$m$')
        mp.savefig(base + '_hubble_fit_opt.pdf', bbox_inches = 'tight')
        fig, axes = mp.subplots(1, 2, figsize=(16, 5))
        map_m = z2mu(est_z_s, map_om_m, 70.0) + map_abs_mag_exp
        map_m_exp = mu_expansion_alt(est_z_s, map_intcpt_exp, \
                                     map_q_0_exp, map_j_0_exp)
        axes[0].errorbar(est_z_s, est_app_mag_s - map_m_exp, \
                         sig_app_mag_s, color = 'k', ls = 'None', \
                         marker = '_')
        axes[1].errorbar(est_z_s, est_app_mag_s - map_m, \
                         sig_app_mag_s, color = 'k', ls = 'None', \
                         marker = '_')
        axes[0].set_xlabel(r'$z$')
        axes[0].set_ylabel(r'$\Delta m$')
        axes[1].set_xlabel(r'$z$')
        axes[1].set_ylabel(r'$\Delta m$')
        mp.savefig(base + '_hubble_fit_opt_res.pdf', bbox_inches = 'tight')
        exit()

    elif sampler == 'emcee':

        # sample using emcee. first, some unused MPI shenanigans
        # NB: apparently MPI doesn't increase number of samples
        '''
        pool = mcu.MPIPool()
        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)

        '''

        # sample! or read in samples...
        if inv_ladder:
            if mu_exp:
                print '* using mu expansion'
                n_pars = 5
                pars = ['M', 'q_0', 'j_0', 'r_d', 'h_0']
                par_names = [r'$M$', r'$q_0$', r'$j_0$', \
                             r'$r_d$', r'$H_0$']
            else:
                print '* using LCDM mu'
                n_pars = 4
                pars = ['M', 'Omega_m', 'r_d', 'h_0']
                par_names = [r'$M$', r'$\Omega_{\rm m}$', \
                             r'$r_d$', r'$H_0$']
        else:
            if mu_exp:
                print '* using mu expansion'
                n_pars = 3
                pars = ['M', 'q_0', 'j_0']
                par_names = [r'$M$', r'$q_0$', r'$j_0$']
            else:
                print '* using LCDM mu'
                n_pars = 2
                pars = ['M', 'Omega_m']
                par_names = [r'$M$', r'$\Omega_{\rm m}$']
        if sample:

            # generate initial conditions. don't need to be 
            # particularly well spread out, according to 
            # documentation: DFM suggests initializing close to the ML
            # point. choose between LCDM mu or q_0/j_0 expansion
            if inv_ladder:
                if mu_exp:
                    ig_mu = np.array([-19.3, -0.5, -0.5, 146.0, 70.0])
                    ig_sig = np.array([0.1, 0.5, 0.5, 0.5, 1.0])
                    while True:
                        ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                                       for i in xrange(n_walk)])
                        ig = np.array(ig)
                        if (-25.0 < ig[:, 0]).all() and \
                           (ig[:, 0] < -15.0).all() and \
                           (-5.0 < ig[:, 1]).all() and \
                           (ig[:, 1] < 5.0).all() and \
                           (-10.0 < ig[:, 2]).all() and \
                           (ig[:, 2] < 10.0).all():
                            break
                else:
                    ig_mu = np.array([-19.3, 0.3, 146.0, 70.0])
                    ig_sig = np.array([0.1, 0.01, 0.5, 1.0])
                    while True:
                        ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                                       for i in xrange(n_walk)])
                        ig = np.array(ig)
                        if (-25.0 < ig[:, 0]).all() and \
                           (ig[:, 0] < -15.0).all() and \
                           (0.2 < ig[:, 1]).all() and \
                           (ig[:, 1] < 0.4).all():
                            break
            else:
                if mu_exp:
                    ig_mu = np.array([-19.3, -0.5, -0.5])
                    ig_sig = np.array([0.1, 0.5, 0.5])
                    while True:
                        ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                                       for i in xrange(n_walk)])
                        ig = np.array(ig)
                        if (-25.0 < ig[:, 0]).all() and \
                           (ig[:, 0] < -15.0).all() and \
                           (-5.0 < ig[:, 1]).all() and \
                           (ig[:, 1] < 5.0).all() and \
                           (-10.0 < ig[:, 2]).all() and \
                           (ig[:, 2] < 10.0).all():
                            break
                else:
                    ig_mu = np.array([-19.3, 0.3])
                    ig_sig = np.array([0.1, 0.01])
                    while True:
                        ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                                       for i in xrange(n_walk)])
                        ig = np.array(ig)
                        if (-25.0 < ig[:, 0]).all() and \
                           (ig[:, 0] < -15.0).all() and \
                           (0.2 < ig[:, 1]).all() and \
                           (ig[:, 1] < 0.4).all():
                            break

            # sample!
            # NB: using "threads = 20" results in seg faults
            print '* starting sampling'
            if inv_ladder:
                if mu_exp:
                    sampler = mc.EnsembleSampler(n_walk, n_pars, \
                                                 emcee_s17_bao_ln_p_exp, \
                                                 args = [r_d_cmb, \
                                                         sig_r_d_cmb, 
                                                         s17_data, \
                                                         s17_cmat_chol, \
                                                         d_bao_boss_dr12, \
                                                         cmat_chol_bao_boss_dr12, \
                                                         r_d_fid_boss_dr12])
                else:
                    sampler = mc.EnsembleSampler(n_walk, n_pars, \
                                                 emcee_s17_bao_ln_p, \
                                                 args = [r_d_cmb, \
                                                         sig_r_d_cmb, 
                                                         s17_data, \
                                                         s17_cmat_chol, \
                                                         d_bao_boss_dr12, \
                                                         cmat_chol_bao_boss_dr12, \
                                                         r_d_fid_boss_dr12])
            else:
                if mu_exp:
                    sampler = mc.EnsembleSampler(n_walk, n_pars, \
                                                 emcee_s17_ln_p_exp, \
                                                 args = [s17_data, \
                                                         s17_cmat_chol])
                else:
                    sampler = mc.EnsembleSampler(n_walk, n_pars, \
                                                 emcee_s17_ln_p, \
                                                 args = [s17_data, \
                                                         s17_cmat_chol])
            pos, prob, state = sampler.run_mcmc(ig, n_samples / 2)
            print '* burned in'
            sampler.reset()
            sampler.run_mcmc(pos, n_samples, rstate0 = state)
            #pool.close()

            # report completion and save samples
            print '* sampling complete'
            print "* mean acceptance fraction: ", \
                  np.mean(sampler.acceptance_fraction)
            samples = sampler.chain.reshape((-1, n_pars))
            hdr_str = '# Sample generated by emcee\n'
            hdr_str += ','.join(pars)
            #hdr_str += 'lp__,' + ','.join(pars)
            # @TODO: need to add log-prob to this array?
            np.savetxt(base + '_chain.csv', samples, delimiter=',', \
                       header=hdr_str, comments='')

        else:

            print '* reading samples'
            samples = np.genfromtxt(base + '_chain.csv', \
                                    delimiter=",", skip_header=2)
        n_samples = len(samples)

elif dataset == 'r16':

    # read data
    if sim:
        outputs = sh0.hh0_sim(setup = 'r16', \
                              fix_redshifts = False, \
                              model_outliers = None, \
                              inc_met_dep = True, \
                              inc_zp_off = True, \
                              constrain = False, \
                              round_data = False)
        n_ch_d, n_ch_p, n_ch_c, n_ch_s, n_c_ch, n_s, dis_anc, \
        sig_dis_anc, est_app_mag_c, sig_app_mag_c, est_p_c, \
        sig_int_c, est_app_mag_s_ch, sig_app_mag_s_ch, \
        est_app_mag_s, sig_app_mag_s, est_z_s, est_x_1_s, sig_x_1_s, \
        est_c_s, sig_c_s, cov_x_1_app_mag_s, cov_c_app_mag_s, \
        cov_x_1_c_s, sig_int_s, est_q_0, sig_q_0, sig_zp_off, \
        zp_off_mask, par_anc_lkc, sim_info, sig_z_s, sig_v_pec, \
        est_z_c = outputs

        # cheeky plot of simulated data versus R16 inputs with and 
        # without standardisation
        # @TODO: why is scatter so much bigger in sim? it's fit okay 
        # in the Stan model, but it looks huge!
        # I BELIEVE this is due to stretch/colour scatter: am i using
        # much broader values in sims?
        # i think so: once corrected, sims and data look v similar
        # there's also the covariance between supernovae!
        '''
        mp.errorbar(est_z_s, \
                    est_app_mag_s - (-0.14) * est_x_1_s - 3.1 * est_c_s, \
                    sig_app_mag_s, color = 'k', ls = 'None')
        #mp.errorbar(est_z_s, est_app_mag_s, sig_app_mag_s, \
                     color = 'k', ls = 'None')

        outputs = ph0.hh0_parse(dataset = 'r16', \
                                fix_redshifts = False, \
                                inc_met_dep = True, \
                                model_outliers = None, \
                                max_col_c = None)
        n_ch_d, n_ch_p, n_ch_c, n_ch_s, n_c_ch, n_s, dis_anc, \
        sig_dis_anc, est_app_mag_c, sig_app_mag_c, est_p_c, sig_int_c, \
        est_app_mag_s_ch, sig_app_mag_s_ch, est_app_mag_s, \
        sig_app_mag_s, est_z_s, est_x_1_s, sig_x_1_s, est_c_s, sig_c_s, \
        cov_x_1_app_mag_s, cov_c_app_mag_s, cov_x_1_c_s, sig_int_s, \
        est_q_0, sig_q_0, sig_zp_off, zp_off_mask, \
        par_anc_lkc, sig_z_s, sig_v_pec, est_z_c = outputs

        mp.errorbar(est_z_s, est_app_mag_s - \
                             (-0.14) * est_x_1_s - 3.1 * est_c_s, \
                    sig_app_mag_s, color = 'g', ls = 'None')
        #mp.errorbar(est_z_s, est_app_mag_s, sig_app_mag_s, \
                     color = 'g', ls = 'None')
    '''
    else:
        n_s, est_app_mag_s, sig_app_mag_s, est_z_s, sig_z_s, \
        est_x_1_s, sig_x_1_s, est_c_s, sig_c_s, cov_x_1_app_mag_s, \
        cov_c_app_mag_s, cov_x_1_c_s = sc_parse(z_s_min = z_min, \
                                                z_s_max = z_max)
    sig_tot_z_s = np.sqrt(sig_z_s ** 2 + (sig_v_pec / c) ** 2)
    #mp.errorbar(est_z_s, est_app_mag_s, sig_app_mag_s, \
    #            color = 'r', ls = 'None')
    #mp.show()

    # construct high-z SNe data and covariance matrices
    data_s_hi_z = np.zeros((n_s, 3))
    for i in range(0, n_s):
        data_s_hi_z[i, :] = [est_app_mag_s[i], est_x_1_s[i], \
                             est_c_s[i]]
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
    cov_s_hi_z = np.zeros((n_s, 3, 3))
    cov_s_hi_z[:, 0, 0] = sig_app_mag_s ** 2
    cov_s_hi_z[:, 1, 0] = cov_x_1_app_mag_s
    cov_s_hi_z[:, 0, 1] = cov_x_1_app_mag_s
    cov_s_hi_z[:, 2, 0] = cov_c_app_mag_s
    cov_s_hi_z[:, 0, 2] = cov_c_app_mag_s
    cov_s_hi_z[:, 1, 1] = sig_x_1_s ** 2
    cov_s_hi_z[:, 1, 2] = cov_x_1_c_s
    cov_s_hi_z[:, 2, 1] = cov_x_1_c_s
    cov_s_hi_z[:, 2, 2] = sig_c_s ** 2
    #print np.divide(cov_l_s_hi_z[100, :, :], \
    #                np.linalg.cholesky(cov_s_hi_z[100, :, :]))

    # choose sampler
    if sampler == 'emcee':

        '''
        # quick plot of likelihood as a function of q_0
        n_grid = 500
        q_0s = np.linspace(-5.0, 5.0, n_grid)
        chisq = np.zeros(n_grid)
        for i in range(n_grid):
            #chisq[i] = approx_chi_sq(est_z_s, np.sqrt(sig_z_s ** 2 + sig_v_pec ** 2), \
            #                data_s_hi_z, cov_s_hi_z, -0.14, 3.1, \
            #                5.0 * np.log10(h_0_cmb) - -19.2 - 10, \
            #                q_0s[i], j_0)
            chisq[i] = approx_chi_sq(est_z_s, sig_tot_z_s, \
                            data_s_hi_z, cov_s_hi_z, -0.14, 3.1, 18.57, \
                            q_0s[i], j_0)
        mp.plot(q_0s, chisq)
        mp.axvline(-0.5575, color = 'Gray', ls = '--')
        mp.show()
        '''

        # sample or read from file
        if sample:

            # @TODO: generalize to LCDM too
            n_pars = 5
            pars = ['alpha', 'beta', 'b', 'q_0', 'j_0']
            par_names = [r'$\alpha$', r'$\beta$', r'$b$', r'$q_0$', \
                         r'$j_0$']
            ig_mu = np.array([-0.1, 3.0, 20.0, -0.5, 0.0])
            ig_sig = np.array([0.5, 3.0, 5.0, 1.0, 5.0])
            while True:
                ig = np.array([npr.randn(n_pars) * ig_sig + ig_mu \
                               for i in xrange(n_walk)])
                ig = np.array(ig)
                if (-5.0 < ig[:, 0]).all() and \
                   (ig[:, 0] < 5.0).all() and \
                   (-10.0 < ig[:, 1]).all() and \
                   (ig[:, 1] < 10.0).all() and \
                   (0.0 < ig[:, 2]).all() and \
                   (ig[:, 2] < 40.0).all() and \
                   (-5.0 < ig[:, 3]).all() and \
                   (ig[:, 3] < 5.0).all() and \
                   (-5.0 < ig[:, 4]).all() and \
                   (ig[:, 4] < 5.0).all():
                    break
            sampler = mc.EnsembleSampler(n_walk, n_pars, emcee_ln_p, \
                                         args = [est_z_s, \
                                                 sig_tot_z_s, \
                                                 data_s_hi_z, \
                                                 cov_s_hi_z])
            pos, prob, state = sampler.run_mcmc(ig, n_samples / 2)
            print '* burned in'
            sampler.reset()
            sampler.run_mcmc(pos, n_samples, rstate0 = state)

            # report completion and save samples
            print '* sampling complete'
            print "* mean acceptance fraction: ", \
                  np.mean(sampler.acceptance_fraction)
            samples = sampler.chain.reshape((-1, n_pars))
            hdr_str = '# Sample generated by emcee\n'
            hdr_str += ','.join(pars)
            #hdr_str += 'lp__,' + ','.join(pars)
            # @TODO: need to add log-prob to this array?
            np.savetxt('r16_chain.csv', samples, delimiter=',', \
                       header=hdr_str, comments='')

        else:

            samples = np.genfromtxt('r16_chain.csv', delimiter=",", \
                                    skip_header=2)
        n_samples = len(samples)

    elif sampler == 'stan':

        # compile Stan model
        if model_outliers == 'ht':
            base += '_ht_outliers'
        if recompile:
            stan_model = ps.StanModel(base + '.stan')
            with open(base + '_model.pkl', 'wb') as f:
                pickle.dump(stan_model, f)
        else:
            try:
                with open(base + '_model.pkl', 'rb') as f:
                    stan_model = pickle.load(f)
            except EnvironmentError:
                print 'ERROR: pickled Stan model (' + base + \
                      '_model.pkl) not found. Please set recompile = True'
                exit()

        # set up stan inputs and sample
        stan_data = {'n_s_hi_z': n_s, 'est_z_s_hi_z': est_z_s, \
                     'sig_z_s_hi_z': sig_z_s, \
                     'data_s_hi_z': data_s_hi_z, \
                     'cov_l_s_hi_z': cov_l_s_hi_z, \
                     'sig_v_pec': sig_v_pec}
        if model_outliers == 'ht':
            stan_pars = ['intcpt_s', 'sig_int_s', 'nu_s', 'alpha_s', \
                         'beta_s', 'q_0', 'j_0']
        else:
            stan_pars = ['intcpt_s', 'sig_int_s', 'alpha_s', \
                         'beta_s', 'q_0', 'j_0']
        if stan_constrain:
            stan_seed = 23102014
        else:
            stan_seed = None
        if sample:
            fit = stan_model.sampling(data = stan_data, \
                                      iter = n_samples, \
                                      chains = n_chains, \
                                      seed = stan_seed, \
                                      pars = stan_pars)
            samples = fit.extract(permuted = False, inc_warmup = True)
            stan_version = ps.__version__.split('.')
            hdr_str = '# Sample generated by Stan\n'
            hdr_str += '# stan_version_major={:s}\n'.format(stan_version[0])
            hdr_str += '# stan_version_minor={:s}\n'.format(stan_version[1])
            hdr_str += '# stan_version_patch={:s}\n'.format(stan_version[2])
            hdr_str += 'lp__,' + ','.join(stan_pars)
            idx = len(stan_pars) + np.arange(0, len(stan_pars) + 1)
            idx = np.mod(idx, len(stan_pars) + 1)
            for i in range(0, n_chains):
                np.savetxt(base + '_minimal_chain_{:d}.csv'.format(i), \
                           samples[:, i, idx], delimiter = ',', \
                           header = hdr_str, comments = '')
            print fit

        # convert samples into 1D array
        pars = stan_pars
        if model_outliers == 'ht':
            par_names = ['$a$', r'$\sigma_{\rm int}$', r'$\nu$', \
                         r'$\alpha$', r'$\beta$', '$q_0$', '$j_0$']
        else:
            par_names = ['$a$', r'$\sigma_{\rm int}$', r'$\alpha$', \
                         r'$\beta$', '$q_0$', '$j_0$']
        raw_samples = []
        for file in os.listdir("."):
            if fnmatch.fnmatch(file, base + '_minimal_chain_*.csv'):
                print "reading " + file
                d = np.genfromtxt(file, delimiter = ",", names = True, \
                                  skip_header = 4)
                raw_samples.append(d)
        n_chains = len(raw_samples)
        n_samples = raw_samples[0].shape[0]
        n_warmup = n_samples / 2
        n_thin = 1
        n_pars = len(pars)
        samples = np.zeros((n_chains * n_warmup / n_thin, n_pars))
        for i in range(0, n_chains):
            for j in range(0, n_pars):
                samples[i * n_warmup / n_thin: (i + 1) * n_warmup / n_thin, j] = \
                    raw_samples[i][pars[j]][n_warmup::n_thin]

# plots!
if plotter == 'corner':

    # simple corner plot including limits in titles
    fig = corner.corner(samples, \
                        show_titles = True, labels = par_names)
    fig.savefig(base + '_corner_plot.pdf', bbox_inches = 'tight')
    #mp.show()

elif plotter == 'getdist':

    # convert samples into GetDist MCSamples object
    par_names = [par_name.replace('$', '') for par_name in par_names]
    par_ranges = {}
    gd_samples = gd.MCSamples(samples = samples, names = pars, 
                              labels = par_names, ranges = par_ranges)
    if dataset == 'r16' and model_outliers == 'ht':
        gd_samples.addDerived(nu2phr(gd_samples.getParams().nu_s), \
                              name='phr_s', label=r't')
        gd_samples.setRanges({'phr_s':(0.0, 1.0)})
        gd_samples.updateBaseStatistics()
        i_nu_s = pars.index('nu_s')
        pars[i_nu_s] = 'phr_s'

    # not sure if we want this, but the standard Planck chains we've 
    # been using include CMB lensing. we might therefore want to 
    # compare to this case even if we're not including lensing in the 
    # inverse distance ladder
    if cf_planck_w_lens and r_d_dataset != 'planck':
        cmb_base = 'base_plikHM_TT_lowTEB_lensing'
        cmb_samples = gd.loadMCSamples(data_dir + cmb_base)
        h_0_cmb = np.mean(cmb_samples.getParams().H0)
        sig_h_0_cmb = np.std(cmb_samples.getParams().H0)
        om_r_cmb = 0.0 # 9.24e-5
        om_m_cmb = np.mean(cmb_samples.getParams().omegam)
        om_l_cmb = np.mean(cmb_samples.getParams().omegal)
        q_0_cmb = om_to_q_0(om_l_cmb)

    # plot parameter fits
    if sim:
        par_vals = [5.0 * np.log10(sim_info['h_0']) - \
                    sim_info['abs_mag_s_std'] - 10, \
                    sim_info['sig_int_s'], sim_info['alpha_s'], \
                    sim_info['beta_s'], sim_info['q_0'], 1.0]
    else:
        par_vals = [None] * n_pars
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
    #mp.show()

    # extract mean-posterior estimates and plot Hubble diagram
    # @TODO: mag-error-only version of below?!
    mp.subplots()
    estimates = gd_samples.getMeans()
    z_plot = np.linspace(np.min(est_z_s), np.max(est_z_s))
    if mu_exp:
        if inv_ladder:
            if dataset == 's17':
                est_inds = [0, 1, 2, 4]
            else:
                est_inds = [2, 4, 5, 7]
            mp_abs_mag, mp_q_0, mp_j_0, mp_h_0 = estimates[est_inds]
            mp_intcpt = 5 * np.log10(mp_h_0) - mp_abs_mag - 10
        else:
            if dataset == 'b14':
                est_inds = [2, 4, 5]
            elif dataset == 's17':
                est_inds = [0, 1, 2]
            elif dataset == 'r16':
                if model_outliers == 'ht':
                    est_inds = [0, 5, 6]
                else:
                    est_inds = [0, 4, 5]
            mp_intcpt, mp_q_0, mp_j_0 = estimates[est_inds]
            if dataset == 'b14' or dataset == 's17':
                mp_intcpt = 5 * np.log10(70.0) - mp_intcpt - 10
        mp.plot(z_plot, mu_expansion_alt(z_plot, mp_intcpt, mp_q_0, mp_j_0))
        mp.plot(z_plot, mu_expansion_alt(z_plot, mp_intcpt, -4, mp_j_0), 'r--')
        mp.plot(z_plot, mu_expansion_alt(z_plot, mp_intcpt, -3, mp_j_0), 'r--')
        mp.plot(z_plot, mu_expansion_alt(z_plot, mp_intcpt, -2, mp_j_0), 'r--')
        mp.plot(z_plot, mu_expansion_alt(z_plot, mp_intcpt, -1, mp_j_0), 'r--')
        mp.plot(z_plot, mu_expansion_alt(z_plot, mp_intcpt, q_0_cmb, mp_j_0), 'r--')
        mp.plot(z_plot, mu_expansion_alt(z_plot, mp_intcpt, 0, mp_j_0), 'r--')
        mp.plot(z_plot, mu_expansion_alt(z_plot, mp_intcpt, 1, mp_j_0), 'r--')
    else:
        if inv_ladder:
            if dataset == 's17':
                est_inds = [0, 1, 4]
            else:
                est_inds = [2, 4, 6]
            mp_abs_mag, mp_om_m, mp_h_0 = estimates[est_inds]
        else:
            if dataset == 's17':
                est_inds = [0, 1]
            else:
                est_inds = [2, 4]
            mp_abs_mag, mp_om_m = estimates[est_inds]
            mp_h_0 = 70.0
        mp.plot(z_plot, z2mu(z_plot, mp_om_m, mp_h_0) + mp_abs_mag)
    if dataset == 's17':
        mp.errorbar(est_z_s, est_app_mag_s, sig_app_mag_s, \
                    color = 'k', ls = 'None', marker = '_')
    else:
        mp.errorbar(est_z_s, est_app_mag_s, xerr = sig_tot_z_s, \
                    yerr = sig_app_mag_s, color = 'k', ls = 'None', \
                    marker = '_')
    mp.xlabel(r'$z$')
    mp.ylabel(r'$m$')
    mp.savefig(base + '_hubble_fit.pdf', bbox_inches = 'tight')
    #mp.show()

    # plot including BAO. overplot either one-sigma H(z) region
    # allowed by SNe or the H(z) curves for a random subsample
    # include H_0 prediction if inferring inverse ladder
    if inv_ladder:
        f = mp.figure()
        gs = mpg.GridSpec(2, 2, width_ratios=[1, 3], \
                          height_ratios=[1, 3])
        gs.update(wspace=0, hspace=0)
        n_z_ax = mp.subplot(gs[0, 1])
        ppd_ax = mp.subplot(gs[1, 0])
        h_z_ax = mp.subplot(gs[1, 1])
    else:
        h_z_fig, h_z_ax = mp.subplots()
    #z_plot = np.linspace(np.min(est_z_s), np.max(z_bao))
    #z_sn = np.linspace(np.min(est_z_s), np.max(est_z_s), 100)
    z_sn = np.linspace(0.0, np.max(est_z_s), 100)
    if ol_h_z_samples:

        # randomly select 500 sampled expansion histories
        n_overlay = 500
        inds = npr.randint(0, n_samples, n_overlay)
        for i in range(n_overlay):
            if mu_exp:
                if inv_ladder:
                    h_z_plot = h_z_expansion(z_sn, \
                                             samples[inds[i], est_inds[3]], \
                                             samples[inds[i], est_inds[1]], \
                                             samples[inds[i], est_inds[2]])
                else:
                    h_z_plot = h_z_expansion(z_sn, h_0_local, \
                                             samples[inds[i], est_inds[1]], \
                                             samples[inds[i], est_inds[2]])
            else:
                if inv_ladder:
                    h_z_plot = h_z(z_sn, samples[inds[i], est_inds[2]], \
                                   1.0 - samples[inds[i], est_inds[1]], \
                                   samples[inds[i], est_inds[1]], 0.0)
                else:
                    h_z_plot = h_z(z_sn, h_0_local, \
                                   1.0 - samples[inds[i], est_inds[1]], \
                                   samples[inds[i], est_inds[1]], 0.0)
            h_z_ax.plot(z_sn, h_z_plot / (1.0 + z_sn), 'k', \
                        alpha=0.05, zorder=0)

    else:

        # add H(z) constraints as derived parameters
        n_nodes = 50
        #z_eval = np.linspace(np.min(est_z_s), np.max(est_z_s), n_nodes)
        if dataset == 's17':
            z_eval = np.linspace(0.0, 1.25, n_nodes)
        else:
            z_eval = np.linspace(0.0, np.max(est_z_s), n_nodes)
        if mu_exp:
            if inv_ladder:
                h_0_samples = gd_samples.getParams().h_0
            q_0_samples = gd_samples.getParams().q_0
            j_0_samples = gd_samples.getParams().j_0
        else:
            if inv_ladder:
                h_0_samples = gd_samples.getParams().h_0
            om_m_samples = gd_samples.getParams().Omega_m
        h_0_samples_cmb = cmb_samples.getParams().H0
        om_m_samples_cmb = cmb_samples.getParams().omegam
        for i in range(n_nodes):
            if mu_exp:
                if inv_ladder:
                    gd_samples.addDerived(h_z_expansion(z_eval[i], \
                                                        h_0_samples, \
                                                        q_0_samples, \
                                                        j_0_samples) / \
                                          (1.0 + z_eval[i]), \
                                          name = 'h_z_{:d}'.format(i), \
                                          label = 'h_z_{:d}'.format(i))
                else:
                    gd_samples.addDerived(h_z_expansion(z_eval[i], \
                                                        h_0_local, \
                                                        q_0_samples, \
                                                        j_0_samples) / \
                                          (1.0 + z_eval[i]), \
                                          name = 'h_z_{:d}'.format(i), \
                                          label = 'h_z_{:d}'.format(i))
            else:
                if inv_ladder:
                    gd_samples.addDerived(h_z(z_eval[i], \
                                              h_0_samples, \
                                              1.0 - om_m_samples, \
                                              om_m_samples, 0.0) / \
                                          (1.0 + z_eval[i]), \
                                          name = 'h_z_{:d}'.format(i), \
                                          label = 'h_z_{:d}'.format(i))
                else:
                    gd_samples.addDerived(h_z(z_eval[i], \
                                              h_0_local, \
                                              1.0 - om_m_samples, \
                                              om_m_samples, 0.0) / \
                                          (1.0 + z_eval[i]), \
                                          name = 'h_z_{:d}'.format(i), \
                                          label = 'h_z_{:d}'.format(i))
            pars.append('h_z_{:d}'.format(i))
            cmb_samples.addDerived(h_z(z_eval[i], \
                                       h_0_samples_cmb, \
                                       1.0 - om_m_samples_cmb, \
                                       om_m_samples_cmb, 0.0) / \
                                   (1.0 + z_eval[i]), \
                                   name = 'h_z_{:d}'.format(i), \
                                   label = 'h_z_{:d}'.format(i))
        gd_samples.updateBaseStatistics()
        gd_stats = gd_samples.getMargeStats()
        cmb_samples.updateBaseStatistics()
        pla_stats = cmb_samples.getMargeStats()
        lo_lim = np.zeros(n_nodes)
        up_lim = np.zeros(n_nodes)
        lo_lim_cmb = np.zeros(n_nodes)
        up_lim_cmb = np.zeros(n_nodes)
        for i in range(0, n_nodes):
            #lim_type = gd_stats.parWithName(pars[i]).limits[j].limitType()
            lo_lim[i] = gd_stats.parWithName('h_z_{:d}'.format(i)).limits[0].lower
            up_lim[i] = gd_stats.parWithName('h_z_{:d}'.format(i)).limits[0].upper
            lo_lim_cmb[i] = pla_stats.parWithName('h_z_{:d}'.format(i)).limits[0].lower
            up_lim_cmb[i] = pla_stats.parWithName('h_z_{:d}'.format(i)).limits[0].upper
        mp.fill_between(z_eval, lo_lim, up_lim, color=cbf_cols[0], \
                        alpha=0.5, edgecolor='None')
        mp.fill_between(z_eval, lo_lim_cmb, up_lim_cmb, \
                        color=cbf_cols[5], alpha=0.5, edgecolor='None')

    # finish plot
    if inv_ladder:

        # overplot best-fit H(z) curve
        if mu_exp:
            h_z_ax.plot(z_sn, \
                        h_z_expansion(z_sn, mp_h_0, mp_q_0, mp_j_0) / \
                        (1.0 + z_sn), color=cbf_cols[0])
        else:
            h_z_ax.plot(z_sn, \
                        h_z(z_sn, mp_h_0, 1.0 - mp_om_m, mp_om_m, 0.0) / \
                        (1.0 + z_sn), color=cbf_cols[0])

        # plot H_0 posterior(s) on separate axes
        n_h_0_grid = 100
        h_0_grid = np.linspace(60.0, 80.0, n_h_0_grid)
        p_h_0 = gd_samples.get1DDensity('h_0')
        p_h_0_norm = np.sum(p_h_0.Prob(h_0_grid)) * \
                     (h_0_grid[-1] - h_0_grid[0]) / n_h_0_grid
        ppd_ax.plot(p_h_0.Prob(h_0_grid) / p_h_0_norm, \
                    h_0_grid, color=cbf_cols[0], zorder=20)
        ppd_ax.invert_xaxis()

        # shade one-sigma region(s)
        y_range = np.linspace(np.mean(h_0_samples) - \
                              np.std(h_0_samples), \
                              np.mean(h_0_samples) + \
                              np.std(h_0_samples), 100)
        ppd_ax.fill_betweenx(y_range, \
                             p_h_0.Prob(y_range) / p_h_0_norm, \
                             color=cbf_cols[0], alpha=0.5, \
                             edgecolor='None', zorder=20)

        # optionally overlay Planck/WMAP's H_0 PPD too
        if ol_cmb_h_0_ppd:

            # plot H_0 posterior(s) on separate axes
            h_0_samples = cmb_samples.getParams().H0
            p_h_0 = cmb_samples.get1DDensity('H0')
            p_h_0_norm = np.sum(p_h_0.Prob(h_0_grid)) * \
                         (h_0_grid[-1] - h_0_grid[0]) / n_h_0_grid
            ppd_ax.plot(p_h_0.Prob(h_0_grid) / p_h_0_norm, \
                        h_0_grid, color=cbf_cols[5], ls='--', \
                        zorder=10)

            # shade one-sigma region
            y_range = np.linspace(np.mean(h_0_samples) - \
                                  np.std(h_0_samples), \
                                  np.mean(h_0_samples) + \
                                  np.std(h_0_samples), 100)
            ppd_ax.fill_betweenx(y_range, p_h_0.Prob(y_range) / \
                                 p_h_0_norm, \
                                 color=cbf_cols[5], alpha=0.5, \
                                 edgecolor='None', zorder=10)

        # chuck on SN n(z) plot f't'l0lz
        if dataset == 's17':
            n_z_ax.hist(s17_data['zcmb'], bins=100, \
                        color=cbf_cols[0], alpha=0.5)
            n_z_ax.hist(s17_data['zcmb'], bins=100, \
                        facecolor='None', edgecolor=cbf_cols[0])
        else:
            n_z_ax.hist(b14_data['zcmb'], bins=100, \
                        color=cbf_cols[0], alpha=0.5)
            n_z_ax.hist(b14_data['zcmb'], bins=100, \
                        facecolor='None', edgecolor=cbf_cols[0])

    else:

        # overplot best-fit H(z) curve
        if mu_exp:
            h_z_ax.plot(z_sn, \
                        h_z_expansion(z_sn, h_0_local, mp_q_0, mp_j_0) / \
                        (1.0 + z_sn), color = 'Teal')
        else:
            h_z_ax.plot(z_sn, \
                        h_z(z_sn, h_0_local, 1.0 - mp_om_m, mp_om_m, 0.0) / \
                        (1.0 + z_sn), color = 'Teal')

    #h_z_ax.errorbar(z_bao, h_z_bao / (1.0 + z_bao), \
    #                sig_h_z_bao / (1.0 + z_bao), linestyle='None', \
    #                color='Coral', zorder=4)
    for zb in z_bao:
        h_z_ax.axvline(zb, ls='--', dashes=(3, 3), \
                       color=(89.0 / 255.0, 164.0 / 255.0, 139.0 / 255.0), zorder=0)
        #h_z_ax.axvline(zb, ls='-', color=cbf_cols[3], zorder=4)
    h_z_ax.plot(z_sn, h_z(z_sn, h_0_cmb, om_l_cmb, \
                            om_m_cmb, om_r_cmb) / \
                        (1.0 + z_sn), ls='--', color=cbf_cols[5])
    h_z_ax.set_xlabel(r'$z$', fontsize=18)
    h_z_ax.set_ylabel(r'$H(z)/(1+z) \, ' + \
                      r'[{\rm km}\,{\rm s}^{-1}\,{\rm Mpc}^{-1}]$', \
                      fontsize=18)
    h_z_ax.set_ylim(55.0, 80.0)
    h_z_ax.tick_params(axis='both', which='major', labelsize=16)
    if inv_ladder:
        h_z_ax.yaxis.tick_right()
        h_z_ax.yaxis.set_ticks_position('both')
        h_z_ax.yaxis.set_label_position('right')
        yticks = h_z_ax.yaxis.get_major_ticks()
        yticks[-1].label2.set_visible(False)
        ppd_ax.set_ylim(h_z_ax.get_ylim())
        ppd_ax.set_ylabel(r'$H_0 \, ' + \
                          r'[{\rm km}\,{\rm s}^{-1}\,{\rm Mpc}^{-1}]$', \
                          fontsize=18)
        ppd_ax.set_xlabel(r'${\rm Pr}(H_0|d)$', fontsize=18)
        #ppd_ax.axhline(h_0_local, color=cbf_cols[3])
        ppd_ax.plot(sps.norm.pdf(h_0_grid, h_0_local, \
                                 sig_h_0_local), \
                    h_0_grid, color=cbf_cols[3])
        y_range = np.linspace(h_0_local - sig_h_0_local, \
                              h_0_local + sig_h_0_local, 100)
        ppd_ax.fill_betweenx(y_range, \
                             sps.norm.pdf(y_range, h_0_local, \
                                          sig_h_0_local), \
                             color=cbf_cols[3], alpha=0.5, \
                             edgecolor='None', zorder=10)
        ppd_ax.locator_params(axis='x', nbins=3)
        ppd_ax.tick_params(axis='both', which='major', labelsize=16)
        ppd_ax.set_xlim(0.45, 0.0)
        xticks = ppd_ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        #ppd_ax.set_xticks([0.0, 0.5, 1.0])
        #ppd_ax.set_xticklabels([0.0, 0.5, 1.0])
        n_z_ax.set_ylabel(r'$N({\rm SN})$', fontsize=18)
        n_z_ax.yaxis.tick_right()
        n_z_ax.yaxis.set_ticks_position('both')
        n_z_ax.yaxis.set_label_position('right')
        n_z_ax.tick_params(axis='both', which='major', labelsize=16)
    if dataset == 's17':
        h_z_ax.set_xlim(0.0, 1.25)
        if inv_ladder:
            n_z_ax.set_xlim(h_z_ax.get_xlim())
    else:
        if np.max(est_z_s) > 1.0:
            h_z_ax.set_xlim(0.0, np.max(est_z_s))
            if inv_ladder:
                n_z_ax.set_xlim(h_z_ax.get_xlim())
    if ol_h_z_samples:
        mp.savefig(base + '.pdf', bbox_inches='tight')
    else:
        mp.savefig(base + '_shaded.pdf', bbox_inches='tight')
    #mp.show()

    # posterior predictive plots
    if inv_ladder:

        # predict H_0! already have samples of H_0 with a uniform
        # prior. just need to simulate observations
        ppd_fig, ppd_ax = mp.subplots()
        h_0_samples = gd_samples.getParams().h_0
        gd_samples.addDerived(h_0_samples + \
                              npr.randn(n_samples) * sig_h_0_local, \
                              name='h_0_obs', label=r'\hat{H}_0')
        pars.append('h_0_obs')
        gd_samples.updateBaseStatistics()
        n_h_0_grid = 1000
        h_0_grid = np.linspace(60.0, 76.0, n_h_0_grid)
        p_h_0 = gd_samples.get1DDensity('h_0')
        p_h_0_norm = np.sum(p_h_0.Prob(h_0_grid)) * \
                     (h_0_grid[-1] - h_0_grid[0]) / n_h_0_grid
        #label = r'SNe+BAO+$r_d$' + '\n' + 'posterior'
        #label = r'${\rm Pr}(H_0|\hat{\bm{m}},\hat{\bm{\alpha}},I)$'
        label = r'${\rm Pr}(H_0|{\rm SNe}+{\rm BAO}+r_d)$'
        #label = r'$H_0$ posterior' + '\n' + \
        #        r'given SNe+BAO+$r_d$'
        ppd_ax.plot(h_0_grid, p_h_0.Prob(h_0_grid) / p_h_0_norm, \
                    color=cbf_cols[0], ls='--', zorder=20, \
                    label=label)
        p_h_0_obs = gd_samples.get1DDensity('h_0_obs')
        p_h_0_obs_norm = np.sum(p_h_0_obs.Prob(h_0_grid)) * \
                                (h_0_grid[-1] - h_0_grid[0]) / n_h_0_grid
        #label = r'SNe+BAO+$r_d$' + '\n' + 'prediction'
        #label = r'${\rm Pr}(\hat{H}_0^{\rm CDL}|\hat{\bm{m}},' + \
        #        r'\hat{\bm{\alpha}},I)$'
        label = r'${\rm Pr}(\hat{H}_0^{\rm CDL}|' + \
                r'{\rm SNe}+{\rm BAO}+r_d)$'
        #label = r'$\hat{H}_0^{\rm CDL}$ prediction' + '\n' + \
        #        r'given SNe+BAO+$r_d$'
        ppd_ax.plot(h_0_grid, p_h_0_obs.Prob(h_0_grid) / p_h_0_obs_norm, \
                    color=cbf_cols[0], zorder=20, \
                    label=label)
        ppd_ax.fill_between(h_0_grid, \
                            p_h_0_obs.Prob(h_0_grid) / p_h_0_obs_norm, \
                            color=cbf_cols[0], zorder=20, alpha=0.3)

        # optionally overlay Planck/WMAP's H_0 PPD too
        if ol_cmb_h_0_ppd:

            h_0_samples = cmb_samples.getParams().H0
            cmb_samples.addDerived(h_0_samples + \
                                   npr.randn(len(h_0_samples)) * \
                                   sig_h_0_local, name='h_0_obs', \
                                   label=r'\hat{H}_0')
            p_h_0 = cmb_samples.get1DDensity('H0')
            p_h_0_norm = np.sum(p_h_0.Prob(h_0_grid)) * \
                                (h_0_grid[-1] - h_0_grid[0]) / n_h_0_grid
            #label = r'CMB $\Lambda$CDM' + '\n' + \
            #        'posterior'
            label = r'${\rm Pr}(H_0|{\rm CMB}+\Lambda{\rm CDM})$'
            #label = r'$H_0$ posterior' + '\n' + \
            #        r'given CMB+$\Lambda$CDM'
            ppd_ax.plot(h_0_grid, p_h_0.Prob(h_0_grid) / p_h_0_norm, \
                        color=cbf_cols[5], ls='--', zorder=10, \
                        label=label)
            p_h_0_obs = cmb_samples.get1DDensity('h_0_obs')
            p_h_0_obs_norm = np.sum(p_h_0_obs.Prob(h_0_grid)) * \
                                    (h_0_grid[-1] - h_0_grid[0]) / n_h_0_grid
            #label = r'CMB $\Lambda$CDM' + '\n' + \
            #        'prediction'
            label = r'${\rm Pr}(\hat{H}_0^{\rm CDL}|' + \
                    r'{\rm CMB}+\Lambda{\rm CDM})$'
            #label = r'$\hat{H}_0^{\rm CDL}$ prediction' + '\n' + \
            #        r'given CMB+$\Lambda$CDM'
            ppd_ax.plot(h_0_grid, p_h_0_obs.Prob(h_0_grid) / p_h_0_obs_norm, \
                        color=cbf_cols[5], zorder=10, \
                        label=label)
            ppd_ax.fill_between(h_0_grid, \
                                p_h_0_obs.Prob(h_0_grid) / p_h_0_obs_norm, \
                                color=cbf_cols[5], alpha=0.3, zorder=10)

        elif ol_wmap_r_d is not None:

            print '* reading more samples!'
            ol_chain = base.replace(dataset, dataset + '_wmap_r_d') + \
                                    '_chain.csv'
            ol_samples = np.genfromtxt(ol_chain, delimiter=",", \
                                       skip_header=2)
            n_ol_samples = len(ol_samples)
            ol_gd_samples = gd.MCSamples(samples = ol_samples, \
                                         names = pars[0: 5], 
                                         labels = par_names, \
                                         ranges = par_ranges)
            h_0_samples = ol_gd_samples.getParams().h_0
            ol_gd_samples.addDerived(h_0_samples + \
                                     npr.randn(n_ol_samples) * \
                                     sig_h_0_local, name='h_0_obs', \
                                     label=r'\hat{H}_0')
            ol_gd_samples.updateBaseStatistics()
            p_h_0 = ol_gd_samples.get1DDensity('h_0')
            p_h_0_norm = np.sum(p_h_0.Prob(h_0_grid)) * \
                                (h_0_grid[-1] - h_0_grid[0]) / n_h_0_grid
            ppd_ax.plot(h_0_grid, p_h_0.Prob(h_0_grid) / p_h_0_norm, \
                        color=cbf_cols[2], ls='--', zorder=15, \
                        label=r'SNe+BAO+$r_d^{\rm WMAP}$' + '\n' + \
                              'posterior')
            p_h_0_obs = ol_gd_samples.get1DDensity('h_0_obs')
            p_h_0_obs_norm = np.sum(p_h_0_obs.Prob(h_0_grid)) * \
                                    (h_0_grid[-1] - h_0_grid[0]) / n_h_0_grid
            ppd_ax.plot(h_0_grid, p_h_0_obs.Prob(h_0_grid) / p_h_0_obs_norm, \
                        color=cbf_cols[2], zorder=15, \
                        label=r'SNe+BAO+$r_d^{\rm WMAP}$' + '\n' + \
                              'prediction')
            ppd_ax.fill_between(h_0_grid, \
                                p_h_0_obs.Prob(h_0_grid) / p_h_0_obs_norm, \
                                color=cbf_cols[2], zorder=15, alpha=0.3)
        
        # finish plot
        #label = r'Riess et al.' + '\n' + '(2016) mean'
        label = r'${\rm observed}\,\hat{H}_0^{\rm CDL}$'
        ppd_ax.axvline(h_0_local, color=cbf_cols[3], zorder=30, \
                       label=label)
        ppd_ax.set_ylim(0.0, 0.5)
        ppd_ax.set_xlabel(r'$H_0 \, ' + \
                          r'[{\rm km}\,{\rm s}^{-1}\,{\rm Mpc}^{-1}]$', \
                          fontsize=18)
        ppd_ax.set_ylabel(r'${\rm Pr}(H_0|d)$',fontsize=18)
        leg = ppd_ax.legend(loc='upper left', fontsize=12)
        #leg = ppd_ax.legend(loc='upper center', fontsize=12, ncol=2)
        leg.set_zorder(40)
        ppd_ax.tick_params(axis='both', \
                           which='major', \
                           labelsize=16)
        if not ol_cmb_h_0_ppd and ol_wmap_r_d:
            mp.savefig(base + '_ppd_ol_wmap_r_d.pdf', \
                       bbox_inches='tight')
        else:
            mp.savefig(base + '_ppd.pdf', bbox_inches='tight')

        # report tension statistic
        post_means = gd_samples.getMeans()
        ind = gd_samples.getParamNames().numberOfName('h_0_obs')
        p_h_0_obs = gd_samples.get1DDensity('h_0_obs')
        loc_pr = p_h_0_obs.Prob(h_0_local)[0] / \
                 p_h_0_obs.Prob(post_means[ind])[0]
        loc_ppd_max_res = so.minimize(gd_opt_wrapper, \
                                      h_0_local, \
                                      args=(p_h_0_obs))
        loc_ppd_max = loc_ppd_max_res['x'][0]
        loc_pr_alt = p_h_0_obs.Prob(h_0_local)[0] / \
                     p_h_0_obs.Prob(loc_ppd_max)[0]
        print 'mean H_0: {:11.5e}'.format(post_means[ind])
        print 'PPD @ mean: {:11.5e}'.format(p_h_0_obs.Prob(post_means[ind])[0])
        print 'PPD @ mode: {:11.5e}'.format(p_h_0_obs.Prob(loc_ppd_max)[0])
        print 'PPD @ meas: {:11.5e}'.format(p_h_0_obs.Prob(h_0_local)[0])
        print 'PPD @ meas/mean: {:11.5e}'.format(loc_pr)
        print 'PPD @ meas/mode: {:11.5e}'.format(loc_pr_alt)
        
    else:

        # predict the BAO measurements!
        # add H(z_bao) and x(z_bao) constraints as derived parameters
        bao_fig = mp.figure()
        z_bao = z_bao[0: 3]
        h_z_bao = h_z_bao[0: 3]
        z_eval = z_bao
        n_z_eval = len(z_eval)
        if np.max(est_z_s) > 1.0:
            z_plot = np.linspace(np.min(est_z_s), np.max(est_z_s))
        else:
            z_plot = np.linspace(np.min(est_z_s), 1.0)
        q_0_samples = gd_samples.getParams().q_0
        j_0_samples = gd_samples.getParams().j_0
        h_0_samples = npr.randn(n_samples) * sig_h_0_local + h_0_local
        #h_z_bao_samples = np.zeros((n_samples, n_z_eval))
        h_z_bao_samples = []
        for i in range(n_z_eval):

            # draw H_0 from "prior"; draw H(z_bao) from likelihood given
            # current posterior sample
            h_z_bao_mean = h_z_expansion(z_eval[i], h_0_samples, \
                                         q_0_samples, j_0_samples)
            h_z_bao_samples.append((npr.randn(n_samples) * sig_h_z_bao[i] + \
                                    h_z_bao_mean) / (1.0 + z_eval[i]))
            #h_z_bao_samples[:, i] = (npr.randn(n_samples) * sig_h_z_bao[i] + \
            #                        h_z_bao_mean) / (1.0 + z_eval[i])
            gd_samples.addDerived(h_z_bao_samples[-1], \
                                  name = 'h_z_bao_{:d}'.format(i), \
                                  label = 'h_z_bao_{:d}'.format(i))
            pars.append('h_z_bao{:d}'.format(i))
            
        gd_samples.updateBaseStatistics()

        mp.plot(z_sn, \
                h_z_expansion(z_sn, h_0_local, mp_q_0, mp_j_0) / \
                              (1.0 + z_sn), color = 'Gray')
        #mp.plot(z_bao, h_z_bao / (1.0 + z_bao), linestyle = 'None', \
        #        marker='_', markersize=8, mew=1.5, color = 'Coral')
        mp.plot(z_plot, h_z(z_plot, h_0_cmb, om_l_cmb, \
                            om_m_cmb, om_r_cmb) / \
                            (1.0 + z_plot), 'k')
        if box_not_vio:
            bpd = mp.boxplot(h_z_bao_samples, positions=z_eval, sym='', \
                             whis=[1, 99], manage_xticks=False, \
                             widths=0.03)#, widths=0.08)
            for bpd_key in bpd:
                for bp in bpd[bpd_key]:
                    bp.set_color('Teal')
                    bp.set_ls('-')
        else:

            # plot complete violin plots
            vpd = mp.violinplot(h_z_bao_samples, z_eval, \
                                showextrema=False, \
                                showmedians=False, widths=0.08)
            for vpd_key in vpd:
                if vpd_key[0] == 'c':
                    vpd[vpd_key].set_color('Teal')
            for vp in vpd['bodies']:
                vp.set_color('Teal')

            # highlight portion more extreme than data. first loop through 
            # the plotted polygons to obtain their vertices
            i = 0
            for vp in vpd['bodies']:
                paths = vp.get_paths()
                for path in paths:

                    # for each polygon, find region more extreme than data
                    # and first blank out, then recolour
                    verts = path.vertices
                    inds = verts[:, 1] < h_z_bao[i] / (1.0 + z_bao[i])
                    patch = mpp.Polygon(verts[inds], closed=False, \
                                        fc='White', ec='White', fill=True, \
                                        visible=True)
                    mp.gca().add_patch(patch)
                    patch = mpp.Polygon(verts[inds], closed=False, \
                                        fc='Coral', ec='Coral', fill=True, \
                                        alpha = 0.5, visible=True)
                    mp.gca().add_patch(patch)
                    
                    # next define long tails, find vertices in tails and 
                    # blank them out
                    delta_x = (np.max(verts[:, 0]) - \
                               np.min(verts[:, 0])) / 2.0
                    mean_x = np.mean(verts[:, 0])
                    mean_y = np.mean(verts[:, 1])
                    inds = (np.abs(verts[:, 0] - mean_x) < delta_x / 50.0) & \
                           (verts[:, 1] < mean_y)
                    patch = mpp.Polygon(verts[inds], closed=False, \
                                        fc='White', ec='White', fill=True, \
                                        visible=True)
                    mp.gca().add_patch(patch)
                    inds = (np.abs(verts[:, 0] - mean_x) < delta_x / 50.0) & \
                           (verts[:, 1] > mean_y)
                    patch = mpp.Polygon(verts[inds], closed=False, \
                                        fc='White', ec='White', fill=True, \
                                        visible=True)
                    mp.gca().add_patch(patch)
                i += 1
        mp.xlabel(r'$z$')
        mp.ylabel(r'$H(z)/(1+z) \, ' + \
                  r'[{\rm km}\,{\rm s}^{-1}\,{\rm Mpc}^{-1}]$')
        if np.max(est_z_s) > 1.0:
            mp.xlim(0.0, np.max(est_z_s))
        mp.ylim(53, 73)
        if box_not_vio:
            mp.savefig(base + '_ppd_box.pdf', bbox_inches = 'tight')
        else:
            mp.savefig(base + '_ppd.pdf', bbox_inches = 'tight')
