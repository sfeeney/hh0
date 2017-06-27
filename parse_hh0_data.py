import numpy as np
import re
import os
import matplotlib.pyplot as mp

c = 299792.458 # km s^-1

def z2d(z):
    q_0 = -0.55
    j_0 = 1.0
    return c * z * (1.0 + 0.5 * (1.0 - q_0) * z - \
                    (1.0 - q_0 - 3.0 * q_0 ** 2 + j_0) * z ** 2 / 6.0)

# define Riess et al fitting and rejection functions
def riess_sn_fit(app_mag_s, app_mag_err_s, z_s, sig_int_s):
    
    # helpful parameters. only fitting an intercept here
    n_s = len(app_mag_s)
    n_obs = n_s
    n_par = 1
    y_vec = np.zeros(n_obs)
    l_mat = np.zeros((n_obs, n_par))
    c_mat_inv = np.zeros((n_obs, n_obs))

    # loop through SNe
    k = 0
    for i in range(0, n_s):
        y_vec[k] = np.log10(z2d(z_s[i])) - 0.2 * app_mag_s[i]
        l_mat[k, 0] = 1.0
        c_mat_inv[k, k] = 1.0 / 0.2 ** 2 / \
                          (app_mag_err_s[i] ** 2 + sig_int_s ** 2)
        k += 1

    # fit, calculate residuals in useable form and return
    ltci = np.dot(l_mat.transpose(), c_mat_inv)
    q_hat_cov = np.linalg.inv(np.dot(ltci, l_mat))
    q_hat = np.dot(np.dot(q_hat_cov, ltci), y_vec)
    res = y_vec - np.dot(l_mat, q_hat)
    return q_hat, np.sqrt(np.diag(q_hat_cov)), res

def riess_sn_reject(app_mag_err_s, sig_int_s, res, threshold = 3.0):

    res_scaled = np.abs(res) / 0.2 / \
                 np.sqrt(app_mag_err_s ** 2 + sig_int_s ** 2)
    to_rej = np.argmax(res_scaled)
    if res_scaled[to_rej] > threshold:
        return to_rej
    else:
        return None

def riess_sn_delete(i_del, target):
    for i in range(len(target)):
        if target is not None:
            target[i] = np.concatenate((target[i][0: i_del], \
                                        target[i][i_del + 1:]))
    return target

def hh0_parse(dataset="r16", fix_redshifts=True, inc_met_dep=True, \
              model_outliers=None, verbose=False):

    # define hosts based on analysis
    if dataset == "r16_one_anc":
        d_anchors = ["N4258"]
        p_anchors = []
        ceph_only_hosts = ["M31"]
        ceph_sn_hosts = ["N3021", "N3370", "N1309", "N3982", "N4639", \
                         "N5584", "N4038", "N4536", "N1015", "N1365", \
                         "N1448", "N3447", "N7250", "N5917", "N4424", \
                         "U9391", "N3972", "N2442", "M101"]
    elif dataset == "r16":
        d_anchors = ["N4258", "LMC"]
        p_anchors = ["lCar", "etaGem", "betaDor", "WSgr", "XSgr", \
                     "YSgr", "delCeph", "FFAql", "TVul", "RTAur", \
                     "SUCas", "BGCru", "DTCyg", "SYAur", "SSCMa"]
        ceph_only_hosts = ["M31"]
        ceph_sn_hosts = ["N3021", "N3370", "N1309", "N3982", "N4639", \
                         "N5584", "N4038", "N4536", "N1015", "N1365", \
                         "N1448", "N3447", "N7250", "N5917", "N4424", \
                         "U9391", "N3972", "N2442", "M101"]
    else:
        exit("ERROR: unknown dataset!")
    ceph_hosts = d_anchors + p_anchors + ceph_only_hosts + ceph_sn_hosts

    # set anchor distances
    # * D(NGC 4258) = 7.54 +/- 0.17(stat.) +/- 0.10(sys.) Mpc (R16)
    # * D(LMC) = 49.97 +/- 0.19 (stat.) +/- 1.11 (sys.) kpc
    #   (http://www.nature.com/nature/journal/v495/n7439/full/nature11878.html)
    # all anchor distances are in Mpc
    dis_anc = np.array([7.54e6, 49.97e3]) / 1.0e6
    sig_dis_anc = np.array([np.sqrt(0.17e6 ** 2 + 0.10e6 ** 2), \
                            np.sqrt(0.19e3 ** 2 + 1.11e3 ** 2)]) / \
                  1.0e6

    # set anchor parallaxes
    # * http://mnras.oxfordjournals.org/content/379/2/723.full: 
    #   "lCar", "etaGem", "betaDor", "WSgr", "XSgr", "YSgr", "delCeph", 
    #   "FFAql", "TVul", "RTAur", "SUCas", "BGCru", "DTCyg")
    # * Riess et al. 2014: SYAur
    # * Casertano et al. 2016: SSCMa
    # all parallaxes are measure in mas
    par_anc = np.array([2.03, 2.74, 3.26, 2.30, 3.17, \
                        2.13, 3.71, 2.64, 2.06, 2.31, \
                        2.57, 2.23, 2.19, 0.428, 0.348])
    sig_par_anc = np.array([0.16, 0.12, 0.14, 0.19, 0.14, \
                            0.29, 0.12, 0.16, 0.22, 0.19, \
                            0.33, 0.30, 0.33, 0.054, 0.038])
    par_anc_lkc = np.array([-0.05, -0.02, -0.02, -0.06, -0.02, \
                            -0.15, -0.01, -0.03, -0.09, -0.06, \
                            -0.13, -0.15, -0.18, -0.04, -0.04])

    # dimension Cepheid / SN host arrays
    n_ch = len(ceph_hosts)
    n_ch_d = len(d_anchors)
    n_ch_p = len(p_anchors)
    n_ch_c = len(ceph_only_hosts)
    n_ch_s = len(ceph_sn_hosts)
    n_c_ch = np.zeros(n_ch)
    est_app_mag_s_ch = np.zeros(n_ch_s)
    sig_app_mag_s_ch = np.zeros(n_ch_s)
    zp_off_mask = np.zeros(n_ch)
    if dataset == "r16":
        zp_off_mask[1: n_ch_p + 2] = 1.0

    # compile geometric measurements
    if n_ch_p == 0:
        dis_anc = dis_anc[0: n_ch_d]
        sig_dis_anc = sig_dis_anc[0: n_ch_d]
        par_anc_lkc = []
    else:
        dis_anc = np.concatenate([dis_anc[0: n_ch_d], \
                                  par_anc[0: n_ch_p]])
        sig_dis_anc = np.concatenate([sig_dis_anc[0: n_ch_d], \
                                      sig_par_anc[0: n_ch_p]])
        par_anc_lkc = par_anc_lkc[0: n_ch_p]

    # read in Cepheid / SN host data
    cut_outs = False
    cut_low_p = False
    r_wesenheit = 0.39
    sig_int_c_r16 = 0.065# 0.08
    sig_int_s_r16 = 0.1
    sig_v_pec = 250.0
    est_q_0 = -0.5575 # Betoule et al. 2014
    sig_q_0 =  0.0510 # Betoule et al. 2014
    sig_zp_off = 0.03
    pardir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pardir, 'data/R16_table4.out')) as f:
        
        # dimension and fill Cepheid/SN host arrays
        n_skip = 40
        for i, l in enumerate(f):
            if (i > n_skip - 1):
                vals = [val for val in l.split()]
                if len(vals) == 0:
                    break
                for j in range(0, n_ch_s):
                    if vals[0] == ceph_sn_hosts[j]:
                        est_app_mag_s_ch[j] = float(vals[2])
                        sig_app_mag_s_ch[j] = np.sqrt(float(vals[3]) ** 2 - \
                                              sig_int_s_r16 ** 2)
                        break
        
        # pass through file to find Cepheid counts
        n_skip = 70
        f.seek(0)
        for i, l in enumerate(f):
            if (i > n_skip - 1):
                vals = [val for val in l.split()]
                if len(vals) == 0:
                    break
                if vals[0] == "Galaxy":
                    for j in range(n_ch_d, n_ch_d + n_ch_p):
                        if vals[3] == ceph_hosts[j]:
                            n_c_ch[j] += 1
                            break
                else:
                    for j in range(0, n_ch):
                        if vals[0].lower() == ceph_hosts[j].lower():
                            n_c_ch[j] += 1
                            break
        n_c_ch = np.array(n_c_ch, dtype = int)
        n_c_tot = np.sum(n_c_ch)

        # dimension and fill Cepheid arrays
        est_app_mag_c = np.zeros((n_ch, np.max(n_c_ch)))
        sig_app_mag_c = np.zeros((n_ch, np.max(n_c_ch)))
        est_p_c = np.zeros((n_ch, np.max(n_c_ch)))
        est_z_c = np.zeros((n_ch, np.max(n_c_ch)))
        f.seek(0)
        k = np.zeros(n_ch, dtype = int)
        for i, l in enumerate(f):
            if (i > n_skip - 1):
                vals = [val for val in l.split()]
                if len(vals) == 0:
                    break
                if vals[0] == "Galaxy":
                    for j in range(n_ch_d, n_ch_d + n_ch_p):
                        if vals[3] == ceph_hosts[j]:
                            est_app_mag_c[j, k[j]] = float(vals[6]) - \
                                                     r_wesenheit * float(vals[5])
                            sig_app_mag_c[j, k[j]] = np.sqrt(float(vals[7]) ** 2 - \
                                                     sig_int_c_r16 ** 2)
                            est_p_c[j, k[j]] = float(vals[4])
                            est_z_c[j, k[j]] = float(vals[8])
                            k[j] += 1
                            break
                else:
                    for j in range(0, n_ch):
                        if vals[0].lower() == ceph_hosts[j].lower():
                            est_app_mag_c[j, k[j]] = float(vals[6]) - \
                                                     r_wesenheit * float(vals[5])
                            sig_app_mag_c[j, k[j]] = np.sqrt(float(vals[7]) ** 2 - \
                                                     sig_int_c_r16 ** 2)
                            est_p_c[j, k[j]] = float(vals[4])
                            est_z_c[j, k[j]] = float(vals[8])
                            k[j] += 1
                            break
    print "= Cepheid / SN hosts ="
    print 'total of {:d} Cepheids selected'.format(n_c_tot)
    if verbose:
        mean_sig_app_mag_c = 0.0
        if n_ch_p > 0:
            mean_sig_app_mag_c_p = \
                np.mean(np.sqrt(sig_app_mag_c[n_ch_d: n_ch_d + n_ch_p, 0] ** 2 + \
                                sig_int_c_r16 ** 2))
            print 'mean MW Cepheid app mag err: {:4.2f}'.format(mean_sig_app_mag_c_p)
        for i in range(0, n_ch):
            mean_sig_app_mag_c += np.sum(sig_app_mag_c[i, 0: n_c_ch[i]] ** 2)
            mean_sig_app_mag_c_ch = \
                np.mean(np.sqrt(sig_app_mag_c[i, 0: n_c_ch[i]] ** 2 + \
                                sig_int_c_r16 ** 2))
            mean_est_p_c_ch = np.mean(est_p_c[i, 0: n_c_ch[i]])
            if i < n_ch_d:
                print "{:10s} {:3d} {:4.2f} {:4.2f}".format(ceph_hosts[i], \
                    n_c_ch[i], mean_sig_app_mag_c_ch, mean_est_p_c_ch)
            elif i < n_ch_d + n_ch_p:
                print "{:10s} {:3d}".format(ceph_hosts[i], n_c_ch[i])
            elif i < n_ch_d + n_ch_p + n_ch_c:
                print "{:10s} {:3d} {:4.2f} {:4.2f}".format(ceph_hosts[i], \
                    n_c_ch[i], mean_sig_app_mag_c_ch, mean_est_p_c_ch)
            else:
                print "{:10s} {:3d} {:4.2f} {:4.2f} {:6.3f} {:5.3f}".format(ceph_hosts[i], \
                    n_c_ch[i], mean_sig_app_mag_c_ch, mean_est_p_c_ch, \
                    est_app_mag_s_ch[i - (n_ch_d + n_ch_p + n_ch_c)], \
                    np.sqrt(sig_app_mag_s_ch[i - (n_ch_d + n_ch_p + n_ch_c)] ** 2 + \
                                sig_int_s_r16 ** 2))
        mean_sig_app_mag_c = np.sqrt(mean_sig_app_mag_c / n_c_tot)
        print 'RMS Cepheid app mag err: {:5.3f}'.format(mean_sig_app_mag_c)
        rms_sig_app_mag_tot = np.sqrt(np.mean(sig_app_mag_s_ch ** 2 + sig_int_s_r16 ** 2))
        print 'RMS Cepheid-host SN app mag err: {:7.5f}'.format(np.sqrt(np.mean(sig_app_mag_s_ch ** 2)))
        print 'RMS Cepheid-host SN total app mag err: {:7.5f}'.format(rms_sig_app_mag_tot)
        
    # read in Hubble Flow SN data
    # Dan Scolnic's comments: 
    # User note: This list does not have duplicate SNe removed - most
    # of the SNe have the same name, but a handful of SDSS that have 
    # duplicates don't:
    print "= = = = = = = = = ="
    print "= Hubble Flow SNe ="
    dupe_sdss = ['16314', '16392', '16333', '14318', '17186', '17784', \
                 '7876']
    dupe_true = ['2006oa', '2006ob', '2006on', '2006py', '2007hx', \
                 '2007jg','2005ir']
    z_s_min = 0.0233 #0.01
    z_s_max = 0.15
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
    r16_sn_hosts = ["744", "1241", "1371", "1794", "2102", "2635", \
                    "2916", "2992", "5549", "6057", "6558", "7147", \
                    "8921", "12779", "12781", "12898", "12950", \
                    "13005", "13038", "13044", "13736", "13796", \
                    "14024", "14108", "14871", "15234", "16021", \
                    "16619", "16641", "17240", "17366", "17497", \
                    "18241", "18298", "18602", "18697", "18809", \
                    "18835", "18855", "19775", "19899", "19953", \
                    "20084", "20470", "21502", "722", "774", "1032", \
                    "2308", "2561", "3592", "5395", "5751", "6406", \
                    "8719", "10028", "10434", "10805", "11067", \
                    "11300", "12860", "12928", "12930", "13135", \
                    "13894", "14437", "15171", "15508", "16069", \
                    "16185", "16259", "17208", "17258", "17280", \
                    "17605", "17629", "17745", "18415", "18612", \
                    "18650", "19543", "19940", "19968", "20064", \
                    "20625", "21034", "21510", "22075", "06D2fb", \
                    "2004ef", "2005eq", "2005hc", "2005hf", "2005hj", \
                    "2005iq", "2005lz", "2005mc", "2005ms", "2005na", \
                    "2006ac", "2006ak", "2006al", "2006bb", "2006bu", \
                    "2006cj", "2006cq", "2006cz", "2006en", "2006ev", \
                    "2006gr", "2006je", "2006mo", "2006oa", "2006on", \
                    "2006qo", "2006sr", "2006te", "2007ae", "2007ai", \
                    "2007ba", "2007bd", "2007co", "2007cq", "2007F", \
                    "2007O", "2008af", "2008bf", "2010dt", "1997dg", \
                    "1998dx", "1998eg", "1999cc", "1999ef", "1999X", \
                    "2000cf", "2000cn", "2001ah", "2001az", "2002bf", \
                    "2002bz", "2002ck", "2002de", "2002G", "2002hd", \
                    "2003cq", "2003it", "2004as", "2004L", "2002he", \
                    "2003fa", "2003ic", "2003iv", "2003U", "2007aj", \
                    "2007cb", "2007cc", "2007is", "2007kh", "2007kk", \
                    "2007nq", "2007ob", "2007su", "2007ux", "2008Y", \
                    "2008ac", "2008ar", "2008at", "2008bw", "2008by", \
                    "2008bz", "2008cf", "2008fr", "2008gb", "2008gl", \
                    "2009D", "2009ad", "2008050", "2008051", "2004gu", \
                    "2005ag", "2005be", "2005ir", "2006eq", "2006lu", \
                    "2006ob", "2006py", "2007hx", "2008bq", "1993ac", \
                    "1994M", "1994Q", "1996ab", "1996bl", "1996C", \
                    "1990af", "1990O", "1990T", "1990Y", "1991S", \
                    "1991U", "1992ae", "1992aq", "1992bg", "1992bh", \
                    "1992bk", "1992bl", "1992bp", "1992bs", "1992J", \
                    "1992P", "1993ag", "1993H", "1993O", "2000bh", \
                    "2000bk", "2000ca", "2001ba"]
    with open(os.path.join(pardir, 'data/supercal_vH0.fitres.txt')) as f:

        # loop through full file
        for i, l in enumerate(f):
            if (i > n_skip - 1):
                vals = [val for val in l.split()]
                if len(vals) == 0:
                    break
                sn = vals[1]

                # check SN is on R16 list and is not a duplicate
                if model_outliers is None:
                    if sn not in r16_sn_hosts:
                        continue
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
                    x_1_err_temp < 1.5 and prob_temp > 0.001 and \
                    peak_time_temp < 2.0 and m_err_temp < 0.2):
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
                elif model_outliers is None:
                    s = '* SN {:s} fails cuts unexpectedly!'
                    print s.format(sn)
                    if (z_temp < z_s_min or z_temp > z_s_max):
                        print 'bad redshift: {:7.5f}'.format(z_temp)
                    if (np.abs(c_temp) >= 0.3):
                        print 'bad colour: {:7.4f}'.format(c_temp)
                    if (np.abs(x_1_temp) >= 3.0):
                        print 'bad stretch: {:7.4f}'.format(x_1_temp)
                    if (x_1_err_temp >= 1.5):
                        print 'bad stretch err: {:7.4f}'.format(x_1_err_temp)
                    if (prob_temp <= 0.001):
                        print 'bad prob: {:7.4f}'.format(prob_temp)
                    if (peak_time_temp >= 2.0):
                        print 'bad peak time: {:7.4f}'.format(peak_time_temp)
                    if (m_err_temp >= 0.2):
                        print 'bad mag err: {:7.4f}'.format(m_err_temp)
                    print '=> rejected'
    
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
    s = '{} SNe after cuts but before sigma-clipping'
    print s.format(n_s)
    s = 'RMS SN app mag err: {:7.5f}'
    print s.format(np.sqrt(np.mean(sig_app_mag_s ** 2)))
    rms_sig_app_mag_tot = np.sqrt(np.mean(sig_app_mag_s ** 2 + \
                                          sig_int_s_r16 ** 2))
    s = 'RMS SN total app mag err: {:7.5f}'
    print s.format(rms_sig_app_mag_tot)
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
    print 'RMS SN app mag err: {:7.5f}'.format(np.sqrt(np.mean(sig_app_mag_s ** 2)))
    rms_sig_app_mag_tot = np.sqrt(np.mean(sig_app_mag_s ** 2 + sig_int_s_r16 ** 2))
    print 'RMS SN total app mag err: {:7.5f}'.format(rms_sig_app_mag_tot)

    # return parsed data
    to_return = [n_ch_d, n_ch_p, n_ch_c, n_ch_s, n_c_ch, n_s, \
                 dis_anc, sig_dis_anc, est_app_mag_c, \
                 sig_app_mag_c, est_p_c, sig_int_c_r16, \
                 est_app_mag_s_ch, sig_app_mag_s_ch, est_app_mag_s, \
                 sig_app_mag_s, est_z_s, est_x_1_s, sig_x_1_s, \
                 est_c_s, sig_c_s, cov_x_1_app_mag_s, \
                 cov_c_app_mag_s, cov_x_1_c_s, sig_int_s_r16, \
                 est_q_0, sig_q_0, sig_zp_off, zp_off_mask, \
                 par_anc_lkc]
    if not fix_redshifts:
        to_return.append(sig_z_s)
        to_return.append(sig_v_pec)
    if inc_met_dep:
        to_return.append(est_z_c)
    return to_return
