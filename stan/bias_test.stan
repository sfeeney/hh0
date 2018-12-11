functions{
    real h2n(real h, vector a) {
        return a[1] + a[2] * h + a[3] * h ^ 2 + a[4] * h ^ 3 + a[5] * h ^ 4;
    }
    real hq2n(real h, real q, vector a) {
        return a[1] + a[2] * h + a[3] * q + a[4] * h ^ 2 + 
               a[5] * q ^ 2 + a[6] * h * q + a[7] * h ^ 3 + 
               a[8] * q ^ 3 + a[9] * q * h ^ 2 + a[10] * h * q ^ 2 + 
               a[11] * h ^ 4 + a[12] * q ^ 4 + a[13] * q * h ^ 3 + 
               a[14] * h * q ^ 3 + a[15] * (h * q) ^ 2;
    }
}
data {
    int<lower=0, upper=1> ntlo;        // use next-to-leading-order expansion
    int<lower=0, upper=1> vary_m_c;    // variable chirp mass
    int<lower=0, upper=1> z_dep_rate;  // redshift merger rate
    int<lower=0, upper=1> fixed_n_bns; // assume sample size known
    int<lower=0> n_bns;                // total number of mergers
    vector[n_bns] obs_amp_plus;        // measured plus amplitude
    vector[n_bns] obs_amp_cross;       // measured cross amplitude
    vector[n_bns] obs_v_pec;           // measured peculiar velocity
    vector[n_bns] obs_z;               // measured redshift
    vector[n_bns] obs_m_c_z;           // measured redshifted chirp masses
    real amp_s;                        // intrinsic GW amplitude
    real amp_n;                        // GW noise level
    real sig_v_pec;                    // std of true peculiar velocities
    real sig_obs_v_pec;                // noise on observed peculiar velocities
    real sig_z;                        // noise on observed redshifts
    real z_max;                        // maximum prior redshift
    real mu_m_c;                       // chirp mass prior mean
    real sig_m_c;                      // chirp mass prior std
    real sig_obs_m_c_z;                // noise on observed redshifted chirp masses
    int<lower=0> n_coeffs;             // number of coefficients of polynomial fit to \bar{N}(H_0,q_0)
    vector[n_coeffs] n_bar_det_coeffs; // coefficients of polynomial fit to \bar{N}(H_0,q_0)
}
transformed data {
    real c;                            // c in km/s
    real g;                            // g in Mpc / M_sol / (km/s)^2
    int n_m_c;                         // number of chirp masses to sample
    c = 2.99792458e5;
    g = 4.301e-9;
    n_m_c = vary_m_c * n_bns;
}
parameters {
    //real<lower=10.0, upper=200.0> h_0;
    //real<lower=-5.0, upper=1.0> q_0[ntlo];
    real<lower=50.0, upper=90.0> h_0;
    real<lower=-2.0, upper=1.0> q_0[ntlo];
    vector<lower=0.0, upper=z_max>[n_bns] true_z_cos;
    vector<lower=-1.0, upper=1.0>[n_bns] true_cos_i;
    vector[n_bns] true_v_pec;
    vector<lower=0.0, upper=10.0>[n_m_c] true_m_c;
}
transformed parameters {

    vector[n_bns] true_z;
    vector<lower=0.0>[n_bns] true_d;
    vector[n_bns] true_amp_plus;
    vector[n_bns] true_amp_cross;
    vector<lower=0.0>[n_m_c] true_m_c_z;
    real<lower=0.0> n_bar_det;

    for(i in 1:n_bns) {
        
        // pick order-appropriate distance-redshift relation
        if (ntlo) {
            true_z[i] = true_z_cos[i] + (1.0 + true_z_cos[i]) * 
                        true_v_pec[i] / c;
            true_d[i] = c * true_z_cos[i] / h_0 * 
                        (1.0 + 0.5 * (1.0 - q_0[1]) * true_z_cos[i]);
        } else {
            true_z[i] = true_z_cos[i] + true_v_pec[i] / c;
            true_d[i] = c * true_z_cos[i] / h_0;
        }
        if (true_d[i] < 0) {
            if (ntlo) {
                print("D BAD! ", true_d[i], " ", true_z_cos[i], " ", h_0, " ", q_0[1]);
            } else {
                print("D BAD! ", n_bar_det, " ", true_z_cos[i], " ", h_0);
            }
        }

        // different amplitudes if sampling chirp masses
        if (n_m_c > 0) {
            true_m_c_z[i] = true_m_c[i] * (1.0 + true_z[i]);
            true_amp_plus[i] = g * true_m_c_z[i] / c ^ 2 * 
                               (1.0 + true_cos_i[i] ^ 2) / 2.0 / true_d[i];
            true_amp_cross[i] = -g * true_m_c_z[i] / c ^ 2 * 
                                true_cos_i[i] / true_d[i];
        } else {
            true_amp_plus[i] = amp_s * (1.0 + true_cos_i[i] ^ 2) / 
                               2.0 / true_d[i];
            true_amp_cross[i] = -amp_s * true_cos_i[i] / true_d[i];
        }

    }
    if (ntlo) {
        n_bar_det = hq2n(h_0, q_0[1], n_bar_det_coeffs);
    } else {
        n_bar_det = h2n(h_0, n_bar_det_coeffs);
    }
    if (n_bar_det < 0) {
        if (ntlo) {
            print("N BAD! ", n_bar_det, " ", h_0, " ", q_0[1]);
        } else {
            print("N BAD! ", n_bar_det, " ", h_0);
        }
    }
    if (is_nan(n_bar_det)) {
        if (ntlo) {
            print("N BAD! ", n_bar_det, " ", h_0, " ", q_0[1]);
        } else {
            print("N BAD! ", n_bar_det, " ", h_0);
        }
    }
    
}
model {

    // priors on true parameters. priors on cos(i) are uniform
    h_0 ~ normal(70.0, 20.0);
    if (ntlo) {
        //q_0 ~ normal(-0.5, 1.0);
        q_0 ~ normal(-0.5, 0.5);
    }
    true_v_pec ~ normal(0.0, sig_v_pec);
    if (n_m_c > 0) {
        true_m_c ~ normal(mu_m_c, sig_m_c);
    }

    // pick order-appropriate volume element. NB: as addition acts 
    // per vector element, the statement below (correctly) applies a 
    // factor of 1/H_0^3 per object
    target += -3.0 * log(h_0) + 2.0 * log(true_z_cos);
    if (ntlo) {
        target += log(1.0 - 2.0 * (1.0 + q_0[1]) * true_z_cos);
    }
    if (z_dep_rate) {
        target += -log(1.0 + true_z_cos);
    }

    // Poisson exponent
    if (fixed_n_bns) {
        target += -n_bns * log(n_bar_det);
    } else {
        target += -n_bar_det;
    }

    // GW likelihoods
    obs_amp_plus ~ normal(true_amp_plus, amp_n);
    obs_amp_cross ~ normal(true_amp_cross, amp_n);
    if (n_m_c > 0) {

        // constraints on redshifted chirp masses
        obs_m_c_z ~ normal(true_m_c_z, sig_obs_m_c_z);
        target += -log(1.0 + true_z);

    }

    // EM likelihoods
    obs_v_pec ~ normal(true_v_pec, sig_obs_v_pec);
    obs_z ~ normal(true_z, sig_z);

}

