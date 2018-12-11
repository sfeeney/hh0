data {
    int<lower=0, upper=1> ntlo;        // use next-to-leading-order expansion
    int<lower=0, upper=1> vary_m_c;    // variable chirp mass
    int<lower=0> n_bns;                // total number of mergers
    vector[n_bns] obs_amp_plus;        // measured plus amplitude
    vector[n_bns] obs_amp_cross;       // measured cross amplitude
    vector[n_bns] obs_m_c_z;           // measured redshifted chirp masses
    real amp_s;                        // intrinsic GW amplitude
    real amp_n;                        // GW noise level
    real z_max;                        // maximum prior redshift
    real d_max;                        // maximum prior distance
    real mu_m_c;                       // chirp mass prior mean
    real sig_m_c;                      // chirp mass prior std
    real sig_obs_m_c_z;                // noise on observed redshifted chirp masses
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
    vector<lower=0.0, upper=z_max>[n_bns] true_z;
    vector<lower=0.0, upper=d_max>[n_bns] true_d;
    vector<lower=-1.0, upper=1.0>[n_bns] true_cos_i;
    vector<lower=0.0, upper=10.0>[n_m_c] true_m_c;
}
transformed parameters {

    vector[n_bns] true_amp_plus;
    vector[n_bns] true_amp_cross;
    vector<lower=0.0>[n_m_c] true_m_c_z;

    for(i in 1:n_bns) {

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
    
}
model {

    // priors on true parameters. priors on cos(i), true_z are uniform
    if (n_m_c > 0) {
        true_m_c ~ normal(mu_m_c, sig_m_c);
    }
    target += 2.0 * log(true_d);

    // GW likelihoods
    obs_amp_plus ~ normal(true_amp_plus, amp_n);
    obs_amp_cross ~ normal(true_amp_cross, amp_n);
    if (n_m_c > 0) {

        // constraints on redshifted chirp masses
        obs_m_c_z ~ normal(true_m_c_z, sig_obs_m_c_z);
        target += -log(1.0 + true_z);

    }

}

