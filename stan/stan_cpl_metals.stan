functions{
    real d2mu(real d) {
        return 5.0 * log10(d * 1.0e6) - 5.0;
    }
    real mu2d(real mu) {
        return 10.0 ^ (mu / 5.0 - 5.0); // in Mpc!!!
    }
}
data {
    int<lower=0> n_ch;                      // total number of Cepheid hosts
    int<lower=0> n_ch_d;                    // number of hosts with distances
    int<lower=0> n_ch_p;                    // number of hosts with parallaxes
    int<lower=0> n_ch_c;                    // number of hosts with only Cepheids
    int<lower=0> n_ch_s;                    // number of Cepheid hosts with SN
    int<lower=0> n_c_tot;                   // total number of Cepheids
    vector[n_ch_d + n_ch_p] est_d_anc;      // measured distance of anchor
    vector[n_ch_d + n_ch_p] sig_d_anc;      // error on above
    int<lower=1,upper=n_ch> c_ch[n_c_tot];  // mapping between Cepheid and host
    vector[n_c_tot] est_app_mag_c;          // measured apparent magnitude of each Cepheid
    vector<lower=0>[n_c_tot] sig_app_mag_c; // error on above
    vector[n_c_tot] log_p_c;                // log10 of Cepheid period in days
    vector[n_c_tot] log_z_c;                // log10 of metallicity of Cepheid
    real sig_zp_off;                        // uncertainty on zero-point offset
    vector[n_c_tot] zp_off_mask;            // which Cepheids are offset?
    vector[n_ch_p] lk_corr;                 // Lutz-Kelker corrections
    real<lower=0.0> period_break;           // optional break in CPL relation p slope
    int<lower=0, upper=1> break_at_intcpt;  // place intercept at p slope break to make CPL continuous?
}
transformed data {
    real c;                                 // c in km/s
    int n_ch_g;                             // number of hosts w/ distance or parallax
    int two_p_slopes;                       // flag to use two period slopes
    real log_period_break;                  // log_10 period at which to break CPL
    c = 2.99792458e5;
    n_ch_g = n_ch_d + n_ch_p;
    if (period_break > 0.0) {
        two_p_slopes = 1;
        log_period_break = log10(period_break);
    } else {
        two_p_slopes = 0;
    }
}
parameters {
    
    // CPL parameters
    real abs_mag_c_std;
    real slope_p;
    real slope_p_low[two_p_slopes];
    real slope_z;
    real<lower=0.01, upper=3.0> sig_int_c;
    real zp_off;
    
    // underlying Cepheid and SNe parameters, scaled to unit normal
    vector[n_c_tot] true_app_mag_c_un;
    
    // true Cepheid host distance moduli
    vector<lower=5, upper=40>[n_ch] true_mu_h;

}
transformed parameters {
    
    vector[n_ch_g] true_d_anc;      // anchor distances (Mpc) or parallaxes (mas)
    vector[n_c_tot] true_app_mag_c; // physical Cepheid app mags

    // convert anchor distance moduli to distance...
    for(i in 1: n_ch_d){
        true_d_anc[i] = mu2d(true_mu_h[i]);
        if (true_d_anc[i] == 0.0)
            reject("BAD SAMPLE: anchor ", i, " distance (", 
                   true_d_anc[i], ") = 0; true_mu_h = ", 
                   true_mu_h[i]);
        if (is_inf(true_d_anc[i]))
            reject("BAD SAMPLE: |anchor ", i, " distance| (", 
                   true_d_anc[i], ") = inf; true_mu_h = ", 
                   true_mu_h[i]);
    }

    // or parallax, as desired, accounting for LK correction
    for(i in n_ch_d + 1: n_ch_g){
        real d;
        d = mu2d(true_mu_h[i] + lk_corr[i - n_ch_d]);
        if (d == 0.0)
            reject("BAD SAMPLE: anchor ", i, " distance (", 
                   d, ") = 0; true_mu_h = ", true_mu_h[i]);
        if (is_inf(d))
            reject("BAD SAMPLE: |anchor ", i, " distance| (", 
                   d, ") = inf; true_mu_h = ", true_mu_h[i]);
        true_d_anc[i] = 1.0e-3 / d; // units are mas
    }

    // rescale true Cepheid apparent magnitudes from unit normal
    if (two_p_slopes) {
        for(i in 1: n_c_tot){
            if (log_p_c[i] >= log_period_break) {
                true_app_mag_c[i] = true_app_mag_c_un[i] * sig_int_c + 
                                    true_mu_h[c_ch[i]] + abs_mag_c_std + 
                                    slope_p * 
                                    (log_p_c[i] - log_period_break * break_at_intcpt) + 
                                    slope_z * log_z_c[i] + 
                                    zp_off_mask[i] * zp_off;
            } else {
                true_app_mag_c[i] = true_app_mag_c_un[i] * sig_int_c + 
                                    true_mu_h[c_ch[i]] + abs_mag_c_std + 
                                    slope_p_low[1] * 
                                    (log_p_c[i] - log_period_break * break_at_intcpt) + 
                                    slope_z * log_z_c[i] + 
                                    zp_off_mask[i] * zp_off;
            }
        }
    } else {
        for(i in 1: n_c_tot){
            true_app_mag_c[i] = true_app_mag_c_un[i] * sig_int_c + 
                                true_mu_h[c_ch[i]] + abs_mag_c_std + 
                                slope_p * log_p_c[i] + 
                                slope_z * log_z_c[i] + 
                                zp_off_mask[i] * zp_off;
        }
    }

}
model {

    // sample distance moduli from broad priors
    //true_mu_h ~ normal(30.0, 5.0);  // broad enough?
    
    // sample CPL parameters. see following link for old prior on sigma_int
    // http://stats.stackexchange.com/questions/156721/define-own-noninformative-prior-in-stan
    // see also https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
    abs_mag_c_std ~ normal(0.0, 20.0);
    slope_p ~ normal(-5.0, 5.0);
    if (two_p_slopes) {
        slope_p_low[1] ~ normal(-5.0, 5.0);
    }
    slope_z ~ normal(0.0, 5.0);
    sig_int_c ~ normal(0.1, 0.2);
    //target += -log(sig_int_c); // Jaynes' version of Jeffreys' prior
    zp_off ~ normal(0.0, sig_zp_off);

    // anchor distance likelihoods
    est_d_anc ~ normal(true_d_anc, sig_d_anc);
    
    // Cepheid likelihood: true app mag from CPL, measurement from true
    true_app_mag_c_un ~ normal(0.0, 1.0);
    est_app_mag_c ~ normal(true_app_mag_c, sig_app_mag_c);

}

