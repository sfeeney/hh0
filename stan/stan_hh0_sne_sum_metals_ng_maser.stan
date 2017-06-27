functions{
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
    int<lower=0> n_comp_m;                  // number of normal components in model of MASER PDF
    vector[n_comp_m] amp_d_m;               // amplitude of each above component
    vector[n_comp_m] mu_d_m;                // mean of each above component
    vector[n_comp_m] sig_d_m;               // std dev of each above component
    vector[n_ch_d + n_ch_p - 1] est_d_anc;  // measured distance of anchors
    vector[n_ch_d + n_ch_p - 1] sig_d_anc;  // error on above
    int<lower=1,upper=n_ch> c_ch[n_c_tot];  // mapping between Cepheid and host
    vector[n_c_tot] est_app_mag_c;          // measured apparent magnitude of each Cepheid
    vector<lower=0>[n_c_tot] sig_app_mag_c; // error on above
    vector[n_c_tot] log_p_c;                // log10 of Cepheid period in days
    vector[n_c_tot] log_z_c;                // log10 of metallicity of Cepheid
    vector[n_ch_s] est_app_mag_s;           // measured app mag of each SN
    vector<lower=0>[n_ch_s] sig_app_mag_s;  // error on above
    real est_a_x;                           // measured intercept of SN m-z rel'n
    real sig_a_x;                           // error on above
    real sig_zp_off;                        // uncertainty on zero-point offset
    vector[n_ch] zp_off_mask;               // which hosts are offset?
    vector[n_ch_p] lk_corr;                 // Lutz-Kelker corrections
}
transformed data {
    real c;                                 // c in km/s
    int n_ch_g;                             // number of hosts w/ distance or parallax
    c = 2.99792458e5;
    n_ch_g = n_ch_d + n_ch_p;
}
parameters {
    
    // CPL parameters
    real abs_mag_c_std;
    real slope_p;
    real slope_z;
    //real<lower=0> sig_int_c;
    real<lower=0.01, upper=3.0> sig_int_c;
    real zp_off;
    
    // SNe parameters
    real abs_mag_s_std;
    //real<lower=0> sig_int_s;
    real<lower=0.01, upper=3.0> sig_int_s;
    
    // cosmology!
    real a_x;
    
    // underlying Cepheid and SNe parameters, scaled to unit normal
    vector[n_c_tot] true_app_mag_c_un;
    vector[n_ch_s] true_app_mag_s_un;
    
    // true Cepheid host distance moduli
    vector<lower=5, upper=40>[n_ch] true_mu_h;

}
transformed parameters {
    
    vector[n_ch_g] true_d_anc;      // anchor distances (Mpc) or parallaxes (mas)
    vector[n_c_tot] true_app_mag_c; // physical Cepheid app mags
    vector[n_ch_s] true_app_mag_s;  // physical SN magnitudes
    real log_h_0;
    real h_0;

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
        true_d_anc[i] = 1.0e-3 / d; # units are mas
    }

    // rescale true Cepheid apparent magnitudes from unit normal
    for(i in 1: n_c_tot){
        true_app_mag_c[i] = true_app_mag_c_un[i] * sig_int_c + 
                            true_mu_h[c_ch[i]] + abs_mag_c_std + 
                            slope_p * log_p_c[i] + 
                            slope_z * log_z_c[i] + 
                            zp_off_mask[c_ch[i]] * zp_off;
    }

    // Cepheid/SN host magnitudes rescaled from latent distance 
    // modulus parameter
    true_app_mag_s[1: n_ch_s] = true_app_mag_s_un[1: n_ch_s] * 
                                sig_int_s + 
                                true_mu_h[n_ch_g + n_ch_c + 1: n_ch] + 
                                abs_mag_s_std;

    // set Hubble Constant (both log and linear)
    log_h_0 = (abs_mag_s_std + 5.0 * a_x + 25.0) / 5.0;
    h_0 = 10.0 ^ log_h_0;
    
}
model {

    // placeholder variable
    real to_sum[n_comp_m];

    // sample distance moduli from broad priors
    //true_mu_h ~ normal(30.0, 5.0);  // broad enough?
    
    // sample CPL parameters. see following link for old prior on sigma_int
    // http://stats.stackexchange.com/questions/156721/define-own-noninformative-prior-in-stan
    // see also https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
    abs_mag_c_std ~ normal(0.0, 20.0);
    slope_p ~ normal(-5.0, 5.0);
    slope_z ~ normal(0.0, 5.0);
    sig_int_c ~ normal(0.1, 0.2);
    //target += -log(sig_int_c); // Jaynes' version of Jeffreys' prior
    zp_off ~ normal(0.0, sig_zp_off);
    
    // sample supernova parameters
    abs_mag_s_std ~ normal(-20.0, 10.0);
    //sig_int_s ~ normal(0.1, 0.2);
    sig_int_s ~ normal(0.1, 0.01);
    //target += -log(sig_int_s); // Jaynes' version of Jeffreys' prior

    // below needed if prior on sig_int_s allows zero
    if (is_inf(sig_int_s))
        reject("BAD SAMPLE! sig_int_s = ", sig_int_s);

    // sample SN m-z intercept
    a_x ~ normal(0.7, 0.1);
    est_a_x ~ normal(a_x, sig_a_x);

    // anchor distance likelihoods
    for(i in 1:n_comp_m) {
        to_sum[i] = log(amp_d_m[i]) + 
                    normal_lpdf(true_d_anc[1] | mu_d_m[i], sig_d_m[i]);
    }
    target += log_sum_exp(to_sum);
    est_d_anc[1: n_ch_g - 1] ~ normal(true_d_anc[2: n_ch_g], 
                                      sig_d_anc[1: n_ch_g - 1]);
    
    // Cepheid likelihood: true app mag from CPL, measurement from true
    true_app_mag_c_un ~ normal(0.0, 1.0);
    est_app_mag_c ~ normal(true_app_mag_c, sig_app_mag_c);

    // SNe likelihoods: true SNe apparent magnitudes are sourced from
    // the true distance moduli (latent parameters for Cepheid hosts
    // and transformed from true redshift for high-z SNe) and the 
    // SNe absolute magnitude. measurements sampled from true
    true_app_mag_s_un ~ normal(0.0, 1.0);
    est_app_mag_s ~ normal(true_app_mag_s[1: n_ch_s], sig_app_mag_s);

}

