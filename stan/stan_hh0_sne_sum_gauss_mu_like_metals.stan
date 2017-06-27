data {
    int<lower=0> n_ch;                      // total number of Cepheid hosts
    int<lower=0> n_ch_d;                    // number of hosts with distances
    int<lower=0> n_ch_p;                    // number of hosts with parallaxes
    int<lower=0> n_ch_c;                    // number of hosts with only Cepheids
    int<lower=0> n_ch_s;                    // number of Cepheid hosts with SN
    int<lower=0> n_c_tot;                   // total number of Cepheids
    vector[n_ch_d + n_ch_p] est_mu_anc;     // measured distance modulus of anchor
    vector[n_ch_d + n_ch_p] sig_mu_anc;     // error on above
    int<lower=1,upper=n_ch> c_ch[n_c_tot];  // mapping between Cepheid and host
    vector[n_c_tot] est_app_mag_c;          // measured apparent magnitude of each Cepheid
    vector<lower=0>[n_c_tot] sig_app_mag_c; // error on above
    real<lower=0> sig_int_c;                // intrinsic scatter in Cepheid PL rel'n
    vector[n_c_tot] log_p_c;                // log10 of Cepheid period in days
    vector[n_c_tot] log_z_c;                // log10 of metallicity of Cepheid
    vector[n_ch_s] est_app_mag_s;           // measured app mag of each SN
    vector<lower=0>[n_ch_s] sig_app_mag_s;  // error on above
    real<lower=0> sig_int_s;                // intrinsic scatter in SNe
    real est_a_x;                           // measured intercept of SN m-z rel'n
    real sig_a_x;                           // error on above
    real sig_zp_off;                        // uncertainty on zero-point offset
    vector[n_ch] zp_off_mask;               // which hosts are offset?
}
transformed data {
    real c;                                 // c in km/s
    int n_ch_g;                             // number of hosts w/ distance or parallax
    vector[n_c_tot] sig_tot_app_mag_c;      // total uncertainty on Cepheid mags
    vector[n_ch_s] sig_tot_app_mag_s;       // total uncertainty on SN mags
    c = 2.99792458e5;
    n_ch_g = n_ch_d + n_ch_p;
    for(i in 1: n_c_tot){
        sig_tot_app_mag_c[i] = sqrt(sig_app_mag_c[i] * 
                                    sig_app_mag_c[i] + 
                                    sig_int_c * sig_int_c);
    }
    for(i in 1: n_ch_s){
        sig_tot_app_mag_s[i] = sqrt(sig_app_mag_s[i] * 
                                    sig_app_mag_s[i] + 
                                    sig_int_s * sig_int_s);
    }
}
parameters {
    
    // CPL parameters
    real abs_mag_c_std;
    real slope_p;
    real slope_z;
    real zp_off;
    
    // SNe parameter
    real abs_mag_s_std;
    
    // cosmology!
    real a_x;
    
    // true Cepheid host distance moduli
    vector<lower=5, upper=40>[n_ch] true_mu_h;

}
transformed parameters {
    
    vector[n_c_tot] true_app_mag_c; // physical Cepheid app mags
    vector[n_ch_s] true_app_mag_s;    // physical SN magnitudes
    real log_h_0;
    real h_0;

    // Cepheid apparent magnitudes
    for(i in 1: n_c_tot){
        true_app_mag_c[i] = true_mu_h[c_ch[i]] + abs_mag_c_std + 
                            slope_p * log_p_c[i] + 
                            slope_z * log_z_c[i] + 
                            zp_off_mask[c_ch[i]] * zp_off;
    }

    // Cepheid/SN host magnitudes
    true_app_mag_s[1: n_ch_s] = true_mu_h[n_ch_g + n_ch_c + 1: n_ch] + 
                                abs_mag_s_std;

    // set Hubble Constant (both log and linear)
    log_h_0 = (abs_mag_s_std + 5.0 * a_x + 25.0) / 5.0;
    h_0 = 10.0 ^ log_h_0;
    
}
model {

    // sample distance moduli from broad priors
    //true_mu_h ~ normal(30.0, 5.0);  // broad enough?
    
    // sample CPL parameters
    abs_mag_c_std ~ normal(0.0, 20.0);
    slope_p ~ normal(-5.0, 5.0);
    slope_z ~ normal(0.0, 5.0);
    zp_off ~ normal(0.0, sig_zp_off);
    
    // sample supernova parameter
    abs_mag_s_std ~ normal(-20.0, 10.0);

    // sample SN m-z intercept
    a_x ~ normal(0.7, 0.1);
    est_a_x ~ normal(a_x, sig_a_x);

    // anchor distance likelihoods
    est_mu_anc ~ normal(true_mu_h[1: n_ch_g], sig_mu_anc);
    
    // Cepheid likelihood: measurement from true
    est_app_mag_c ~ normal(true_app_mag_c, sig_tot_app_mag_c);

    // SNe likelihoods: true SNe apparent magnitudes are sourced from
    // the true distance moduli and the SNe absolute magnitude. 
    // measurements sampled from true
    est_app_mag_s ~ normal(true_app_mag_s, sig_tot_app_mag_s);

}

