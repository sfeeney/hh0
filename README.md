# hh0

[![arXiv](https://img.shields.io/badge/arXiv-1707.00007-red.svg)](https://arxiv.org/abs/1707.00007)
[![arXiv](https://img.shields.io/badge/arXiv-1802.03404-orange.svg)](https://arxiv.org/abs/1802.03404)
[![arXiv](https://img.shields.io/badge/arXiv-1811.11723-yellow.svg)](https://arxiv.org/abs/1811.11723)

Code used in [arXiv:1707.00007](https://arxiv.org/abs/1707.00007) ("Clarifying the Hubble constant tension with a Bayesian hierarchical model of the local distance ladder"); [arXiv:1802.03404](https://arxiv.org/abs/1802.03404) ("Prospects for resolving the Hubble constant tension with standard sirens"); and [arXiv:1811.11723](https://arxiv.org/abs/1811.11723) ("Unbiased Hubble constant estimation from binary neutron star mergers").

Authors
 - Stephen Feeney, Daniel Mortlock and Niccolò Dalmasso ([arXiv:1707.00007](https://arxiv.org/abs/1707.00007))
 - Stephen Feeney, Hiranya Peiris, Andrew Williamson, Samaya Nissanke, Daniel Mortlock, Justin Alsing and Dan Scolnic ([arXiv:1802.03404](https://arxiv.org/abs/1802.03404))
 - Daniel Mortlock, Stephen Feeney, Hiranya Peiris, Andrew Williamson and Samaya Nissanke ([arXiv:1811.11723](https://arxiv.org/abs/1811.11723))

### Clarifying the Hubble constant tension with a Bayesian hierarchical model of the local distance ladder

A more complete README is incoming, as is a more user-friendly interface. For now, note the following dependencies

 - [PyStan](https://pystan.readthedocs.io/en/latest/)
 - [GetDist](http://getdist.readthedocs.io/en/latest/intro.html)

and that the Planck chains can be downloaded from the [Planck Legacy Archive](http://pla.esac.esa.int/pla/#cosmology) for comparison. Why not try the following?
```
export PYTHONPATH=$PYTHONPATH:<PATH TO LOCAL CLONE OF REPO>
cd stan
python stan_cpl.py
python stan_cpl_sample_plots.py
```

The following variables can be changed in `stan_cpl.py` to set up different runs. Any changes should also be made to `stan_cpl_sample_plots.py` to process the resulting outputs. Note that some recent changes (`ceph_only`, `period_break`, `max_col_c`) have not been propagated to `stan_cpl_sample_plots.py` yet: bug me if you want them!
```python
n_chains = 4                # number of independent parallel chains
n_samples = 10000           # 100000 will take 10-20 hours (linear scaling)
recompile = True            # True recompiles Stan model: set to False after first run
use_riess_rejection = False # sigma clip Cepheids before passing to BHM
ceph_only = True            # only fit Cepheid PL relation with BHM: ignore SNe
sne_sum = False             # replace full SN dataset with intercept of mag-log(z) relation
gauss_mu_like = False       # sample from anchor distance moduli, not distances
model_outliers = None       # Gauss (None) or heavy-tailed ('ht') intrinsic scatter
period_break = 10.0         # fit two period-slopes, breaking at period_break days (set to 0 to ignore)
break_at_intcpt = True      # if using two slopes, set intercept equal to break point; else 1 day
ng_maser_pdf = False        # replace Gauss MASER distance likelihood with (approx) non-Gauss form
nir_sne = False             # use near-infra-red SNe from Dhawan et al. (1707.00715)
fit_cosmo_delta = None      # fit H_0 (None) or perform model selection ('hq')
v_pla = 2015                # use Planck 2015 or 2016 inputs in model selection
constrain = True            # if using simulated data, fix random seed to test stability
stan_constrain = True       # fix random seed in sampling run
setup = 'r16'               # dataset: try 'r16' (1604.01424) or 'd17' (1707.00715)
sim = True                  # fit existing data or simulation
max_col_c = None            # set a maximum colour limit for Cepheids
```

### Prospects for resolving the Hubble constant tension with standard sirens

Note the following dependencies:

 - [emcee](http://dfm.io/emcee/current/)
 - [PyStan](https://pystan.readthedocs.io/en/latest/)
 - [corner](http://corner.readthedocs.io/en/latest/)
 - [GetDist](http://getdist.readthedocs.io/en/latest/intro.html)
 - [scikit-learn](http://scikit-learn.org/stable/install.html)

Try `python h_of_z.py` to produce inverse distance ladder constraints, and `python gw_grb_h_0_ppd.py` (or `mpirun -np X python gw_grb_h_0_ppd.py` if you have mpi4py installed and want to use it) to generate binary neutron star merger constraints.

### Unbiased Hubble constant estimation from binary neutron star mergers

Note the following dependencies:

 - [PyStan](https://pystan.readthedocs.io/en/latest/)
 - [GetDist](http://getdist.readthedocs.io/en/latest/intro.html)
 - [h5py](http://docs.h5py.org/en/latest/build.html)
 - [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html) (recommended)

The main code is contained in `bias_test_stan.py`. This generates simple simulations of catalogues of binary neutron star mergers, selected by their gravitational-wave signal to noise, and then samples either their distances and inclinations (conditioned on the measured strains only) or the cosmological parameters (Hubble constant and deceleration parameter, if selected) conditioned on both gravitational-wave and electromagnetic observations. It can be run using `mpirun -np X python bias_test_stan.py` if you have MPI4PY installed and want to use it, which you should unless you have a lot of time. By default, Stan uses 4 OpenMP threads to sample, so use 4 CPUs per MPI process and set `OMP_NUM_THREADS=4`. Additional plots for the paper were produced using `python bias_test_stan_plots.py`.

A lot of variables can be changed in `bias_test_stan.py` to set up different runs, the most useful of which are explained below. Any changes should also be made to `bias_test_stan_plots.py` to process the resulting outputs.
```python
n_event = 100                   # Catalogue size
n_rpt = 1                       # Number of catalogues to simulate
use_mpi = True                  # Use MPI? Yes!
constrain = True                # Use a standard initial seed to generate simulations?
sample = True                   # Sample or not? Turn off to save time plotting
n_samples = 1000                # Number of samples to take per chain: half will be discarded as warmup by Stan
n_chains = 4                    # Number of Stan chains
recompile = True                # Recompile Stan code: do this once then turn off to save time
fixed_n_bns = False             # Fix the catalogue size when sampling
ntlo = True                     # Adopt a quadratic Hubble law with variable q_0
vary_m_c = True                 # Draw chirp masses from a Gaussian rather than fixing them all to one value
inc_rate_redshift = True        # Include rate redshifting in inference 
plot_dist_cos_i_posts = False   # Plot gravitational-wave-only distance and inclination posteriors
sample_dist_cos_i_posts = False # Sample gravitational-wave-only distance and inclination posteriors
find_n_bar = True               # Find parameter-dependent fit to number of detectable merger events: do once then turn off
```
