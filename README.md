# hh0
Hierarchical Hubble Constant Inference

Authors
 - Stephen Feeney
 - Daniel Mortlock
 - Niccolò Dalmasso

A more complete README is incoming, as is a more user-friendly interface. For now, note the following dependencies

 - [PyStan](https://pystan.readthedocs.io/en/latest/)
 - [GetDist](http://getdist.readthedocs.io/en/latest/intro.html)

and that the Planck chains can be downloaded from the [Planck Legacy Archive](http://pla.esac.esa.int/pla/#cosmology) for comparison. Why not try the following?
```
python stan_cpl.py
python stan_cpl_sample_plots.py
```

The following variables can be changed in `stan_cpl.py` to set up different runs. Any changes should also be made to `stan_cpl_sample_plots.py` to process the resulting outputs.
```python
n_chains = 4                # number of independent parallel chains
n_samples = 10000           # 100000 will take 10-20 hours (linear scaling)
recompile = True            # True recompiles Stan model: set to False after first run
use_riess_rejection = False # sigma clip Cepheids before passing to BHM
sne_sum = False             # replace full SN dataset with intercept of mag-log(z) relation
gauss_mu_like = False       # sample from anchor distance moduli, not distances
model_outliers = None       # Gauss (None) or heavy-tailed ('ht') intrinsic scatter
ng_maser_pdf = False        # replace Gauss MASER distance likelihood with (approx) non-Gauss form
nir_sne = False             # use near infra-red SNe from Dhawan et al. (1707.00715)
fit_cosmo_delta = None      # fit H_0 (None) or perform model selection ('hq')
v_pla = 2015                # use Planck 2015 or 2016 inputs in model selection
constrain = True            # if using simulated data, fix random seed to test stability
stan_constrain = True       # fix random seed in sampling run
setup = 'r16'               # dataset: try 'r16' (1604.01424) or 'd17' (1707.00715)
sim = True                  # fit existing data or simulation
```
