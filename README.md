# cellblocktango
Pipeline for calculating and doing cosmology with the C_\ell's of galaxy density and CMB lensing convergence maps.

- Create maps and (apodized) masks from data.
- Calculate HALOFIT theory curves for a given cosmology and galaxy redshift distribution.
- Use MASTER algorithm to perform mask deconvolution, calculate true (binned) angular power spectra.
- Determine gaussian part of (binned) covariance matrix.
- Bin theory curves according to same binning scheme for comparison.
- Use Bayesian likelihood methods to constrain $\sigma_8$ and fit to Lagrangian Perturbation Theory model.
- Use Bayesian likelihood methods to constrain $\sigma_8$ and HOD parameters from mock galaxy catalogs.
