# cellblocktango
Pipeline for calculating, analyzing, and doing cosmology with the C_ell's of galaxy density and CMB lensing convergence maps.

- Create maps and (apodized) masks from data.
- Calculate theory curves for a given cosmology and galaxy redshift distribution.
- Use MASTER algorithm to perform mask deconvolution, calculate true (binned) angular power spectra.
- Determine gaussian part of (binned) covariance matrix.
- Bin theory curves according to same binning scheme for comparison.
- Use least likelihood methods to infer best fit HOD parameters from mock galaxy catalogs.
