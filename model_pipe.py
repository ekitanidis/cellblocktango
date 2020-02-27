import os
os.environ["OMP_NUM_THREADS"] = "1"  # see: https://emcee.readthedocs.io/en/stable/tutorials/parallel/
import numpy as np
from numpy.linalg import inv
from scipy import stats, integrate, interpolate, special
from mcfit import xi2P
from classy import Class
from multiprocessing import Pool, cpu_count
import emcee
import time
import cleft_fftw

start_overhead = time.time()

# Note: The seemingly bad practice of defining global variables instead of passing arguments into emcee is intentional here. 
# It's ugly but fast because it avoids pickling and unpickling the data every time the model is called. 
# See https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments for more info.

class LinearPowerSpectrum:
    
    def __init__(self, k, p, sigma8):
        self.k = k
        self.p = p
        self.sigma8 = sigma8
        
    def rescale(self, sigma8_new):
        self.p = self.p * (sigma8_new / self.sigma8) ** 2

        
def fetch_planck_cosmology(z_max):
    
    cosmo = Class()
    cosmo.set({'H0' : 67.36, 'T_cmb' : 2.7255, 'A_s' : 2.14e-9, 'n_s' : 0.9649,
               'reio_parametrization' : 'reio_camb', 'tau_reio' : 0.0544, 'reionization_exponent' : 1.5,
               'reionization_width' : 0.5, 'helium_fullreio_redshift' : 3.5, 'helium_fullreio_width' : 0.5,
               'omega_b' : 0.02237, 'omega_cdm' : 0.1200, 'Omega_k' : 0., 'Omega_Lambda' : 0.6847,              
               'output' : 'mPk', 'P_k_max_1/Mpc' : 100, 'z_max_pk' : z_max, 'non linear' : 'halofit'})
    cosmo.compute()
    return cosmo


def precompute(CosmoClass, ell_min = 0, ell_max = 1200, z_min = 0.1, z_max = 1.3):

    def get_comoving_info(z_min, z_max, CosmoClass, z_width = 0.05):        
        # returns chi, dz/dchi in Mpc, 1/Mpc
        z_arr = np.arange(z_min, z_max, z_width)        
        chi_arr = CosmoClass.z_of_r(z_arr)[0]
        dzdchi_arr = CosmoClass.z_of_r(z_arr)[1]
        return z_arr, chi_arr, dzdchi_arr
    
    def limber_grid(ell_arr, chi_arr):
        # assumes chi_arr in units of Mpc, returns k in units of 1/Mpc
        def k_from_ell(ell, chi_arr):
            k_arr = 1. * (ell + 0.5) / chi_arr 
            return k_arr
        k_matrix = np.zeros((len(ell_arr), len(chi_arr)))
        for i,l in enumerate(ell_arr):
            k_matrix[i,:] = k_from_ell(l, chi_arr)       
        return k_matrix
    
    def get_integrands(CosmoClass, tab, z2chi):
        z_tab, dndz_tab = tab
        z_arr, chi_arr, dzdchi_arr = z2chi
        dndz_arr = interpolate.interp1d(z_tab, dndz_tab, bounds_error = False, fill_value = 0)(z_arr)
        dndz_arr = dndz_arr / integrate.trapz(dndz_arr, z_arr)
        c = 299792                                                             # speed of light in km/s
        H0 = 100 * CosmoClass.h()                                              # H0, H(z) in km/s/Mpc
        H_arr = 100 * np.sqrt(CosmoClass.Omega_Lambda() +                    
                             CosmoClass.Omega_m() * (1 + z_arr) ** 3 + 
                             CosmoClass.Omega_g() * (1 + z_arr) ** 4 )    
        chi_ls = CosmoClass.z_of_r([1089.92])[0]                               # lss co-moving distance in Mpc
        w_gal = dndz_arr
        w_cmb = 3. / (2. * c) * CosmoClass.Omega_m() * H0 ** 2 / H_arr * (1 + z_arr) * chi_arr * (chi_ls - chi_arr) / chi_ls 
        zintegrand = 1. / chi_arr ** 2 * H_arr / c
        zintegrand_gg = zintegrand * w_gal ** 2
        zintegrand_kg = zintegrand * w_gal * w_cmb
        return zintegrand_gg, zintegrand_kg

    ell_arr = np.arange(ell_min, ell_max)
    z_arr, chi_arr, dzdchi_arr = get_comoving_info(z_min, z_max, CosmoClass)    # chi, dz/dchi in units Mpc, 1/Mpc
    k_matrix = limber_grid(ell_arr, chi_arr)                                    # k in units 1/Mpc
    
    z_tab, dndz_tab = np.loadtxt('/global/homes/e/elliek/photodesi/data/dndz/dr8/dndz_lrg.txt', unpack=True)
    integrand_gg, integrand_kg = get_integrands(CosmoClass, (z_tab, dndz_tab), (z_arr, chi_arr, dzdchi_arr))
    
    return k_matrix, ell_arr, z_arr, integrand_gg, integrand_kg

    
def cl_from_pk(k_cleft, p_cleft, k_matrix, ell_arr, z_arr, integrand_arr):
    p_func = interpolate.interp1d(k_cleft, p_cleft, bounds_error = True)
    C_arr = np.zeros(len(ell_arr))
    for i in range(len(ell_arr)):
        C_arr[i] = np.trapz(p_func(k_matrix[i,:]) * integrand_arr, z_arr)
    C_func = interpolate.interp1d(ell_arr, C_arr, bounds_error = True)
    return C_func

Planck18 = fetch_planck_cosmology(z_max = 1.3)

k_lin = np.logspace(-5, 1, 100)                                         # k in 1/Mpc
Pk_lin = np.array([Planck18.pk_lin(k,z_eff) for k in k_lin])            # P(k) in Mpc^3
P0 = LinearPowerSpectrum(k_lin, Pk_lin, Planck18.sigma8())
cl = cleft_fftw.CLEFT(k = P0.k, p = P0.p, one_loop = True, shear = True, threads = 64)
cl.make_ptable(kmin = 1e-5, kmax = 10, nk = 100)
cl.export_wisdom()

# k in 1/Mpc, integrands in 1/Mpc^3
k_matrix, ell_arr, z_arr, integrand_gg, integrand_kg = precompute(Planck18, ell_min = 0, ell_max = 1200, z_min = 0.1, z_max = 1.3)

l_gg, Cgg, sigma_gg = np.loadtxt('/global/homes/e/elliek/kitanidis++20b/output/C_gg.dat', unpack=True)
l_kg, Ckg, sigma_kg = np.loadtxt('/global/homes/e/elliek/kitanidis++20b/output/C_kg.dat', unpack=True)
l_xx, sigma_xx = np.loadtxt('/global/homes/e/elliek/kitanidis++20b/output/sigma_cross.dat', unpack=True)

cov_gg = np.diag(sigma_gg**2)
cov_kg = np.diag(sigma_kg**2)
cov_xx = np.diag(sigma_xx**2)
cov = np.block([[cov_gg, cov_xx],[cov_xx, cov_kg]])
invcov = inv(cov)

def lnlike(theta):
    s8, ax, aa, b1, b2, bs2, sa = theta
    P0.rescale(s8)    
    cleft = cleft_fftw.CLEFT(k = P0.k, p = P0.p, one_loop = True, shear = True, threads = 1, import_wisdom=True)
    cleft.make_ptable(kmin = 0.95 * k_matrix.min(), kmax = 1.05 * k_matrix.max(), nk = 100)
    bvec = [b1, b2, bs2, ax, aa]
    kgg, pgg = cleft.combine_bias_terms_pgg(bvec)                               # k, P(k) in 1/Mpc, Mpc^3
    kmg, pmg = cleft.combine_bias_terms_pmg(bvec)                               # k, P(k) in 1/Mpc, Mpc^3
    Cgg_pred = cl_from_pk(kgg, pgg, k_matrix, ell_arr, z_arr, integrand_gg)
    Ckg_pred = cl_from_pk(kmg, pmg, k_matrix, ell_arr, z_arr, integrand_kg)
    C_diff = np.concatenate([(Cgg_pred(l_gg) - Cgg + sa) ** 2, (Ckg_pred(l_kg) - Ckg) ** 2])
    lnlike = -0.5 * np.dot(C_diff, np.dot(invcov, C_diff)
    return lnlike

def lnflatprior(theta):
    s8, ax, aa, b1, b2, bs2, sa = theta
    sensible  = (s8 >= 0.5) & (s8 <= 1.) 
    sensible &= (b1 >= 0.) & (b1 <= 1.) & (b2 >= -1.) & (b2 <= 0.) & (bs2 >= -1.) & (bs2 <= 1.)
    sensible &= (ax >= -100.) & (ax <= 100.) & (aa >= -100.) & (aa <= 100.)
    sensible &= (sa >= 0.75 * Ngg) & (sa <= 1.25 * Ngg)
    if sensible == False:
        return -np.inf
    else:
        return 0.
    
def lnprob(theta):                      
    lp = lnflatprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

guess = [0.83, -10., -10., 0.6, -0.7, 0.5, Ngg]
ndim, nwalkers = len(guess), cpu_count()
nsteps = 10000
pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

filename = "samples.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

end_overhead = time.time()
print("Overhead took {0:.1f} seconds".format(end_overhead - start_overhead))

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    start = time.time()
    sampler.run_mcmc(pos, nsteps, progress=True, store=True)
    end = time.time()
    print("Multiprocessing took {0:.1f} seconds".format(end - start))
