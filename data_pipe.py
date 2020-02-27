import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits

nside = 2048
nbar = 608.4655

nlb, lmax = 20, 4096    
bins = nmt.NmtBin(nside, nlb = nlb, lmax = lmax)
eff_ells = bins.get_effective_ells()
nbins = bins.get_n_bands()
print('Using %s bins with %s bands per bin.' % (nbins, nlb))

# Create custom NmtBin binning scheme; logarithmically spaced \ells with equal weights
#lmin, lmax, nbins = 5, 4096, 30
#ells = np.arange(lmin, lmax + 1, 1)
#bin_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbins + 1)
#bin_edges = np.array([int(np.round(b,0)) for b in bin_edges])          # round to nearest integer
#bin_widths = np.diff(bin_edges)
#bpws, weights = np.zeros_like(ells), np.ones_like(ells)
#i = 0
#for j in range(nbins):
#    bpws[i:i+bin_widths[j]+1] = j
#    i += bin_widths[j]
#bins = nmt.NmtBin(nside, ells = ells, weights = weights, bpws = bpws, lmax = lmax)
#eff_ells = bins.get_effective_ells()
#print('Using %s bins, logarithmically spaced between %s and %s.' % (nbins, lmin, lmax))

print('Reading maps and (pre-apodized) masks...')
cmb_map   = hp.read_map(PATH + 'data/planck18/' + cmb_option + '/map.fits', verbose = False)
cmb_mask  = hp.read_map(PATH + 'data/planck18/' + cmb_option + '/mask_apo.fits', verbose = False)
desi_map  = hp.read_map(PATH + 'data/desi_lrg/map.fits', verbose = False)
desi_mask = hp.read_map(PATH + 'data/desi_lrg/mask.fits', verbose = False)
print('Done!')

cmb_map = cmb_map * cmb_mask
desi_map = desi_map * desi_mask

print('Creating fields...')
f1 = nmt.NmtField(desi_mask, [desi_map])
f2 = nmt.NmtField(cmb_mask, [cmb_map])
print('Done!')

def compute_master(f_a, f_b, bins):
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f_a, f_b, bins)
    Cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    Cl_decoupled = w.decouple_cell(Cl_coupled)
    return w, Cl_decoupled[0]

print('Calculating galaxy-lensing spectrum...')
w_kg, Cl_kg = compute_master(f1, f2, bins)
print('Calculating galaxy-galaxy spectrum...')
w_gg, Cl_gg = compute_master(f1, f1, bins)
print('Calculating lensing-lensing spectrum...')
w_kk, Cl_kk = compute_master(f2, f2, bins)
print('Done!')

print('Dividing binned gg and gk spectra by binned pixel window function...')
pixwin_fun = hp.pixwin(nside)
pixwin_lbin = np.zeros(nbins)
for i in range(nbins):
    lmaxvec = bins.get_ell_list(i)
    pixwin_i = 0
    for ell in lmaxvec:
        pixwin_i += pixwin_fun[ell]
    pixwin_lbin[i] = pixwin_i / len(lmaxvec)     
Cl_gg = Cl_gg / (pixwin_lbin ** 2)
Cl_kg = Cl_kg / pixwin_lbin
print('Done!')

print('Reading in theory curves and getting binned versions...')
l_th, Cl_kg_th, Cl_gg_th, Cl_kk_th = np.loadtxt(PATH + 'data/theory_curves_%s.txt' % dndz_option, unpack=True)
if ((np.min(l_th) != 0) | (np.max(l_th) < lmax)):
    print('Warning: Theory curves must span \ell = 0 to \ell = %s (or greater).' % lmax)
Cl_kg_th_binned = w_kg.decouple_cell(w_kg.couple_cell([Cl_kg_th[:w_kg.wsp.lmax + 1]]))[0]
Cl_gg_th_binned = w_gg.decouple_cell(w_gg.couple_cell([Cl_gg_th[:w_gg.wsp.lmax + 1]]))[0]
Cl_kk_th_binned = w_kk.decouple_cell(w_kk.couple_cell([Cl_kk_th[:w_kk.wsp.lmax + 1]]))[0]
np.savetxt(PATH + 'output/binned_theory_curves_%s.dat' % dndz_option, 
           np.column_stack((eff_ells, Cl_kg_th_binned, Cl_gg_th_binned, Cl_kk_th_binned)), 
           header='eff_ell  C_bin_kg  C_bin_gg  C_bin_kk')
print('Done!')

print('Reading in CMB lensing noise, getting binned version, and subtracting from kk spectrum...')
N_kk = np.loadtxt(PATH + 'data/planck18/' + cmb_option + '/nlkk.dat', usecols = 1)
N_kk_binned = w_kk.decouple_cell(w_kk.couple_cell([N_kk[:w_kk.wsp.lmax + 1]]))[0]
np.savetxt(PATH + 'output/binned_lensing_noise.dat', np.column_stack((eff_ells, N_kk_binned)), header='eff_ell  noise_bin')
print('Done!')

fsky_kg = np.sum(desi_mask * cmb_mask) / len(desi_mask * cmb_mask)
w2_kg = np.sum((desi_mask * cmb_mask) ** 2) / np.sum(desi_mask * cmb_mask)
w4_kg = np.sum((desi_mask * cmb_mask) ** 4) / np.sum(desi_mask * cmb_mask)
print('kg:  fsky = %.3f, w2 = %.3f, w4 = %.3f' % (fsky_kg, w2_kg, w4_kg))
fsky_gg = np.sum(desi_mask) / len(desi_mask)
w2_gg = np.sum((desi_mask) ** 2) / np.sum(desi_mask)
w4_gg = np.sum((desi_mask) ** 4) / np.sum(desi_mask)
print('gg:  fsky = %.3f, w2 = %.3f, w4 = %.3f' % (fsky_gg, w2_gg, w4_gg))
fsky_kk = np.sum(cmb_mask) / len(cmb_mask)
w2_kk = np.sum((cmb_mask) ** 2) / np.sum(cmb_mask)
w4_kk = np.sum((cmb_mask) ** 4) / np.sum(cmb_mask)
print('kk:  fsky = %.3f, w2 = %.3f, w4 = %.3f' % (fsky_kk, w2_kk, w4_kk))

print('Calculating covariance matrices...')
nbar = nbar * (180. / np.pi) ** 2                        # gal per degree --> gal per steradian
N_gg = 1. / nbar
Cl_kg_th = b * Cl_kg_th
Cl_gg_th = b ** 2 * Cl_gg_th #+ N_gg
Cl_kk_th = Cl_kk_th[:w_kk.wsp.lmax + 1] #+ N_kk
sigma_kg, sigma_gg, sigma_kk, sigma_xx = np.zeros(nbins), np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)
for i in range(nbins):  
    lmaxvec = bins.get_ell_list(i)
    dl = len(lmaxvec)
    for l in lmaxvec:
        sigma_kg[i] += (fsky_kg * (2. * l + 1.) / dl * w2_kg ** 2 / w4_kg) / (Cl_kg_th[l] ** 2 + Cl_gg_th[l] * Cl_kk_th[l])
        sigma_gg[i] += (fsky_gg * (2. * l + 1.) / dl * w2_gg ** 2 / w4_gg) / (2. * Cl_gg_th[l] ** 2)
        sigma_kk[i] += (fsky_kk * (2. * l + 1.) / dl * w2_kk ** 2 / w4_kk) / (2. * Cl_kk_th[l] ** 2) 
        sigma_xx[i] += (fsky_kg * (2. * l + 1.) / dl * w2_kg ** 2 / w4_kg) / (2. * Cl_kg_th[l] * Cl_gg_th[l]) 
    sigma_kg[i] = 1. / sigma_kg[i]   
    sigma_gg[i] = 1. / sigma_gg[i]   
    sigma_kk[i] = 1. / sigma_kk[i]   
    sigma_xx[i] = 1. / sigma_xx[i]   
sigma_kg = np.sqrt(sigma_kg)
sigma_gg = np.sqrt(sigma_gg)   
sigma_kk = np.sqrt(sigma_kk) 
sigma_xx = np.sqrt(sigma_xx) 
print('Done!')

print('Saving binned angular power spectra and covariances...')
np.savetxt(PATH + 'output/C_kg.dat', np.column_stack((eff_ells, Cl_kg, sigma_kg)), header='eff_ell  C_bin_kg  sigma_bin_kg')
np.savetxt(PATH + 'output/C_gg.dat', np.column_stack((eff_ells, Cl_gg, sigma_gg)), header='eff_ell  C_bin_gg  sigma_bin_gg')
np.savetxt(PATH + 'output/C_kk.dat', np.column_stack((eff_ells, Cl_kk, sigma_kk)), header='eff_ell  C_bin_kk  sigma_bin_kk')
np.savetxt(PATH + 'output/sigma_cross.dat', np.column_stack((eff_ells, sigma_xx)), header='eff_ell  sigma_bin_kg-gg')
print('Done!')
