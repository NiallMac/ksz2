# ksz2

How to run the DR6 pipeline

# Generate ILC alms

- Run harmonic ILC on data k-space coadds, and also sims (including Planck). Use `/global/homes/m/maccrann/cmb/lensing/dr6_lensing_fgs/ilc/run_ilc.sh`.
- It turns out these simulations don't match the data power sufficiently well, so we need to run a few other things.
- Generate Gaussian simulations with same power as the data. Use `/global/homes/m/maccrann/cmb/ksz2/sims/generate_gaussian_sims.py`.
- Generate CMB signal-only simulations using `sim_e2e_test/get_noiseless_cmb_sims.sh`
- Run `~/cmb/ksz2/sims/replace_signal.py` to replace signal in noisy sims with CMB + Gaussian foregrounds. 
