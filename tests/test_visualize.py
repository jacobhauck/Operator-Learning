import data.external.pdebench as pdb

d = pdb.PDEBenchStandardDataset('reaction-diffusion', 1, nu=0.5, rho=1.0)
f = d[2]
f.quick_visualize()
