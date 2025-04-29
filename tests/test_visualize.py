import data.external.pdebench as pdb

d = pdb.PDEBenchStandardDataset('burgers', 1, nu=0.001)
f = d[2]
f.quick_visualize(x_min=f.x_min, x_max=2*f.x_max)
f.quick_visualize(x_min=f.x_min, x_max=f.x_max)
