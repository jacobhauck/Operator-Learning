import data.external.pdebench as pdb

d = pdb.PDEBenchStandardDataset('incomp-navier-stokes', 2)
fluid, force = d[2]

force.quick_visualize()
