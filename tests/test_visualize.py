import data.external.pdebench as pdb

d = pdb.PDEBenchStandardDataset('comp-navier-stokes', 2, M=0.1, eta=0.01, zeta=0.01, boundary_conditions='periodic', initial_condition='random')
state = d[2]

state.trace(0, 0).quick_visualize()
state.trace(-1, 0).quick_visualize()
