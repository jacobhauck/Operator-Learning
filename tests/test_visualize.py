import data.external.pdebench as pdb

d = pdb.PDEBenchStandardDataset('comp-navier-stokes', 3, M=1.0, eta=1e-8, zeta=1e-8, boundary_conditions='periodic', initial_condition='random')
state = d[2]

state.trace(0, 0).trace(64, 0).quick_visualize()
state.trace(-1, 0).trace(64, 0).quick_visualize()
