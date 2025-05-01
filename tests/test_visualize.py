import data.external.pdebench as pdb

d = pdb.PDEBenchStandardDataset('shallow-water', 2)
f = d[2]

ic = f.trace(0, 0).quick_visualize()
fc = f.trace(-1, 0).quick_visualize()
