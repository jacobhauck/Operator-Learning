import data.external.cfdbench as cdb

d = cdb.CFDBenchDataset('cylinder-prop')
state, metadata = d[5]

print(metadata)

for t in range(0, len(state.xs[state.ii('t')]), 100):
    state.trace(t, 't').quick_visualize()
