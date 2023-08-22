using Spasm, JLD, SparseArrays

m3 = JLD.load("../../7/boundary_C_7_3.jld","boundary_C_7_3")
m4 = JLD.load("../../7/boundary_C_7_4.jld","boundary_C_7_4")

m3s = Spasm.spasm(m3)
m4s = Spasm.spasm(m4)

k3s = Spasm.kernel(transpose(m3s))
k4s = Spasm.kernel(transpose(m4s))
