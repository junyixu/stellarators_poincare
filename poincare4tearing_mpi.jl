#=
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

# %%
using Mine:B_ca2cy,cycle,get_weight
using Ptcs:Unit,MetaData,get_vertex_id_by_pos,interpolateB
using Ptcs:plot_tri
using PyCall
using LaTeXStrings
using DifferentialEquations
using MPI
using HDF5
using OffsetArrays


# %%
@pyimport matplotlib.pyplot as plt
# %%
const B0=5.18
const R0=6.2
const a=2.0
const Δϕ = 2π/18

u = Unit(B0, "alpha")
# MD = MetaData("./tearing_mode_3tables_compress.h5")
MD = MetaData("./tearing_mode_3tables_compress_199.h5")
# fid=h5open("./BX_BY_BZ_tearing.h5", "r")
fid=h5open("./BX_BY_BZ_20.h5", "r")
B = OffsetArray(reshape(read(fid["B"]), 3, :, 18), -1,-1, -1)
close(fid)

# %%

function f_B(r::Float64, ϕ::Float64, z::Float64)
	while ϕ < 0
		ϕ+=2π
	end
	ϕ %= 2π
	iϕ = floor(Int, ϕ / Δϕ)
	subB1=@view B[:, :, cycle(iϕ, 18)]
	subB2=@view B[:, :, cycle(iϕ+1, 18)]
	xy = [r-R0/u.x, z]
	# 在一个平面内以周围三角形加权
	id = get_vertex_id_by_pos(xy, MD)
	v = parent(MD.XY[:, id])
	w = get_weight(xy, v)
	B1 = interpolateB(w, subB1,id)
	B2 = interpolateB(w, subB2,id)
	# 得到两个面上的磁场，接着以 dϕ 加权
	dϕ = (ϕ - iϕ*Δϕ) / Δϕ
	w = [1-dϕ, dϕ]
	return B1*w[1] + B2*w[2]
end


function poincare!(du, u, p, ϕ)
	Bx,By,Bz=f_B(u[1], ϕ, u[2])
	Br,Bϕ,Bz = B_ca2cy(Bx,By,Bz,ϕ)
	du[1] = dr = u[1] * Br / Bϕ # dr/dϕ = r * Br / Bϕ
	du[2] = dz = u[1] * Bz / Bϕ # dz/dϕ = r * Bz / Bϕ
end
# %%

function solve_poincare(rz0::Vector, end_point::Float64)
	ϕspan = (0.001, end_point)
	prob = ODEProblem(poincare!, rz0, ϕspan)
	sol = solve(prob, TRBDF2(autodiff=false),dtmax=0.005)
	println(rz0)
	return sol
end

function plot_setting(ax)
	ax.axis("equal")
	ax.set_title("Poincare Plot Of Magnetic Field")
	ax.set_xlabel("R")
	ax.set_ylabel("Z")
end

# %%

function main()
	MPI.Init()
	comm=MPI.COMM_WORLD
	world_rank=MPI.Comm_rank(comm)
	world_size=MPI.Comm_size(comm)

	r = 0.2+0.1*world_rank
	rz0 = [(r+R0)/u.x, 0.0]

	nϕ=100
	end_point=nϕ*2π
	sol=solve_poincare(rz0, end_point);

	ϕ = 2π/3
	RZ=stack(sol.(ϕ:2π:end_point))
	R=MPI.Gather(RZ[1, :], comm)
	Z=MPI.Gather(RZ[2, :], comm)
	MPI.Finalize()

	if 0==world_rank
		fig,ax = plt.subplots()
		c=repeat(1.0:world_size, inner=nϕ)
		img=ax.scatter(R*u.x,Z*u.x, s=1,c=c)
		fig.colorbar(img)
		plot_setting(ax)
		plt.savefig("./figures/$(bytes2hex(rand(UInt8, 4))).pdf", bbox_inches="tight")
	end
end
main()
