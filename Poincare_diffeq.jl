#=
    仿星器 Poincare plot
    Copyright © 2023 junyi <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#


using HDF5
using Interpolations
using DifferentialEquations
using PyCall
using LaTeXStrings

@pyimport matplotlib.pyplot as plt

# 读取全部数据
fid=h5open("./fieldlines_ncsx_c09r00_free.h5", "r")
for key in keys(fid)
	sb=Symbol(key)
	eval(:($sb = read(fid[$key])))
end
close(fid)

# 磁场线性插值
interp_Br = linear_interpolation((raxis, phiaxis, zaxis), B_R)
interp_Bϕ = linear_interpolation((raxis, phiaxis, zaxis), B_PHI)
interp_Bz = linear_interpolation((raxis, phiaxis, zaxis), B_Z)

function poincare!(du, u, p, ϕ)
	ϕ_tmp = ϕ % (2π/3)
	Br = interp_Br(u[1],ϕ_tmp,u[2])
	Bϕ = interp_Bϕ(u[1],ϕ_tmp,u[2])
	Bz = interp_Bz(u[1],ϕ_tmp,u[2])
	du[1] = dr = u[1] * Br / Bϕ # dr/dϕ = r * Br / Bϕ
	du[2] = dz = u[1] * Bz / Bϕ # dz/dϕ = r * Bz / Bϕ
end


@time begin
# phiaxis[16] 是 π/3
# for ir in [27, 30, 35, 40, 45, 50, 60, 65, 67, 70, 72, 75, 80]
# for ir in [45, 50, 65, 67, 70, 72, 75, 76, 78, 80]
for (i,ir) in enumerate([78])
# for (i,ir) in enumerate([ 71, 72, 73, 74, 75,76, 77, 78])
	u0 = [raxis[ir], zaxis[66]]
	ϕspan = (phiaxis[16], 142*2π)
	prob = ODEProblem(poincare!, u0, ϕspan)
	dt=0.001
	sol = solve(prob, TRBDF2(), dtmax=dt)
	mask=@. (sol.t-π/3+dt) % (2π/3) <= dt
	X=reduce(hcat, sol.u[mask]) # R: X[1], Z: X[2]
	plt.scatter(X[1, :], X[2, :], marker=".", s=0.1, label=string(i))
end
plt.legend()
plt.title(L"$\theta = 60^\circ$")
plt.show()
end
# 每次前进 0.01*180/π == 0.05 度
