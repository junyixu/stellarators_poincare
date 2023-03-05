#=
    仿星器 Poincare plot
    Copyright © 2023 junyi <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#


using HDF5
using Interpolations
using PyCall
using LaTeXStrings

@pyimport matplotlib.pyplot as plt

# 柱坐标到直角坐标
function C2X(r::AbstractFloat, ϕ::AbstractFloat, z::AbstractFloat)
	x=r*cos(ϕ)
	y=r*sin(ϕ)
	z=z
	return [x,y,z]
end
function c2x(xs::AbstractArray)
	Xs=similar(xs)
	for i in 1:size(xs, 2)
		Xs[:, i] = C2X(xs[:, i]...)
	end
	return Xs
end

# 读取全部数据
fid=h5open("./fieldlines_ncsx_c09r00_free.h5", "r")
for key in keys(fid)
	sb=Symbol(key)
	eval(:($sb = read(fid[$key])))
end
close(fid)

# 磁场线性插值
interp_Br = linear_interpolation((raxis, phiaxis, zaxis), B_R);
interp_Bϕ = linear_interpolation((raxis, phiaxis, zaxis), B_PHI);
interp_Bz = linear_interpolation((raxis, phiaxis, zaxis), B_Z);


function main()
	times=100000
	xs=zeros(3, times)
	dϕ=2π/6000
	for (iϕ,j) in zip([(Int(6000/12).*[10, 11])..., 1], 1:3)
		for ir in [20,30, 40, 50,  60, 70, 80]
			x0=[raxis[ir], phiaxis[16], zaxis[64]]
			phi=xs[2, iϕ]
			str_phi = string(round((phi % (2π) / π)*180))
			xs[:, 1]=x0

			for i in 1:times-1
				r = xs[1, i]
				ϕ = xs[2, i]
				ϕ = ϕ % phiaxis[end]
				z = xs[3, i]

				Br = interp_Br(r,ϕ,z)
				Bϕ = interp_Bϕ(r,ϕ,z)
				Bz = interp_Bz(r,ϕ,z)

				xs[1, i+1] = xs[1, i] + xs[1, i] * (Br/Bϕ) * dϕ
				xs[2, i+1] = xs[2, i] + dϕ
				xs[3, i+1] = xs[3, i] + xs[1, i] * (Bz/Bϕ) * dϕ
			end

			new_xs= xs[:, iϕ:2000:end]
			Xs=c2x(new_xs)
			R=@. sqrt(Xs[1, :]^2+Xs[2, :]^2)
			Z=Xs[3, :]

			plt.subplot(1,3,j)
			plt.scatter(R[1], Z[1], color="r", marker=".")
			plt.scatter(R[2:end], Z[2:end], color="b", marker=".", alpha=0.6)
			plt.title(L"\phi="*str_phi)
			plt.xlabel("R")
			plt.ylabel("Z")
			plt.axis("equal")
		end
	end
	plt.show()
end

main()
