pwd()
using Pkg;
Pkg.activate(".");

using SharpenedCorrelation, Debugger

break_on(:error)
Debugger.@run SharpenedConvolution(28, 28, 1, 25, 5, 2, 1)

SharpenedConvolution(28, 28, 1, 25, 5, 2, 1)

patch_convolution(::Array{Float32,4},
                  ::SubArray{Float32,3,
                             PaddedView{Float32,4,NTuple{4,UnitRange{Int64}},
                                        OffsetArrays.OffsetArray{Float32,4,
                                                                 Array{Float32,4}}},
                             Tuple{UnitRange{Int64},UnitRange{Int64},
                                   Base.Slice{UnitRange{Int64}},Int64},false}, ::Int64,
                  ::Float32, ::Float32, ::Int64)
