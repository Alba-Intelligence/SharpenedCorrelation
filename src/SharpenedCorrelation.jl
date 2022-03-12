module SharpenedCorrelation

using Parameters: @with_kw
using Statistics: mean
using Tullio, PaddedViews

using CUDA

using Flux
using Flux: params
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader

using BenchmarkTools

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

if CUDA.has_cuda() && CUDA.has_cuda_gpu()
    try
        using CuArrays, NNlibCUDA
        Training_Device = gpu
    catch ex
        @warn "CUDA installed, but CuArrays.jl fails to load. Defaulting to cpu " exception = (ex,
            catch_backtrace())
        Training_Device = cpu
    end
else
    Training_Device = cpu
end

include("model.jl")

export HyperParameters,
    SharpenedConvolution,
    SharpenedCorrelationModel
end
