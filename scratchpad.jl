function (a::AdaptiveMaxPool{S})(x::AbstractArray{T, S}) where {S, T}
  insize = size(x)[1:end-2]
  outsize = a.out
  stride = insize .÷ outsize
  k = insize .- (outsize .- 1) .* stride
  pad = 0
  pdims = PoolDims(x, k; padding=pad, stride=stride)
  return maxpool(x, pdims)
end
