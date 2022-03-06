const DEFAULT_ϵ = 1e-12

# Calculate the size of the result of a convolution and corresponding iterators
# for the top left coordinates of each convolution.
function convolved_dimensions(in_width, in_height, kernel, stride, padding)
    padded_width = in_width + 2 * padding
    max_width = min(kernel * (padded_width ÷ kernel), padded_width - kernel)

    padded_height = in_height + 2 * padding
    max_height = min(kernel * (padded_height ÷ kernel), padded_height - kernel)

    iter_width = 1:stride:max_width
    iter_height = 1:stride:max_height

    out_width = length(iter_width)
    @assert out_width > 0 "Dimensions of a convolution yields a width equal to 0. " *
                          "width in = $(in_width), height in = $(in_height),  " *
                          "kernel = $(kernel), stride = $(stride), padding = $(padding)"

    out_height = length(iter_height)
    @assert out_height > 0 "Dimensions of a convolution yields a height equal to 0.  " *
                           "width in = $(in_width), height in = $(in_height),  " *
                           "kernel = $(kernel), stride = $(stride), padding = $(padding)"

    return out_width, out_height, iter_width, iter_height
end

###########################################################################################
# New Sharpened Correlation Layer

# Struct + inner constructor
@with_kw struct SharpenedConvolution
    W::AbstractArray{Float32}
    p::Float32
    q::Float32
    kernel::Integer = 5
    stride::Integer = 2
    padding::Integer = 1

    out_width::Integer = 1
    out_height::Integer = 1
    out_chan::Integer = 1
end
# Specifiy what is trainable (and everything else is a frozen parameter)
Flux.@functor SharpenedConvolution
Flux.trainable(s::SharpenedConvolution) = (s.W, s.p, s.q)

# Constructor
function SharpenedConvolution(in_width, in_height,
                              in_chan, out_chan,
                              kernel, stride, padding;
                              p=0.0, q=0.1,
                              init=Flux.glorot_uniform)

    # Allocate the result tensor. width and height reflect the number of convolutions over the input
    # accounting for padding and stride.
    out_width, out_height, _, _ = convolved_dimensions(in_width, in_height,
                                                       kernel, stride, padding)

    return SharpenedConvolution(init(kernel, kernel, in_chan, out_chan), p, q,
                                kernel, stride, padding,
                                out_width, out_height, out_chan)
end

function make_view(input_padded, v_w, v_h, k, b)
    return view(input_padded, v_w:(v_w + k - 1), v_h:(v_h + k - 1), :, b)
end

"""
    channels_convolution()

The convolution is carried out simultaneously across all the  channels of the input. How
the correlation is done channel-wise.

Dimensions W: width x height x in_chan
Returns:
"""
function channels_convolution(W, input_padded, k, p, q, v_w, v_h, out_c, b)
    in_c = size(input_padded)[end - 1]
    input_patch = view(input_padded, v_w:(v_w + k - 1), v_h:(v_h + k - 1), 1:in_c, b)

    return patch_convolution(W, input_patch, k, p, q, out_c)
end

function patch_convolution(W, in_patch, k, p, q, out_c)
    # Average input values (channel-wise)
    @tullio patch_mean[in_c] := in_patch[h, w, in_c] / (k * k)
    @cast patch_centered[h, w, in_c] := in_patch[w, h, in_c] - patch_mean[in_c]

    # Second moment adjusted for q
    @tullio x_view_norm[in_c] := (in_patch[h, w, in_c] - patch_mean[in_c])^2
    x_view_norm = Float32(√sum(x_view_norm) + DEFAULT_ϵ + q)

    # Average convolution kernel (channel-wise)
    W_cur = view(W, :, :, :, out_c)
    @tullio W_mean[c] := W_cur[w, h, c] / k^2
    @cast W_centered[h, w, in_c] := W_cur[w, h, in_c] - W_mean[in_c]

    # Second moment adjusted for q
    @tullio W_norm[c] := (W_cur[w, h, c] - W_mean[c])^2
    W_norm = Float32(√sum(W_norm) + DEFAULT_ϵ + q)

    # Calculate the sharpened cross-correlation for all channel combinations
    # No loop to modify single values. Otherwise Zygote complains
    # Note the transpose of the sum!!! Otherwise mismatch with col extraction
    # Calculate the sharpened cross-correlation
    @tullio sk := patch_centered[w, h, c] * (W_centered[w, h, c]) /
                  (x_view_norm * W_norm)

    # Exponent maps -∞:+∞ to 0:+∞
    return Float32(sign(sk) * (abs(sk))^log(1 + exp(p)))
end

# 'Forwarding' function. Everything is differentiated.
# Dimensions x: width x height x channels x batch
function (sc::SharpenedConvolution)(x::AbstractArray{Float32})
    # kernel matrix: kernel_size x kernel_size x in channels x out channel
    k, _, in_c, out_c = size(sc.W)

    # Create a view of a batch of images where each channel of each image is padded with zeros.
    # The padding is around the image, i.e. only along dim 1 and 2.
    in_w, in_h, _, batch = size(x)

    s, p = sc.stride, sc.padding
    x_pad = PaddedView(0.0, x,
                       (1:(in_w + 2 * p),     # new dimension along width
                        1:(in_h + 2 * p),     # new dimension along height
                        1:in_c,               # no change along channels
                        1:batch),             # no change along batch
                       ((1 + p):(in_w + p),   # from where to where to insert width
                        (1 + p):(in_h + p),   # from where to where to insert height
                        1:in_c,               # no change along channels
                        1:batch))             # no change along batch

    _, _, iter_w, iter_h = convolved_dimensions(in_w, in_h, k, s, p)

    # Convolution with stride and padding. It is ignored for the Zygote autodiff since
    # it is a constant.
    # W: kernel_size x kernel_size x n channels IN x n channels OUT
    result_tensor = [channels_convolution(sc.W, x_pad, k, sc.p, sc.q, v_w, v_h, c, b)
                     for v_h in iter_h, v_w in iter_w, c in 1:out_c, b in 1:batch]

    # Dimensions should be output_x, output_y, n_out, batch
    return result_tensor
end

# Block of Sharpened Crosscorr -> Batch Normalization -> MaxPool
function SC_Norm_Pool_Block(in_width, in_height, params_in, params_out, batch)
    in_chan, _, _, _, _ = params_in
    out_chan, sc_kernel, sc_stride, sc_padding, maxpool_out = params_out

    # SharpenedCrosscorr:
    #     dimensions in:  input_width,  input_height,  in channels,  batch
    #     dimensions out: output_width, output_height, out channels, batch
    #
    # Batch normalisation:
    #     dimensions in:  output_width, output_height, out channels, batch
    #     dimensions out: output_width, output_height, out channels, batch
    #
    println("""
                Creating block with output dimension of the SC layer:
                width in = $(in_width) x height in $(in_height)
                channels in $(in_chan) - out channels $(out_chan)

                """)

    return [SharpenedConvolution(in_width, in_height, in_chan, out_chan, sc_kernel,
                                 sc_stride, sc_padding),
            BatchNorm(out_chan),
            AdaptiveMaxPool((maxpool_out, maxpool_out))]
end

struct HyperParameters
    batchsize::Int
    n_classes::Int
    n_epochs::Int
    n_runs::Int
    max_lr::Float32

    # n channels out, sharp kernel, sharp stride, sharp padding, size of adaptative max pooling
    block_params::Any

    function HyperParameters(; batchsize=1_024, n_classes=10, n_epochs=100, n_runs=100,
                             max_lr=0.01,
                             block_params=Dict(1 => [3, 0, 0, 0],
                                               2 => [16, 5, 2, 1, 16],
                                               3 => [24, 5, 2, 1, 16],
                                               4 => [48, 5, 2, 1, 16]))
        return new(batchsize, n_classes, n_epochs, n_runs, max_lr, block_params)
    end
end

struct SharpenedCorrelationModel
    model::Any
    ps::HyperParameters
    in_width::Integer
    in_height::Integer

    function SharpenedCorrelationModel(ps::HyperParameters, in_width::Integer,
                                       in_height::Integer)
        return new(create_sc_model(ps, in_width, in_height), ps, in_width, in_height)
    end
end
Flux.@functor SharpenedCorrelationModel

function (sc::SharpenedCorrelationModel)(x::AbstractArray{Float32})
    return sc.model(x)
end

function create_sc_model(ps::HyperParameters, in_width::Integer, in_height::Integer)
    bp = ps.block_params

    # Number of blocks to build (exclude the input layer)
    n_blocks = length(bp)
    blocks = []

    width, height = in_width, in_height
    for i in 2:n_blocks
        new_block = SC_Norm_Pool_Block(width, height, bp[i - 1], bp[i], ps.batchsize)
        append!(blocks, new_block)
        width, height = new_block[1].out_width, new_block[1].out_height
    end

    return Chain(blocks..., Flux.flatten, Dense(bp[n_blocks][5]^2 * bp[n_blocks][1], 10))
end
