pwd()
using Pkg;
Pkg.activate(".");

using SharpenedCorrelation, Debugger
using Flux, MLDatasets, ImageCore, PaddedViews

train_x, train_y = MNIST.traindata(Float32);
test_x, test_y = MNIST.testdata(Float32);
# Let's use global variables to easily use other datasets
WIDTH, HEIGHT, N_TRAIN_SAMPLE = size(MNIST.traintensor());
_, _, N_TEST_SAMPLE = size(MNIST.testtensor());
N_CHANNELS = 1;
# Insert a channel dimension for the MNIST dataset
train_x = reshape(train_x, WIDTH, HEIGHT, N_CHANNELS, N_TRAIN_SAMPLE);
test_x = reshape(test_x, WIDTH, HEIGHT, N_CHANNELS, N_TEST_SAMPLE);

# reencode the labels as One-Hot
train_y, test_y = Flux.onehotbatch(train_y, 0:9), Flux.onehotbatch(test_y, 0:9);


# Create DataLoaders (mini-batch iterators)
BATCH_SIZE = 256;
train_loader = Flux.DataLoader((data = train_x, label = train_y), batchsize = BATCH_SIZE, shuffle = true);
test_loader = Flux.DataLoader((data = test_x, label = test_y), batchsize = BATCH_SIZE, shuffle = false);

data, label = first(train_loader);



break_on(:error)
Debugger.@run SharpenedConvolution(28, 28, 1, 25, 5, 2, 1)

sc = SharpenedConvolution(28, 28, 1, 16, 5, 2, 1)

# kernel matrix: kernel_size x kernel_size x in channels x out channel
k, _, in_c, out_c = size(sc.W)

# Create a view of a batch of images where each channel of each image is padded with zeros.
# The padding is around the image, i.e. only along dim 1 and 2.

in_w, in_h, _, batch = size(data)

s, p = sc.stride, sc.padding
x_pad = PaddedView(0.0, data,
    (1:(in_w+2*p),     # new dimension along width
        1:(in_h+2*p),     # new dimension along height
        1:in_c,               # no change along channels
        1:batch),             # no change along batch
    ((1+p):(in_w+p),   # from where to where to insert width
        (1+p):(in_h+p),   # from where to where to insert height
        1:in_c,               # no change along channels
        1:batch));             # no change along batch

_, _, iter_w, iter_h = SharpenedCorrelation.convolved_dimensions(in_w, in_h, k, s, p)

# Convolution with stride and padding. It is ignored for the Zygote autodiff since
# it is a constant.
# W: kernel_size x kernel_size x n channels IN x n channels OUT
result_tensor = [SharpenedCorrelation.channels_convolution(sc.W, x_pad, k, sc.p, sc.q, v_w, v_h, c)
                 for v_h in iter_h, v_w in iter_w, c in 1:out_c]



# Dimensions should be output_x, output_y, n_out, batch

patch_convolution(::Array{Float32,4},
    ::SubArray{Float32,3,
        PaddedView{Float32,4,NTuple{4,UnitRange{Int64}},
            OffsetArrays.OffsetArray{Float32,4,
                Array{Float32,4}}},
        Tuple{UnitRange{Int64},UnitRange{Int64},
            Base.Slice{UnitRange{Int64}},Int64},false}, ::Int64,
    ::Float32, ::Float32, ::Int64)




####################################################################################
#
#
#

pwd()
using Pkg;
Pkg.activate(".");

using Debugger, Flux, MLDatasets, ImageCore, PaddedViews

train_x, train_y = MNIST.traindata(Float32);
test_x, test_y = MNIST.testdata(Float32);

# Let's use global variables to easily use other datasets
WIDTH, HEIGHT, N_TRAIN_SAMPLE = size(MNIST.traintensor());
_, _, N_TEST_SAMPLE = size(MNIST.testtensor());
N_CHANNELS = 1;

# Insert a channel dimension for the MNIST dataset
train_x = reshape(train_x, WIDTH, HEIGHT, N_CHANNELS, N_TRAIN_SAMPLE);
test_x = reshape(test_x, WIDTH, HEIGHT, N_CHANNELS, N_TEST_SAMPLE);

# reencode the labels as One-Hot
train_y, test_y = Flux.onehotbatch(train_y, 0:9), Flux.onehotbatch(test_y, 0:9);

# Create DataLoaders (mini-batch iterators)
BATCH_SIZE = 256;
train_loader = Flux.DataLoader((data = train_x, label = train_y), batchsize = BATCH_SIZE, shuffle = true);
test_loader = Flux.DataLoader((data = test_x, label = test_y), batchsize = BATCH_SIZE, shuffle = false);

data, label = first(train_loader, BATCH_SIZE);

# Average input values (channel-wise)
using Tullio

p = 1.0f0;
q = 0.1f0;
in_w, in_h, in_c, in_b = size(data[1])

stride, pad = 3, 2
input_padded = PaddedView(0.0f0, data[1],
    (1:(in_w+2*pad),     # new dimension along width
        1:(in_h+2*pad),     # new dimension along height
        1:in_c,               # no change along channels
        1:in_b),             # no change along batch
    ((1+pad):(in_w+pad),   # from where to where to insert width
        (1+pad):(in_h+pad),   # from where to where to insert height
        1:in_c,               # no change along channels
        1:in_b));             # no change along batch


v_w, v_h, k = 5, 5, 5
patch = input_padded[v_w:(v_w+k-1), v_h:(v_h+k-1), 1:end, 1:BATCH_SIZE];
patch = input_padded[v_w:(v_w+k-1), v_h:(v_h+k-1), 1:end, 1:BATCH_SIZE];

@tullio sk[b] := begin
    patch_mean[in_c, b] = patch[w, h, in_c, b] / ($k * $k)

    # Second moment adjusted for DEFAULT_ϵ and q
    patch_centered[w, h, in_c, b] = patch[w, h, in_c, b] - patch_mean[in_c, b]
    patch_norm[in_c, b] = sqrt <| (patch[w, h, in_c, b] - patch_mean[in_c, b])^2

    # $ prefix is for constants with no gradient calculation
    patch_norm[in_c, b] = Float32 <| (patch_norm[in_c, b] + $DEFAULT_ϵ + q)

    # Average convolution kernel (channel-wise)
    W_cur[w, h, in_c] = W[w, h, in_c, $out_c]
    W_mean[in_c] = W_cur[w, h, in_c] / ($k * $k)
    W_centered[w, h, in_c] = W_cur[w, h, in_c] - W_mean[in_c]

    # Second moment adjusted for DEFAULT_ϵ + q
    W_norm[in_c] = sqrt <| (W_cur[w, h, in_c] - W_mean[in_c])^2
    W_norm[in_c] = Float32 <| (W_norm[in_c] + $DEFAULT_ϵ + q)

    # Calculate the sharpened cross-correlation for all channel combinations
    # No loop to modify single values. Otherwise Zygote complains
    # Note the transpose of the sum!!! Otherwise mismatch with col extraction
    # Calculate the sharpened cross-correlation

    (patch_centered[w, h, in_c, b] * W_centered[w, h, in_c]) /
    (patch_norm[in_c, b] * W_norm[in_c])
end

# Exponent maps -∞:+∞ to 0:+∞
# return Float32(sign(sk) * (abs(sk))^log(1 + exp(p)))
return Float32.(sign.(sk) * (abs.(sk)) .^ p)
