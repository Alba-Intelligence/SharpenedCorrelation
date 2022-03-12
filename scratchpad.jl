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
train_loader = Flux.DataLoader((data = train_x, label = train_y); batchsize = BATCH_SIZE,
    shuffle = true);
test_loader = Flux.DataLoader((data = test_x, label = test_y); batchsize = BATCH_SIZE,
    shuffle = false);

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
result_tensor = [SharpenedCorrelation.channels_convolution(sc.W, x_pad, k, sc.p, sc.q, v_w,
    v_h, c)
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

####################################################################################################
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
train_loader = Flux.DataLoader((data = train_x, label = train_y); batchsize = BATCH_SIZE,
    shuffle = true);
test_loader = Flux.DataLoader((data = test_x, label = test_y); batchsize = BATCH_SIZE,
    shuffle = false);

data, label = first(train_loader);

# Average input values (channel-wise)
using Tullio

p = 1.0f0;
q = 0.1f0;
in_w, in_h, in_c, in_b = size(data)

stride, padding = 3, 2
input_padded = PaddedView(0.0f0, data,
    (1:(in_w+2*padding),     # new dimension along width
        1:(in_h+2*padding),     # new dimension along height
        1:in_c,               # no change along channels
        1:in_b),             # no change along batch
    ((1+padding):(in_w+padding),   # from where to where to insert width
        (1+padding):(in_h+padding),   # from where to where to insert height
        1:in_c,               # no change along channels
        1:in_b));             # no change along batch

v_w, v_h, k = 5, 5, 5

patch = input_padded[v_w:(v_w+k-1), v_h:(v_h+k-1), 1:end, 1:BATCH_SIZE];
patch = @view input_padded[v_w:(v_w+k-1), v_h:(v_h+k-1), 1:end, 1:BATCH_SIZE];

@tullio verbose = 2 patch_mean[in_c, b] := patch[w, h, in_c, b] / ($k * $k)
@tullio verbose = 2 patch_centered[w, h, in_c, b] := patch[w, h, in_c, b] - patch_mean[in_c, b]

DEFAULT_ϵ = 0.00001f0

@tullio verbose = 2 patch_norm2[c_in, batch] := begin
    patch_mean[in_c, b] = patch[w, h, in_c, b] / ($k * $k)

    # Second moment adjusted for DEFAULT_ϵ and q
    patch[w, h, c_in, batch] - patch_mean[c_in, batch]
    # patch_centered[w, h, in_c, b] = patch[w, h, in_c, b] - patch_mean[in_c, b]
    # patch_norm[in_c, b] = (patch[w, h, in_c, b] - patch_mean[in_c, b])^2 |> sqrt

    # $ prefix is for constants with no gradient calculation
    # (patch_norm[in_c, b] + $DEFAULT_ϵ + q) |> Float32
end


@tullio verbose = 2 sk[b] := begin
    patch_mean[in_c, b] = patch[w, h, in_c, b] / ($k * $k)

    patch_centered[w, h, in_c, b] = patch[w, h, in_c, b] - patch_mean[in_c, b]

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
end (b in 1:BATCH_SIZE)

# Exponent maps -∞:+∞ to 0:+∞
# return Float32(sign(sk) * (abs(sk))^log(1 + exp(p)))
return Float32.(sign.(sk) * (abs.(sk)) .^ p)

in_c, out_c = 1, 24
W = rand(Float32, 5, 5, 1, 24)

# WARNING: no := within a @tullio block. Only =
# @tullio sk[b] := begin
@tullio verbose = 2 A := begin
    patch_mean[in_c, b] = patch[w, h, in_c, b] / ($k * $k)

    # Second moment adjusted for DEFAULT_ϵ and q
    patch_centered[w, h, in_c, b] = patch[w, h, in_c, b] - patch_mean[in_c, b]
    patch_norm[in_c, b] = (patch[w, h, in_c, b] - patch_mean[in_c, b])^2 |> sqrt |> (x) -> x + $DEFAULT_ϵ + q |> (x) -> Float32(x)

    # $ prefix is for constants with no gradient calculation
    # patch_norm_floored[in_c, b] = patch_norm[in_c, b] + $DEFAULT_ϵ + q |> Float32

    # Average convolution kernel (channel-wise)
    W_cur[w, h, in_c] = W[w, h, in_c, $out_c]
    # W_mean[in_c] = W_cur[w, h, in_c] / ($k * $k)
    # W_centered[w, h, in_c] = W_cur[w, h, in_c] - W_mean[in_c]

    # Second moment adjusted for DEFAULT_ϵ + q
    # W_norm[in_c] = (W_cur[w, h, in_c] - W_mean[in_c])^2 |> sqrt
    # W_norm_floored[in_c] = (W_norm[in_c] + $DEFAULT_ϵ + q) |> Float32

    # Calculate the sharpened cross-correlation for all channel combinations
    # No loop to modify single values. Otherwise Zygote complains
    # Note the transpose of the sum!!! Otherwise mismatch with col extraction
    # Calculate the sharpened cross-correlation

    # (patch_centered[w, h, in_c, b] * W_centered[w, h, in_c]) /
    # (patch_norm_floored[in_c, b] * W_norm_floored[in_c])
end

###################################################################################################


A = rand(4, 4, 5, 5)
B = rand(4, 4)

@tullio C[k, l] := A[i, j, k, l] * B[i, j]
@tullio C[k, l] := A[i, j, k, l] * B[i, j] |> sqrt
@tullio C[k, l] := sqrt <| A[i, j, k, l] * B[i, j]

@tullio C[k, l] := begin
    A[i, j, k, l] * B[i, j]
end

@tullio C[k, l] := begin
    A[i, j, k, l] * B[i, j]
end

@tullio C[i, j] := begin
    B[i, j]
end

@tullio C[i, j] := begin
    B[i, j] |> (x) -> x + 100
end

@tullio D[k, l] := begin
    A[i, j, k, l] * B[i, j]
end

@tullio D[k, l] := begin
    A[i, j, k, l] * B[i, j] |> (x) -> x + 100
end

using Tullio

A = rand(4, 4, 5, 5)
B = rand(4, 4)

@tullio B2[i, j] := begin
    B3[i, j] = B[i, j]
end

@tullio verbose = 2 D[k, l] := begin
    C[k, l] = A[i, j, k, l] * B[i, j]
    C[k, l]
end

@tullio D[k, l] := begin
    A[i, j, k, l] * B[i, j]
end



@tullio D[k, l] := begin
    C[k, l] = A[i, j, k, l] * B[i, j]
    C[k, l] |> (x) -> x + 100
end




@tullio D[k, l] := begin
    A[i, j, k, l] * B[i, j] |> (x) -> x + 100
end

@tullio D[k, l] := begin
    C[k, l] = A[i, j, k, l] * B[i, j]
    C[k, l] |> (x) -> x + 100
    # C[k, l] = C[k, l] |> Float32
end



@tullio D[k, l] := begin
    C[k, l] = A[i, j, k, l] * B[i, j]
    C[k, l] |> (x) -> x + 100
    # C[k, l] = C[k, l] |> Float32
end

@tullio C[k, l] := begin
    sqrt <| A[i, j, k, l] * B[i, j]
end

###################################################################################################

using Tullio

A = rand(4, 4, 5, 5)
B = rand(4, 4)

# OK
@tullio C1[k, l] := A[i, j, k, l] * B[i, j]
C1

# OK
@tullio C2[k, l] := begin
    A[i, j, k, l] * B[i, j]
end


# TMP was just used, but defined in a local scope. Not exported.
# This doesn't work
@tullio C4 := begin
    TMP4 = A[i, j, k, l] * B[i, j]
    TMP4
end

@tullio C5 := begin
    A[i, j, k, l] * B[i, j]
end


@tullio C6[k, l] := begin
    TMP6 = A[i, j, k, l] * B[i, j]
end

@tullio C6_2[k, l] := begin
    TMP6[k, l] = A[i, j, k, l] * B[i, j]
end


@tullio verbose = 2 C7 := begin
    TMP7[k, l] = A[i, j, k, l] * B[i, j]
    TMP7
end


@tullio C8[k, l] := begin
    TMP8[k, l] = A[i, j, k, l] * B[i, j]
    TMP8
end


@tullio C9[k, l] := begin
    TMP9[k, l] = A[i, j, k, l] * B[i, j]
end


@tullio C10[k, l] := begin
    TMP10[k, l] = A[i, j, k, l] * B[i, j]
    TMP10[k, l]
end


TMP11 = zeros(5, 5)
@tullio C11[k, l] := begin
    TMP11[k, l] = A[i, j, k, l] * B[i, j]
    TMP11[k, l]
end

TMP12 = zeros(5, 5)
@tullio C12[k, l] := begin
    TMP12[k, l] = A[i, j, k, l] * B[i, j]
end

TMP13 = zeros(5, 5)
@tullio C11[k, l] := begin
    TMP13[k, l] = A[i, j, k, l] * B[i, j]
    TMP13
end



###################################################################################################

A = rand(4, 4, 5, 5)
B = rand(5, 5, 6, 6)
C = rand(6, 6)

@tullio D1[i, j, m, n] := A[i, j, k, l] * B[k, l, m, n]
@tullio E1[i, j] := D1[i, j, m, n] * C[m, n]

# Breaks D2 undefined. Exoected with soft local scope. Not within macro that should define the array.
@tullio F2 := begin
    D2[i, j, m, n] = A[i, j, k, l] * B[k, l, m, n]
    E2[i, j] = D[i, j, m, n] * C[m, n]
end

# Breaks D3 undefined. No idea why
@tullio F3 := begin
    D3 = A[i, j, k, l] * B[k, l, m, n]
    D3[i, j, m, n] * C[m, n]
end

# Bounds error
D4 = zeros(4, 4, 6, 6)
@tullio F4 := begin
    D4 = A[i, j, k, l] * B[k, l, m, n]
    E4 = D4[i, j, m, n] * C[m, n]
end

# Bounds error
D8 = zeros(4, 4, 6, 6)
@tullio F8[i, j] := begin
    D8 = A[i, j, k, l] * B[k, l, m, n]
    E8 = D8[i, j, m, n] * C[m, n]
end

# Bounds error
D9 = zeros(4, 4, 6, 6)
E9 = zeros(4, 4)
@tullio F9[i, j] := begin
    D9 = A[i, j, k, l] * B[k, l, m, n]
    E9 = D9[i, j, m, n] * C[m, n]
end


# Breaks: returns a sum instead of 2 x 2 matrix
D5 = zeros(4, 4, 6, 6)
@tullio F5 := begin
    D5[i, j, m, n] = A[i, j, k, l] * B[k, l, m, n]
    E5 = D5[i, j, m, n] * C[m, n]
end

# Breaks: E6 not defined
D6 = zeros(4, 4, 6, 6)
@tullio F6 := begin
    D6[i, j, m, n] = A[i, j, k, l] * B[k, l, m, n]
    E6[i, j] = D6[i, j, m, n] * C[m, n]
end

# Works! D7 is modified as soft local scope
# D7 needs to be defined outside of scope and captured
# E7 cannot be already defined. See example 9
D7 = zeros(4, 4, 6, 6)
@tullio F7[i, j] := begin
    D7[i, j, m, n] = A[i, j, k, l] * B[k, l, m, n]
    E7 = D7[i, j, m, n] * C[m, n]
end
D7

# Breaks D10 undefined
@tullio F10[i, j] := begin
    D10 = zeros(4, 4, 6, 6)
    D10[i, j, m, n] = A[i, j, k, l] * B[k, l, m, n]
    E10 = D10[i, j, m, n] * C[m, n]
end



@show a = 2

function test()
    return a = 2
end

test()



###################################################################################################