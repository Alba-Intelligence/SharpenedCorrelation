#!/usr/bin/env julia
# coding: utf-8

# ___WORK IN PROGRESS___

#
# # Prologue
#
# ## The Sharpened Cosine Similarity
#
# The Sharpened Cosine Similarity is a modified form of cosine distance that showed over the past few months, thought up by Brandon Rohrer .
#
#
#
# [https://www.rpisoni.dev/posts/cossim-convolution/]()
# [https://www.rpisoni.dev/posts/cossim-convolution-part2/]()

# The usual cosine transform
#
# $$scs(s, k) = \frac{s \cdot k}{\Vert{s}\Vert \Vert{k} \Vert}$$
#
# is modified It looks like
#
# $$scs(s, k) = sign(s \cdot k)\Biggl(\frac{s \cdot k}{(\Vert{s}\Vert + q)(\Vert{k}\Vert + q)}\Biggr)^p$$

# ## How does it work?
#
# 2 Parameters are introduced:
#
# - $q$ to floor the value of the norm of either vector;
# - $p$ an exponentiation factor to decide
#
# Let's imagine the case of a picture. Some zones might contain a lot of information (e.g. an airplane); some just noise (e.g. various shades of blue in the background).

# ### Ignore the irrelevant
#
# When running a convolution over the picture, and under the assumption that the entire picture has been normalised, the norm of a patch around the plane will be higher that over the sky. But, when it comes to apply the cosine transform, those local vectors will be renormalised to 1. The effect is that the the convolution filters will be trained to find information assuming that each of those 2 zones are of equal relevance.
#
# The $q$ parameter helps. Over a patch of sky, $q$ will be much higher than $\Vert{s}\Vert$. Therefore the value of the convolution will be seriously decreased.
#
# No training budget will be wasted on training on the useless. $q$ clips the noise out.

# ### Focus on anything even remotely interesting, or only zoom in the critical
#
# The $p$ exponentiation parameters could be anything from $0^+$ to $+\infty$.
#
# If $p = 1$, we have the normal cosine distance.
#
# If $p > 1$, we have a convex function. At the extreme, we have a curve that is almost $0$ almost everywhere until the cosine distance becomes close to $1$. __If $p > 1$, we only look at the truely interesting patches__. __$p > 1$ is super sensitivity__
#
# Conversely, if $p < 1$, we get a concave curve. For very low (positive) values of $p$, the curve will be close to $1$ everywhere until the cosine distance gets close to $0$. __If $p \rightarrow 0$, the truely useless is ignored, anything else is worth considering__. __$p \rightarrow 0$ is super specificity__.
#
# (P.S. despite decades of exposure to those words, I still reach out to a dictionary to know specificity vs.sensitivity. Intuition A+. Vocabulary. Z-)
#

# ## How to improve?
#
# A few improvements, some implemented in the code, some not.

# ### Keep $p$ reasonable (implemented)
#
# Instead of using a parameter $p$, use a _Soft ReLU_ shape:
#
# $$scs(s, k) = sign(s \cdot k)\Biggl(\frac{s \cdot k}{(\Vert{s}\Vert + q)(\Vert{k}\Vert + q)}\Biggr)^ {\log \left( 1 + \exp \left( p \right) \right) }$$
#
# $p$ can now range from $-\infty$ to $+\infty$ with the highest gradients around 0, i.e. around an exponent of 1.
#

# ### Get rid of the cosine transform and go for a real Pearson cross-correlation  (implemented)
#
# Cosine distance is great for embeddings. But for determining similarities, not so much. Let's go for a true cross-correlation.
#
# $$S = \sum{s - \bar{s}}$$
# $$K = \sum{k - \bar{k}}$$
#
# $$scs(s, k) = sign(S \times K))\Biggl(\frac{S \times K}{\left( \sqrt{(\sum{s - \bar{s})^2}} + q \right) \cdot \left( \sqrt{(\sum{k - \bar{k})^2}} + q \right)}\Biggr)^ {\log \left( 1 + \exp \left( p \right) \right) }
# $$
#
#
# A bit more of a mouthful and would translate into a dog's breakfast in TensorFlow...
#

# ### Robustify (not implemented)
#
# $q$ brings noise removal. But the cross-correlation is still sensitive to outliers. Clipping for example at σ = 2 (Winsorised correlation) after re-normalisation, Spearman or one of myriad other variants with different performance profiles.

# ### Use actual information content as the norm  (not implemented)
#
# When the Salvator Mundi was put on the market a couple of years ago (and eventually bought for half a billion US), numerous methods were used to confirm its authenticity. An interesting approach came from an AI company that had aplly a similar method to Rembrandt's. See [https://www.art-critique.com/en/2019/04/a-eye-another-tool-for-the-authenticating-artworks/](), on Arxiv [https://arxiv.org/abs/2005.10600]() for a (hardly) little more technical content on Salvator Mundi work, this [https://arxiv.org/abs/1907.12436]() for a bit more, and the code at [https://github.com/stevenjayfrank/A-Eye]().
#
# Key to their method method is to focus the training and inference on sections of the painting where the information content is high.
#
# This could be use to replace the $q$ parameter.
#

# # Show me the __Julia__ code
#
# Many have contributed TensorFlow and PyTorch implementations. See [https://e2eml.school/scs.html]().
#
# Time for some Julia supremacy with the [https://fluxml.ai/](Flux.jl) library. Code is in the repo. Running under Julia 1.7, packages upgraded to the latest versions.
#

# ## Using the MNIST dataset


# In[1]:

pkg_path = normpath(joinpath(@__DIR__, ".."))
cd(pkg_path)

using Pkg
Pkg.activate(; temp = true)

# Speed up by avoiding updating the repository when adding packages
Pkg.UPDATED_REGISTRY_THIS_SESSION[] = true

# Add useful package
Pkg.add([
    "Revise",
    "ProgressMeter",
    "BenchmarkTools",
    "Debugger",
    "Images",
    "Flux",
    "MLDatasets",
    "ImageCore",
    "PaddedViews",
])

using Revise,
    ProgressMeter, BenchmarkTools, Debugger, Images, Flux, MLDatasets, ImageCore, PaddedViews

# In[2]:

Pkg.develop(path = pkg_path)

using SharpenedCorrelation


# We here only use the MNIST dataset described in [https://juliaml.github.io/MLDatasets.jl/stable](). If not already available in `~/.julia/datadeps/`, it will be automatically downloaded. Some `MLDataSets` request accepting the terms of use. If a prompt appears, type `y` + `Enter`.
#
# Let's just check that we see something.
#
# (To download for example `MNIST` separately: `MNIST.download(; i_accept_the_terms_of_use=true)`)

# In[3]:


# Image does not show up in Github
using ImageCore


# Let's use global variables to easily use other datasets
WIDTH, HEIGHT, N_TRAIN_SAMPLE = size(MNIST(split = :train).features);
_, _, N_TEST_SAMPLE = size(MNIST(split = :test).features);
N_CHANNELS = 1;


# Flux helps with the data loaders.

# In[8]:

train_x = MNIST(split = :train).features;
train_y = MNIST(split = :train).targets;
test_x = MNIST(split = :test).features;
test_y = MNIST(split = :test).targets;


# In[9]:


# Insert a channel dimension for the MNIST dataset
train_x = reshape(train_x, WIDTH, HEIGHT, N_CHANNELS, N_TRAIN_SAMPLE);
test_x = reshape(test_x, WIDTH, HEIGHT, N_CHANNELS, N_TEST_SAMPLE);

# reencode the labels as One-Hot
train_y = Flux.onehotbatch(train_y, 0:9);
test_y = Flux.onehotbatch(test_y, 0:9);

# `train_x` and `train_y` are functions that deliver the data. They are combined into a data loader.

# In[10]:


# Create DataLoaders (mini-batch iterators)
BATCH_SIZE = 256;
train_loader =
    Flux.DataLoader((data = train_x, label = train_y), batchsize = BATCH_SIZE, shuffle = true);
test_loader =
    Flux.DataLoader((data = test_x, label = test_y), batchsize = BATCH_SIZE, shuffle = false);


# ## Sharpened Cross-correlation Similarity
#
# Let's start building the layers

# In[12]:


using Debugger
using Flux, MLDatasets, ImageCore, PaddedViews

data, label = first(train_loader);


# In[20]:

# Let's create a single Sharpened Cosine Transform with a single B&W input channel yielding 8 output channels. The other 3 parameters are the kernel size, padding width around an image (padded at 0,0), and the stride.

# Setup all the relevant model variables with values (here are the default values taken from [https://github.com/brohrer/sharpened_cosine_similarity_torch/blob/main/demo_cifar10.py]()):
#
# - Batch size: `batchsize = 1_024`
# - Maximum learning rate:  `max_lr = 0.01`
# - Number of classes: `n_classes = 10`
# - Number of training epochs: `n_epochs = 100`
# - Number of runs: `n_runs::Int = 1_000`
#
# as well as description of the model layers.
#
# `
# block_params::Dict = Dict(1 => [3,  0, 0, 0, 0, 0],
#                           2 => [16, 3, 2, 1, 2, 2],
#                           3 => [24, 3, 2, 1, 2, 2],
#                           4 => [48, 3, 2, 1, 2, 2])
# `
#
# Here, the model receives input described in layer 1, followed by 3 block layers of _Sharpened Cosine Similarity_ + _Batch Normalisation_ + _Maximum Absolute Pooling_, then followed by a final _linear_ layer (called `MultiDense` in the code) to generate the final classes.
#
# Each entry for a dual layer is of the format: `[n_out, scs_kernel, stride, scs_padding, width/height of the max pooling]`. For the input layer 1, the input channels are irrelevant, the layer only generates the input values.
#
# Let's start with the smallest model possible, a single layer.
#

# In[14]:


model_params = HyperParameters(;
    batchsize = BATCH_SIZE,
    n_classes = size(train_y)[1],
    n_epochs = 100,
    n_runs = 500,
    block_params = Dict(1 => [1, 0, 0, 0, 0], 2 => [24, 5, 2, 1, 8]),
)


# We create a full model.

# In[15]:

sc = SharpenedCorrelationModel(model_params, WIDTH, HEIGHT)


# A Flux layer expects the following input dimension: `width` x `height` x `channels` x `batch size`. There for we need to reshape an image before checking that the layer actually works.

# In[16]:


# Check it works on single image
img = @btime train_x[:, :, 1:1, 1:1]
size(img)


# In[23]:


# Let's check it doesn't bug out on a single image. Size should be n classes x batch size (here 1)
out_label = @time sc(img)
# RUN FROM THE REPL ONLY Debugger.@enter out_label = sc(img) / Debugger.@run out_label = sc(img)


# In[ ]:


size(out_label)


# ## Loss function and optimiser

# In[15]:


function loss_and_accuracy(data_loader, model, device)
    accuracy = 0
    loss = 0.0f0
    count = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        loss += Flux.Losses.logitcrossentropy(ŷ, y, agg = sum)
        accuracy += sum(onecold(ŷ) .== onecold(y))
        count += size(x)[end]
    end
    return loss / count, accuracy / count
end


# In[16]:


optimiser = ADAM(model_params.max_lr);
ps = Flux.params(sc)


# ## Training - Work in progress

# In[26]:


data, label = first(train_loader);


# In[73]:


l1 = SharpenedConvolution(WIDTH, HEIGHT, 1, 16, 5, 2, 1)


# In[75]:


l1(data)


# In[20]:


using ProfileView


# In[21]:


@code_warntype l1(data)


# In[22]:


@profview l1(data)


# In[22]:


@btime Flux.Losses.logitcrossentropy(sc(data), label)


# In[23]:


gradient(() -> Flux.Losses.logitcrossentropy(sc(data), label), ps)


# In[24]:


# Work in progress.

## Training
# @showprogress for epoch in 1:model_params.n_epochs
@showprogress for epoch = 1:2
    for (data, label) in train_loader
        # Transfer data to device - Uncomment to use automatic detection in the package
        # data, label = SCS.Training_Device(data), SCS.Training_Device(label)

        # Compute gradient
        gs = gradient(() -> Flux.Losses.logitcrossentropy(sc(data), label), ps)
        Flux.Optimise.update!(optimiser, ps, gs) # update parameters
    end

    # Report on train and test
    train_loss, train_acc = loss_and_accuracy(train_loader, scs, SCS.Training_Device)
    test_loss, test_acc = loss_and_accuracy(test_loader, scs_model, SCS.Training_Device)
    println("Epoch: $(epoch) / $(model_params.n_epochs)")
    println("  train_loss = $(train_loss), train_accuracy = $(train_acc)")
    println("  test_loss = $(test_loss), test_accuracy = $(test_acc)")
end
