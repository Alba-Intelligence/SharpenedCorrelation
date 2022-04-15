___WORK IN PROGRESS___


# Prologue

## The Sharpened Cosine Similarity

The Sharpened Cosine Similarity is a modified form of cosine distance that showed over the past few months, thought up by Brandon Rohrer .



[https://www.rpisoni.dev/posts/cossim-convolution/]()
[https://www.rpisoni.dev/posts/cossim-convolution-part2/]()

The usual cosine transform 

$$scs(s, k) = \frac{s \cdot k}{\Vert{s}\Vert \Vert{k} \Vert}$$ 

is modified It looks like

$$scs(s, k) = sign(s \cdot k)\Biggl(\frac{s \cdot k}{(\Vert{s}\Vert + q)(\Vert{k}\Vert + q)}\Biggr)^p$$

## How does it work?

2 Parameters are introduced: 

- $q$ to floor the value of the norm of either vector;
- $p$ an exponentiation factor to decide 

Let's imagine the case of a picture. Some zones might contain a lot of information (e.g. an airplane); some just noise (e.g. various shades of blue in the background).

### Ignore the irrelevant

When running a convolution over the picture, and under the assumption that the entire picture has been normalised, the norm of a patch around the plane will be higher that over the sky. But, when it comes to apply the cosine transform, those local vectors will be renormalised to 1. The effect is that the the convolution filters will be trained to find information assuming that each of those 2 zones are of equal relevance. 

The $q$ parameter helps. Over a patch of sky, $q$ will be much higher than $\Vert{s}\Vert$. Therefore the value of the convolution will be seriously decreased. 

No training budget will be wasted on training on the useless. $q$ clips the noise out.

### Focus on anything even remotely interesting, or only zoom in the critical

The $p$ exponentiation parameters could be anything from $0^+$ to $+\infty$.

If $p = 1$, we have the normal cosine distance. 

If $p > 1$, we have a convex function. At the extreme, we have a curve that is almost $0$ almost everywhere until the cosine distance becomes close to $1$. __If $p > 1$, we only look at the truely interesting patches__. __$p > 1$ is super sensitivity__ 

Conversely, if $p < 1$, we get a concave curve. For very low (positive) values of $p$, the curve will be close to $1$ everywhere until the cosine distance gets close to $0$. __If $p \rightarrow 0$, the truely useless is ignored, anything else is worth considering__. __$p \rightarrow 0$ is super specificity__.

(P.S. despite decades of exposure to those words, I still reach out to a dictionary to know specificity vs.sensitivity. Intuition A+. Vocabulary. Z-)


## How to improve?

A few improvements, some implemented in the code, some not.

### Keep $p$ reasonable (implemented)

Instead of using a parameter $p$, use a _Soft ReLU_ shape:

$$scs(s, k) = sign(s \cdot k)\Biggl(\frac{s \cdot k}{(\Vert{s}\Vert + q)(\Vert{k}\Vert + q)}\Biggr)^ {\log \left( 1 + \exp \left( p \right) \right) }$$

$p$ can now range from $-\infty$ to $+\infty$ with the highest gradients around 0, i.e. around an exponent of 1.


### Get rid of the cosine transform and go for a real Pearson cross-correlation  (implemented)

Cosine distance is great for embeddings. But for determining similarities, not so much. Let's go for a true cross-correlation.

$$S = \sum{s - \bar{s}}$$
$$K = \sum{k - \bar{k}}$$

$$scs(s, k) = sign(S \times K))\Biggl(\frac{S \times K}{\left( \sqrt{(\sum{s - \bar{s})^2}} + q \right) \cdot \left( \sqrt{(\sum{k - \bar{k})^2}} + q \right)}\Biggr)^ {\log \left( 1 + \exp \left( p \right) \right) }
$$


A bit more of a mouthful and would translate into a dog's breakfast in TensorFlow...


### Robustify (not implemented)

$q$ brings noise removal. But the cross-correlation is still sensitive to outliers. Clipping for example at σ = 2 (Winsorised correlation) after re-normalisation, Spearman or one of myriad other variants with different performance profiles. 

### Use actual information content as the norm  (not implemented)

When the Salvator Mundi was put on the market a couple of years ago (and eventually bought for half a billion US), numerous methods were used to confirm its authenticity. An interesting approach came from an AI company that had aplly a similar method to Rembrandt's. See [https://www.art-critique.com/en/2019/04/a-eye-another-tool-for-the-authenticating-artworks/](), on Arxiv [https://arxiv.org/abs/2005.10600]() for a (hardly) little more technical content on Salvator Mundi work, this [https://arxiv.org/abs/1907.12436]() for a bit more, and the code at [https://github.com/stevenjayfrank/A-Eye]().

Key to their method method is to focus the training and inference on sections of the painting where the information content is high.

This could be use to replace the $q$ parameter.


# Show me the __Julia__ code

Many have contributed TensorFlow and PyTorch implementations. See [https://e2eml.school/scs.html]().

Time for some Julia supremacy with the [https://fluxml.ai/](Flux.jl) library. Code is in the repo. Running under Julia 1.7, packages upgraded to the latest versions.


## Using the MNIST dataset


```julia
cd(@__DIR__); using Revise, Pkg; Pkg.activate(@__DIR__);
```

    [32m[1m  Activating[22m[39m project at `~/Development/julia/projects/SharpenedCorrelation`



```julia
using ProgressMeter
using BenchmarkTools
using Flux
using MLDatasets
```

We here only use the MNIST dataset described in [https://juliaml.github.io/MLDatasets.jl/stable](). If not already available in `~/.julia/datadeps/`, it will be automatically downloaded. Some `MLDataSets` request accepting the terms of use. If a prompt appears, type `y` + `Enter`.

Let's just check that we see something.

(To download for example `MNIST` separately: `MNIST.download(; i_accept_the_terms_of_use=true)`)


```julia
# Image does not show up in Github
using ImageCore
```


```julia
MNIST.convert2image(MNIST.traintensor(1))
```


    
![svg](README_files/README_17_0.svg)
    



```julia
MNIST.trainlabels(1)
```


    5



```julia
size(MNIST.testtensor())
```


    (28, 28, 10000)



```julia
# Let's use global variables to easily use other datasets
WIDTH, HEIGHT, N_TRAIN_SAMPLE = size(MNIST.traintensor());
_, _, N_TEST_SAMPLE = size(MNIST.testtensor());
N_CHANNELS = 1;
```

Flux helps with the data loaders.


```julia
train_x, train_y = MNIST.traindata(Float32);
test_x, test_y = MNIST.testdata(Float32);
```


```julia
# Insert a channel dimension for the MNIST dataset
train_x = reshape(train_x, WIDTH, HEIGHT, N_CHANNELS, N_TRAIN_SAMPLE);
test_x = reshape(test_x, WIDTH, HEIGHT, N_CHANNELS, N_TEST_SAMPLE);

# reencode the labels as One-Hot
train_y, test_y = Flux.onehotbatch(train_y, 0:9), Flux.onehotbatch(test_y, 0:9);
```

`train_x` and `train_y` are functions that deliver the data. They are combined into a data loader.


```julia
# Create DataLoaders (mini-batch iterators)
BATCH_SIZE = 256;
train_loader = Flux.DataLoader((data=train_x, label=train_y), batchsize=BATCH_SIZE, shuffle=true);
test_loader = Flux.DataLoader((data=test_x, label=test_y), batchsize=BATCH_SIZE, shuffle=false);
```

## Sharpened Cross-correlation Similarity

Let's start building the layers


```julia
using Pkg; Pkg.activate(".");

using Debugger
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
train_loader = Flux.DataLoader((data=train_x, label=train_y), batchsize=BATCH_SIZE, shuffle=true);
test_loader = Flux.DataLoader((data=test_x, label=test_y), batchsize=BATCH_SIZE, shuffle=false);

data, label = first(train_loader);


```

    [32m[1m  Activating[22m[39m project at `~/Development/julia/projects/SharpenedCorrelation`



```julia
using Debugger
using SharpenedCorrelation
```

Let's create a single Sharpened Cosine Transform with a single B&W input channel yielding 8 output channels. The other 3 parameters are the kernel size, padding width around an image (padded at 0,0), and the stride. 

Setup all the relevant model variables with values (here are the default values taken from [https://github.com/brohrer/sharpened_cosine_similarity_torch/blob/main/demo_cifar10.py]()):

- Batch size: `batchsize = 1_024`
- Maximum learning rate:  `max_lr = 0.01`
- Number of classes: `n_classes = 10`
- Number of training epochs: `n_epochs = 100`
- Number of runs: `n_runs::Int = 1_000`

as well as description of the model layers.

`
block_params::Dict = Dict(1 => [3,  0, 0, 0, 0, 0],
                          2 => [16, 3, 2, 1, 2, 2],
                          3 => [24, 3, 2, 1, 2, 2],
                          4 => [48, 3, 2, 1, 2, 2])
`

Here, the model receives input described in layer 1, followed by 3 block layers of _Sharpened Cosine Similarity_ + _Batch Normalisation_ + _Maximum Absolute Pooling_, then followed by a final _linear_ layer (called `MultiDense` in the code) to generate the final classes. 

Each entry for a dual layer is of the format: `[n_out, scs_kernel, stride, scs_padding, width/height of the max pooling]`. For the input layer 1, the input channels are irrelevant, the layer only generates the input values.

Let's start with the smallest model possible, a single layer.



```julia
model_params = HyperParameters(;
    batchsize = BATCH_SIZE, 
    n_classes = size(train_y)[1], 
    n_epochs = 100, 
    n_runs = 500, 
    block_params = Dict(
        1 => [1,   0, 0, 0, 0], 
        2 => [24,  5, 2, 1, 8]))

```


    HyperParameters(256, 10, 100, 500, 0.01f0, Dict(2 => [24, 5, 2, 1, 8], 1 => [1, 0, 0, 0, 0]))


We create a full model.


```julia
sc = SharpenedCorrelationModel(model_params, WIDTH, HEIGHT)
```

    Creating block with output dimension of the SC layer:
    width in = 28 x height in 28
    channels in 1 - out channels 24
    
    



    SharpenedCorrelationModel(Chain(SharpenedConvolution
      W: Array{Float32}((5, 5, 1, 24)) [-0.031233555 -0.07007872 … -0.026747588 -0.03856306; 0.06355132 -0.009666526 … -0.0025519917 -0.047310982; … ; 0.018177062 -0.071618915 … -0.04488449 0.08430854; 0.08547553 -0.08475591 … -0.05108846 0.02518769;;;; -0.0236827 -0.04122835 … 0.015123422 -0.05662714; -0.013727969 -0.096033506 … -0.04438845 -0.07119618; … ; -0.036920853 0.074999586 … -0.027951803 0.02913976; -0.085526235 0.08498706 … 0.034141857 0.09705671;;;; 0.088725865 -0.009792378 … -0.02564953 0.007713301; -0.087156564 -0.08501619 … -0.036866225 -0.07188923; … ; 0.016780043 -0.083418995 … 0.08992146 0.09478006; -0.06262868 0.010099202 … -0.031116182 0.062464505;;;; … ;;;; 0.06902271 -0.07821407 … -0.06542191 0.021741284; -0.034649838 -0.05076378 … 0.06995046 -0.016830921; … ; 0.03972163 0.027923573 … -0.07625523 -0.041069616; 0.033891834 -0.0077314167 … -0.04129378 -0.0040707407;;;; -0.046357643 -0.06940553 … -0.03049646 0.06404475; 0.038377535 -0.07978094 … 0.08572427 -0.046445455; … ; 0.041246265 0.046869174 … 0.020374084 -0.03540311; -0.08714931 -0.074553795 … -0.039860833 -0.08235737;;;; -0.06959252 -0.05420857 … -0.008168158 0.04107403; 0.0077349325 0.071501225 … 0.060124844 -0.05800553; … ; 0.0737097 0.08966871 … 0.007337121 -0.0037692194; 0.08179974 0.022886071 … -0.037506573 -0.0050130426]
      p: Float32 1.0f0
      q: Float32 0.1f0
      kernel: Int64 5
      stride: Int64 2
      padding: Int64 1
      out_width: Int64 13
      out_height: Int64 13
      out_chan: Int64 24
    , BatchNorm(24, relu), AdaptiveMaxPool((8, 8)), flatten, Dense(1536, 10)), HyperParameters(256, 10, 100, 500, 0.01f0, Dict(2 => [24, 5, 2, 1, 8], 1 => [1, 0, 0, 0, 0])), 28, 28)


A Flux layer expects the following input dimension: `width` x `height` x `channels` x `batch size`. There for we need to reshape an image before checking that the layer actually works.


```julia
# Check it works on single image
img = @btime train_x[:, :, 1:1, 1:1]
size(img)
```

      1.760 μs (7 allocations: 3.45 KiB)



    (28, 28, 1, 1)



```julia
using Debugger
```


```julia
# Let's check it doesn't bug out on a single image. Size should be n classes x batch size (here 1)
# out_label = @btime sc(img)
@enter out_label = sc(img)

```


    Debugger.jl needs to be run in a Julia REPL


    


    Stacktrace:


      [1] error(s::String)


        @ Base ./error.jl:35


      [2] RunDebugger(frame::JuliaInterpreter.Frame, repl::Nothing, terminal::Nothing; initial_continue::Bool)


        @ Debugger ~/.julia/packages/Debugger/APRPi/src/repl.jl:15


      [3] RunDebugger (repeats 2 times)


        @ ~/.julia/packages/Debugger/APRPi/src/repl.jl:13 [inlined]


      [4] top-level scope


        @ ~/.julia/packages/Debugger/APRPi/src/Debugger.jl:127


      [5] eval


        @ ./boot.jl:368 [inlined]


      [6] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)


        @ Base ./loading.jl:1277


      [7] #invokelatest#2


        @ ./essentials.jl:729 [inlined]


      [8] invokelatest


        @ ./essentials.jl:727 [inlined]


      [9] (::VSCodeServer.var"#164#165"{VSCodeServer.NotebookRunCellArguments, String})()


        @ VSCodeServer ~/.vscode/extensions/julialang.language-julia-1.6.17/scripts/packages/VSCodeServer/src/serve_notebook.jl:19


     [10] withpath(f::VSCodeServer.var"#164#165"{VSCodeServer.NotebookRunCellArguments, String}, path::String)


        @ VSCodeServer ~/.vscode/extensions/julialang.language-julia-1.6.17/scripts/packages/VSCodeServer/src/repl.jl:184


     [11] notebook_runcell_request(conn::VSCodeServer.JSONRPC.JSONRPCEndpoint{Base.PipeEndpoint, Base.PipeEndpoint}, params::VSCodeServer.NotebookRunCellArguments)


        @ VSCodeServer ~/.vscode/extensions/julialang.language-julia-1.6.17/scripts/packages/VSCodeServer/src/serve_notebook.jl:13


     [12] dispatch_msg(x::VSCodeServer.JSONRPC.JSONRPCEndpoint{Base.PipeEndpoint, Base.PipeEndpoint}, dispatcher::VSCodeServer.JSONRPC.MsgDispatcher, msg::Dict{String, Any})


        @ VSCodeServer.JSONRPC ~/.vscode/extensions/julialang.language-julia-1.6.17/scripts/packages/JSONRPC/src/typed.jl:67


     [13] serve_notebook(pipename::String, outputchannel_logger::Base.CoreLogging.SimpleLogger; crashreporting_pipename::String)


        @ VSCodeServer ~/.vscode/extensions/julialang.language-julia-1.6.17/scripts/packages/VSCodeServer/src/serve_notebook.jl:136


     [14] top-level scope


        @ ~/.vscode/extensions/julialang.language-julia-1.6.17/scripts/notebook/notebook.jl:32


     [15] include(mod::Module, _path::String)


        @ Base ./Base.jl:422


     [16] exec_options(opts::Base.JLOptions)


        @ Base ./client.jl:303


     [17] _start()


        @ Base ./client.jl:522



```julia
size(out_label)
```

## Loss function and optimiser


```julia
function loss_and_accuracy(data_loader, model, device)
    accuracy = 0
    loss = 0.0f0
    count = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        loss += Flux.Losses.logitcrossentropy(ŷ, y, agg=sum)
        accuracy += sum(onecold(ŷ) .== onecold(y))
        count +=  size(x)[end]
    end
    return loss / count, accuracy / count
end
```




    loss_and_accuracy (generic function with 1 method)




```julia
optimiser = ADAM(model_params.max_lr);
ps = Flux.params(sc)
```


    UndefVarError: model_params not defined

    

    Stacktrace:

     [1] top-level scope

       @ In[16]:1

     [2] eval

       @ ./boot.jl:373 [inlined]

     [3] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

       @ Base ./loading.jl:1196


## Training - Work in progress


```julia
data, label = first(train_loader);
```


```julia
l1 = SharpenedConvolution(WIDTH, HEIGHT, 1, 16, 5, 2, 1)
```




    SharpenedConvolution
      W: Array{Float32}((5, 5, 1, 16)) [-0.09097837 -0.07746911 … -0.0782877 0.10234335; -0.07949978 0.035947695 … -0.03898492 0.09566406; … ; -0.031589366 -0.08400774 … -0.118056305 -0.06652068; 0.05039504 0.072273605 … 0.036681302 -0.06439322;;;; 0.06528427 -0.031971585 … -0.014891504 0.021438185; 0.002922749 0.030369377 … -0.030466728 -0.010516672; … ; 0.047203455 -0.06020778 … -0.1088815 0.041800275; -0.012337831 0.0004136222 … 0.020026186 -0.063673384;;;; 0.07209249 -0.11482985 … -0.03032194 -0.078399494; 0.042712264 -0.017164867 … -0.04245942 -0.08434749; … ; -0.053068373 -0.036270924 … -0.01190234 0.04457287; 0.09645407 0.06579522 … 0.08758384 -0.07599504;;;; … ;;;; -0.063273504 -0.044768434 … 0.068598226 0.008047517; -0.04085146 -0.10570993 … 0.08708025 0.017031923; … ; -0.017329399 -0.08272421 … -0.056110084 0.08124348; -0.043610003 -0.08443125 … -0.019943524 -0.112279095;;;; -0.023709128 0.027822658 … 0.024176616 0.02918409; -0.11875814 0.11234228 … 0.01227252 -0.021421798; … ; 0.06870384 -0.106364034 … 0.007173403 0.06393765; 0.040792737 -0.057107314 … -0.03841498 -0.10485477;;;; -0.06500159 -0.022357002 … -0.094185725 -0.06541265; -0.11485138 -0.031750668 … -0.038430534 -0.09678374; … ; -0.09747197 0.04425735 … 0.045868594 0.058573604; 0.08482695 0.030379916 … 0.08272931 -0.10676909]
      p: Float32 0.0f0
      q: Float32 0.1f0
      kernel: Int64 5
      stride: Int64 2
      padding: Int64 1
      out_width: Int64 13
      out_height: Int64 13
      out_chan: Int64 16





```julia
l1(data)
```


    MethodError: objects of type SharpenedConvolution are not callable

    

    Stacktrace:

     [1] top-level scope

       @ In[75]:1

     [2] eval

       @ ./boot.jl:373 [inlined]

     [3] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

       @ Base ./loading.jl:1196



```julia
using ProfileView
```

    Gtk-Message: 12:10:02.643: Failed to load module "xapp-gtk3-module"
    Gtk-Message: 12:10:02.643: Failed to load module "canberra-gtk-module"



```julia
@code_warntype l1(data) 

```


```julia
@profview l1(data)
```




    Gtk.GtkWindowLeaf(name="", parent, width-request=-1, height-request=-1, visible=TRUE, sensitive=TRUE, app-paintable=FALSE, can-focus=FALSE, has-focus=FALSE, is-focus=FALSE, focus-on-click=TRUE, can-default=FALSE, has-default=FALSE, receives-default=FALSE, composite-child=FALSE, style, events=0, no-show-all=FALSE, has-tooltip=FALSE, tooltip-markup=NULL, tooltip-text=NULL, window, opacity=1.000000, double-buffered, halign=GTK_ALIGN_FILL, valign=GTK_ALIGN_FILL, margin-left, margin-right, margin-start=0, margin-end=0, margin-top=0, margin-bottom=0, margin=0, hexpand=FALSE, vexpand=FALSE, hexpand-set=FALSE, vexpand-set=FALSE, expand=FALSE, scale-factor=1, border-width=0, resize-mode, child, type=GTK_WINDOW_TOPLEVEL, title="Profile", role=NULL, resizable=TRUE, modal=FALSE, window-position=GTK_WIN_POS_NONE, default-width=800, default-height=600, destroy-with-parent=FALSE, hide-titlebar-when-maximized=FALSE, icon, icon-name=NULL, screen, type-hint=GDK_WINDOW_TYPE_HINT_NORMAL, skip-taskbar-hint=FALSE, skip-pager-hint=FALSE, urgency-hint=FALSE, accept-focus=TRUE, focus-on-map=TRUE, decorated=TRUE, deletable=TRUE, gravity=GDK_GRAVITY_NORTH_WEST, transient-for, attached-to, has-resize-grip, resize-grip-visible, application, is-active=FALSE, has-toplevel-focus=FALSE, startup-id, mnemonics-visible=FALSE, focus-visible=FALSE, is-maximized=FALSE)




```julia
@btime Flux.Losses.logitcrossentropy(sc(data), label)
```


    UndefVarError: sc not defined

    

    Stacktrace:

      [1] var"##core#332"()

        @ Main ~/.julia/packages/BenchmarkTools/7xSXH/src/execution.jl:489

      [2] var"##sample#333"(::Tuple{}, __params::BenchmarkTools.Parameters)

        @ Main ~/.julia/packages/BenchmarkTools/7xSXH/src/execution.jl:495

      [3] _run(b::BenchmarkTools.Benchmark, p::BenchmarkTools.Parameters; verbose::Bool, pad::String, kwargs::Base.Pairs{Symbol, Integer, NTuple{4, Symbol}, NamedTuple{(:samples, :evals, :gctrial, :gcsample), Tuple{Int64, Int64, Bool, Bool}}})

        @ BenchmarkTools ~/.julia/packages/BenchmarkTools/7xSXH/src/execution.jl:99

      [4] #invokelatest#2

        @ ./essentials.jl:718 [inlined]

      [5] #run_result#45

        @ ~/.julia/packages/BenchmarkTools/7xSXH/src/execution.jl:34 [inlined]

      [6] run(b::BenchmarkTools.Benchmark, p::BenchmarkTools.Parameters; progressid::Nothing, nleaves::Float64, ndone::Float64, kwargs::Base.Pairs{Symbol, Integer, NTuple{5, Symbol}, NamedTuple{(:verbose, :samples, :evals, :gctrial, :gcsample), Tuple{Bool, Int64, Int64, Bool, Bool}}})

        @ BenchmarkTools ~/.julia/packages/BenchmarkTools/7xSXH/src/execution.jl:117

      [7] #warmup#54

        @ ~/.julia/packages/BenchmarkTools/7xSXH/src/execution.jl:169 [inlined]

      [8] warmup(item::BenchmarkTools.Benchmark)

        @ BenchmarkTools ~/.julia/packages/BenchmarkTools/7xSXH/src/execution.jl:169

      [9] top-level scope

        @ ~/.julia/packages/BenchmarkTools/7xSXH/src/execution.jl:575

     [10] eval

        @ ./boot.jl:373 [inlined]

     [11] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

        @ Base ./loading.jl:1196



```julia
gradient(() -> Flux.Losses.logitcrossentropy(sc(data), label), ps)
```


    UndefVarError: ps not defined

    

    Stacktrace:

     [1] top-level scope

       @ In[23]:1

     [2] eval

       @ ./boot.jl:373 [inlined]

     [3] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

       @ Base ./loading.jl:1196



```julia
# Work in progress.

## Training
# @showprogress for epoch in 1:model_params.n_epochs
@showprogress for epoch in 1:2
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

```


    UndefVarError: ps not defined

    

    Stacktrace:

     [1] macro expansion

       @ In[24]:11 [inlined]

     [2] top-level scope

       @ ~/.julia/packages/ProgressMeter/Vf8un/src/ProgressMeter.jl:940

     [3] eval

       @ ./boot.jl:373 [inlined]

     [4] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

       @ Base ./loading.jl:1196

