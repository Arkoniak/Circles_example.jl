using BenchmarkTools
include("circles1.jl")

img = load("test.jpg");
img = scale(img, 1e6);
seg = segment(img);

min_radius = 2
h, w = size(img)
m = findall(x -> x == 5, seg.image_indexmap)
left, top, mask = make_mask(m, h, w)
circles = Circle[]

# @time calc_label!(circles, mask, min_radius, left, top, img)
# m = 1:
# 22.657704 seconds (20.90 k allocations: 10.143 GiB, 0.56% gc time)

@benchmark calc_label!(circles2, mask2, $min_radius, $left, $top, $img) setup=(mask2 = deepcopy($mask); circles2 = deepcopy($circles)) evals=1
# BenchmarkTools.Trial:
#   memory estimate:  647.05 KiB
#   allocs estimate:  224
#   --------------
#   minimum time:     851.187 μs (0.00% GC)
#   median time:      905.883 μs (0.00% GC)
#   mean time:        952.213 μs (1.78% GC)
#   maximum time:     5.956 ms (0.00% GC)
#   --------------
#   samples:          5240
#   evals/sample:     1

########################################
# Mask simplification
########################################
include("circles2.jl")

img = load("test.jpg");
img = scale(img, 1e6);
seg = segment(img);

min_radius = 2
h, w = size(img)
m = findall(x -> x == 5, seg.image_indexmap)
left, top, mask = make_mask(m, h, w)
circles = Circle[]

# @time calc_label!(circles, mask, min_radius, left, top, img)
# m = 1: 
# 21.367136 seconds (13.88 k allocations: 10.132 GiB, 0.60% gc time)

@benchmark calc_label!(circles2, mask2, $min_radius, $left, $top, $img) setup=(mask2 = deepcopy($mask); circles2 = deepcopy($circles)) evals=1
# BenchmarkTools.Trial:
#   memory estimate:  631.52 KiB
#   allocs estimate:  142
#   --------------
#   minimum time:     799.053 μs (0.00% GC)
#   median time:      858.704 μs (0.00% GC)
#   mean time:        912.187 μs (1.89% GC)
#   maximum time:     6.035 ms (0.00% GC)
#   --------------
#   samples:          5473
#   evals/sample:     1

# (851 - 799)/851 ≈ 6%

########################################
# Radius calculation simplification 
########################################

include("circles3.jl")

img = load("test.jpg");
img = scale(img, 1e6);
seg = segment(img);

min_radius = 2
h, w = size(img)
m = findall(x -> x == 5, seg.image_indexmap)
left, top, mask = make_mask(m, h, w)
circles = Circle[]

# @time calc_label!(circles, mask, min_radius, left, top, img)
# m = 1:
# 20.130693 seconds (12.03 k allocations: 6.774 GiB, 0.44% gc time)

@benchmark calc_label!(circles2, mask2, $min_radius, $left, $top, $img) setup=(mask2 = deepcopy($mask); circles2 = deepcopy($circles)) evals=1
# BenchmarkTools.Trial:
#   memory estimate:  439.89 KiB
#   allocs estimate:  128
#   --------------
#   minimum time:     737.991 μs (0.00% GC)
#   median time:      785.722 μs (0.00% GC)
#   mean time:        823.667 μs (1.47% GC)
#   maximum time:     4.761 ms (0.00% GC)
#   --------------
#   samples:          6062
#   evals/sample:     1

# (851 - 738)/851 ≈ 13%

########################################
# Feature transformation improvement
########################################

include("circles4.jl")

img = load("test.jpg");
img = scale(img, 1e6);
seg = segment(img);

min_radius = 2
h, w = size(img)
m = findall(x -> x == 1, seg.image_indexmap)
left, top, mask = make_mask(m, h, w)
circles = Circle[]
# @time calc_label!(circles, mask, min_radius, left, top, img)
# m = 1:
# 18.979654 seconds (945 allocations: 7.571 MiB)

@benchmark calc_label!(circles2, mask2, $min_radius, $left, $top, $img) setup=(mask2 = deepcopy($mask); circles2 = deepcopy($circles)) evals=1
# BenchmarkTools.Trial:
#   memory estimate:  32.42 KiB
#   allocs estimate:  24
#   --------------
#   minimum time:     696.389 μs (0.00% GC)
#   median time:      739.001 μs (0.00% GC)
#   mean time:        773.319 μs (0.07% GC)
#   maximum time:     4.537 ms (0.00% GC)
#   --------------
#   samples:          6450
#   evals/sample:     1

# (851 - 696)/851 ≈ 18%

########################################
# Overall results
########################################

# ➜  circles julia --project=. circles1.jl
# No args supplied, using defaults: Dict{String, Any}("min_radius" => 2, "image" => "test.jpg")
# Loading test.jpg ...
#   1.429699 seconds (1.94 M allocations: 141.943 MiB, 2.51% gc time, 63.19% compilation time)
# Resizing ...
#   0.433025 seconds (1.31 M allocations: 90.066 MiB, 6.20% gc time, 94.54% compilation time)
# Segmenting ...
#   1.233071 seconds (2.68 M allocations: 247.583 MiB, 2.17% gc time, 35.45% compilation time)
# Circling ...
#  88.037141 seconds (3.50 M allocations: 41.992 GiB, 0.98% gc time, 1.50% compilation time)
# Made 8517 circles
# Drawing ...
#   0.690908 seconds (174.99 k allocations: 12.977 MiB, 0.58% gc time, 38.12% compilation time)


# ➜  circles julia --project=. circles4.jl
# No args supplied, using defaults: Dict{String, Any}("min_radius" => 2, "image" => "test.jpg")
# Loading test.jpg ...
#   1.109887 seconds (1.94 M allocations: 141.757 MiB, 4.24% gc time, 79.77% compilation time)
# Resizing ...
#   0.407106 seconds (1.31 M allocations: 90.177 MiB, 3.77% gc time, 93.59% compilation time)
# Segmenting ...
#   1.191467 seconds (2.67 M allocations: 247.250 MiB, 2.81% gc time, 35.59% compilation time)
# Circling ...
#  73.818242 seconds (3.21 M allocations: 288.362 MiB, 0.28% gc time, 1.41% compilation time)
# Made 8517 circles
# Drawing ...
#   0.463433 seconds (174.99 k allocations: 12.977 MiB, 1.33% gc time, 56.77% compilation time)

