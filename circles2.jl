using ArgParse
using ImageSegmentation
using ImageMorphology
using Images
using Luxor
using Statistics


int_round(f) = round(Int, f)


struct Circle{T}
    x::Int
    y::Int
    rgb::T
    r::Float64
end

# Is there a more normal way to do a keyword constructor for a type?
function Circle(;x::Int, y::Int, rgb::Tuple{Int64, Int64, Int64}, r::Float64)
    return Circle(x, y, rgb, r)
end


function parse_commandline()
    s = ArgParse.ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler

    @ArgParse.add_arg_table s begin
        "--image", "-i"
            help = "Path to input image"
            arg_type = String
            required = true
        "--min_radius", "-r"
            help = "Minimum radius of generated circles"
            arg_type = Float64
            default = 2.0
    end

    return ArgParse.parse_args(s)
end


function scale(img, target_px)
    h, w = size(img)
    # Need sqrt because ratio gets applied in both x and y dimension.
    return Images.imresize(img, ratio=(target_px / (h*w))^.5)
end


function segment(img)
    # k controls segment size. 5 - 500 is a good range
    # min_size gets rid of small segments 
    return ImageSegmentation.felzenszwalb(img, 60, 100)
end


function make_mask(label_ix, imgh, imgw)
    # Create a mask array that's just large enough to contain
    # the segment we are working on. This will allow the distance transform
    # to operate the smallest array possible, for speed.
    pad = Dict(true => 0, false => 1)
    (top, left), (bottom, right) = map(ix -> ix.I, extrema(label_ix))

    # Allocate a pixel-wide boundary strip if we aren't at the image border.
    top_pad, bottom_pad = pad[top == 1], pad[bottom == imgh]
    left_pad, right_pad = pad[left == 1], pad[right == imgw]

    mask = ones(Bool,
        bottom - top + 1 + top_pad + bottom_pad,
        right - left + 1 + left_pad + right_pad
    )

    new_ix = map(x -> x - CartesianIndex(top - top_pad - 1, left - left_pad - 1), label_ix)
    mask[new_ix] .= 0

    return (left - left_pad, top - top_pad, mask)
end


function circle_indices(mask, offset, img, center, radius, height, width)
    # Code inspired by:
    # https://github.com/JuliaImages/ImageDraw.jl/blob/master/src/ellipse2d.jl

    # indices = CartesianIndex{2}[]
    r = CartesianIndex(int_round(radius), int_round(radius))
    y, x = center.I
    # Examine (2*radius + 1)^2 spots, expecting array length ~ pi * radius^2.
    cnt = 0
    colour = (0., 0., 0.)
    @inbounds for ix in center - r: center + r
        row, col = ix.I
        if 1 <= row <= height && 1 <= col <= width
            if (row - y)^2 + (col - x)^2 <= radius^2
                mask[ix] = true
                cnt += 1
                clr = img[ix + offset]
                colour = colour .+ (clr.r, clr.g, clr.b)
            end
        end
    end

    return round.(Int, colour ./ cnt .* 256)
end

float2int(f) = round(Int, 256*f)
to_clr(l) = float2int.( (l.r, l.g, l.b) )

function calc_label!(circles, mask, min_radius, left, top, img)
    offset = CartesianIndex(top - 1, left - 1)
    while true
        f = feature_transform(mask)
        edt = distance_transform(f)
        ix = argmax(edt)
        r = edt[ix]

        if r < min_radius || r == Inf
            return circles
        end

        clr = circle_indices(mask, offset, img, ix, r, size(edt)...)

        y, x = ix.I

        push!(circles, Circle(
                              x=left + x,
                              y=top + y,
                              rgb=clr,
                              r=r
                             ))
    end
end

function make_circles(img, seg, min_radius)

    h, w = size(img)

    # Add a big circle with the avg color for the background.
    circles = [Circle(
        x=int_round(w/2),
        y=int_round(h/2),
        rgb= to_clr(Statistics.mean(img)),
        r=(w*w + h*h)^.5 + 2
    )]

    for label in seg.segment_labels
        m = findall(x -> x == label, seg.image_indexmap)

        left, top, mask = make_mask(m, h, w)
        calc_label!(circles, mask, min_radius, left, top, img)
    end

    return circles
end


function draw(img, circles, filename)
    h, w = size(img)
    Luxor.Drawing(w, h, filename)
    # Paint from biggest to smallest
    for c in sort(circles, by=c -> c.r, rev=true)
        r, g, b = map(x -> x / 255, c.rgb)
        Luxor.setcolor(r, g, b)
        Luxor.circle(c.x, c.y, c.r - .5, :fill)
    end
    Luxor.finish()
end


function main()
    args = nothing
    try
        args = parse_commandline()
        println("args", args)
    catch err
        args = Dict("image" => "test.jpg", "min_radius" => 2)
        println("No args supplied, using defaults: ", args)
    end

    path = args["image"]
    name, ext = splitext(basename(path))


    println("Loading $(path) ...")
    @time img = Images.load(path)

    println("Resizing ...")
    @time img = scale(img, 1e6)  # Downsample to ~1M pixels
    Images.save("./$(name)-orig.png", img)

    println("Segmenting ... ")
    @time seg = segment(img)
    Images.save("./$(name)-seg.png", map(i-> seg.segment_means[i], seg.image_indexmap))

    println("Circling ...")
    @time circles = make_circles(img, seg, args["min_radius"])
    println("Made $(length(circles)) circles")

    println("Drawing ...")
    @time draw(img, circles, "./$(name)-circled.png")
end

if !isinteractive()
    main()
end
