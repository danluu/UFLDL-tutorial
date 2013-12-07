function read_float_vector (fname::String)
    f = open(fname)
    lines = readlines(f)
    floats = [x |> strip |> float for x in lines]
    convert(Array{Float64,1}, floats)
end

x = read_float_vector("../ex2x.dat")
y = read_float_vector("../ex2y.dat")


