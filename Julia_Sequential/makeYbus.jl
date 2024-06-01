using LinearAlgebra
function makeYbus(bus, line)
    # load case
    # data2a;
    # PW5bus;
    
    nbus = size(bus, 1)
    nline = size(line, 1)
    
    # external to internal conversion
    bus_int = zeros(Int, Int(maximum(bus[:, 1])))
    bus_i = collect(1:nbus)
    bus_int[map(Int, bus[:,1])] = bus_i
    
    # initialization
    r = zeros(nline)
    x = zeros(nline)
    b = zeros(nline)
    z = zeros(nline)
    y = zeros(nline)
    Y = sparse([1], [1], [0], nbus, nbus)
    
    # read values
    Gb = bus[:, 8]
    Bb = bus[:, 9]
    r = line[:, 3]
    x = line[:, 4]
    b = im * sparse(diagm(0.5 .* line[:,5]))
    from = line[:, 1]
    from_int = bus_int[Int.(from)]
    to = line[:, 2]
    to_int = bus_int[map(Int, to)]
    tap = ones(nline)
    phase_shift = line[:, 7]
    tap_index = findall(line[:, 6] .!= 0)
    tap[tap_index] .= 1.0 ./ line[tap_index, 6]
    tap = tap .* exp.(-im * phase_shift .* pi ./ 180)
    
    # calculate z and y
    z = r + im * x
    y = spdiagm(1 ./ z)
    
    # form sparse matrix structure
    line_i = collect(1:nline)
    C_from = sparse(from_int, line_i, tap, nbus, nline)
    C_to = sparse(to_int, line_i, ones(nline), nbus, nline)
    C_line = C_from - C_to
    
    # Calculate Y
    Y = C_from * b * C_from' + C_to * b * C_to'
    Y = Y + C_line * y * C_line'
    Y = Y + sparse(bus_i, bus_i, Gb + im * Bb, nbus, nbus)
    # for i in 1:size(Y, 1)
    #     for j in 1:size(Y, 2)
    #         if Y[i, j] != 0
    #             Y[i,j] = (Y[i, j]*1.0e-2)
    #             #println("($i,$j)\t$(Y[i, j])")
    #         end
    #     end
    # end
    #println(Y)
    return Y
end
