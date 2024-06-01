using SparseArrays
using Printf

# load file (manual change)
include("dPW5bus.jl")
#include("d1248.jl")
#include("data2a.jl")
#include("data48em.jl")
#include("d19968.jl")
#include("d39936.jl")

include("makeYbus.jl")
include("claims.jl")
include("form_J.jl")

# find size of bus and line
nbus = size(bus, 1)
nline = size(line, 1)

# ext2int bus indices conversion 
i2e = bus[:, 1]
e2i = sparsevec(i2e, 1:nbus)
bus_int = e2i[bus[:, 1]]
line_int = e2i[line[:, 1]]
line_int_2 = e2i[line[:, 2]]

bus_new = hcat(bus_int, bus[:, 2:end])
line_new = hcat(line_int, line_int_2, line[:, 3:end])

Y = makeYbus(bus_new, line_new)

# read bus data
bus_no = bus[:,1];
V = bus[:,2];
ang = bus[:,3] .* pi/180;
Pg = bus[:,4];
Qg = bus[:,5];
Pl = bus[:,6];
Ql = bus[:,7];
Gb = bus[:,8];
Bb = bus[:,9];
bus_type = bus[:,10];
qg_max = bus[:,11];
qg_min = bus[:,12];

# setup non_ref_array
ref = findall(bus_type .== 1);
PQV = findall(bus_type .>= 2);
PQ  = findall(bus_type .== 3);
PV  = findall(bus_type .== 2);

noref_id = ones(nbus);
noref_id[ref] .= 0;

# setup load array
nogen_id = ones(nbus);
nogen_id[PV] .= 0;

# Angle (all buses but ref)
len_of_noref = length(PQV);
len = 1:len_of_noref;
ang_red = sparse(len, PQV, ones(len_of_noref), len_of_noref, nbus);

# Voltage (all load buses)
len_of_load = length(PQ);
len = 1:len_of_load;
volt_red = sparse(len, PQ, ones(len_of_load), len_of_load, nbus);

# assign some variable values
iter = 0;
flag = 1;
tol = 1e-9;
iter_max = 30;

# calculate power mismatch
(P, Q, dP, dQ, conv_flag, mis) = calmis(V, ang, Pg, Qg, Pl, Ql, Y, noref_id, nogen_id, tol)

while conv_flag == 1 && iter < iter_max
    global iter += 1
    # form the Jacobian matrix
    # Initial values read from original system; will add flat start later;
    (J, S, V_CPLX_diag, S_diag, J11, J12, J21, J22) = form_J(V, ang, Y, ang_red, volt_red)
    
    # reduced real and reactive power mismatch vectors
    global dP = dP
    dP_red = ang_red * dP
    
    global dQ = dQ
    dQ_red = volt_red * dQ
    sol = J \ [dP_red; dQ_red] # it is LU by default

    # find angle elements
    dang = ang_red' * sol[1:length(PQV), end]

    # find voltage elements
    dV = sol[length(PQV)+1:length(PQV)+length(PQ),end]' * volt_red;
    # update V and ang 
    V .+= dV'
    ang .+= dang
    # calculate the power mismatch
    (P, Q, dP, dQ, conv_flag, mis) = calmis(V, ang, Pg, Qg, Pl, Ql, Y, noref_id, nogen_id, tol)
    global P = P;
    global Q = Q;
    global conv_flag = conv_flag;
    global mis = mis;
    
    # update Qg
    Qg[PV] = Q[PV] + Ql[PV]
    
    # check Qglimit; will add PV->PQ conversion later.
    if any(Qg .> qg_max)
        println("There are Qg > Qgmax at iter ", iter, " at gen index: ", findall(Qg .> qg_max))
    end
    
    if any(Qg .< qg_min)
        println("There are Qg < Qgmin at iter ", iter, " at gen index: ", findall(Qg .< qg_min))
    end
end
    
Pg[ref] .= P[ref] + Pl[ref] 
Pg[PV]  .= P[PV]  + Pl[PV]
Pl[PQ]  .= Pg[PQ] .- P[PQ]
Qg[ref] .= Q[ref] + Ql[ref]
Qg[PV]  .= Q[PV]  + Ql[PV]
Ql[PQ]  .= Qg[PQ] .- Q[PQ]

# int2ext bus indices conversion 
#bus[:, 1] .= i2e[map(Int, bus[:, 1])]
for i in 1:size(bus, 1)
    idx = Int(bus[i, 1])
    if 1 <= idx <= length(i2e)
        bus[i, 1] = i2e[idx]
    end
end

bus_no = bus[:, 1]
#line[:, 1] .= i2e[map(Int,line[:, 1])]
for i in 1:length(line[:, 1])
    index = Int(line[i, 1])
    if 1 <= index <= length(i2e)
        line[i, 1] = i2e[index]
    end
end

#line[:, 2] .= i2e[map(Int,line[:, 2])]
for i in 1:size(line, 1)
    idx = Int(line[i, 2])
    if 1 <= idx <= length(i2e)
        line[i, 2] = i2e[idx]
    end
end


bus_sol = [bus_no V  ang .* 180 / pi Pg Qg Pl Ql]

if conv_flag == 1
    println("AC load flow diverged")
    #error("stop")
else
    # assuming bus_sol is a matrix with dimensions (n, 7)
    println("                                    GENERATION             LOAD")
    println("    BUS       VOLTS    ANGLE      REAL  REACTIVE      REAL  REACTIVE ")
    for i in 1:size(bus_sol, 1)
        @printf("%7.4f   %7.4f   %7.4f   %7.4f   %7.4f   %7.4f   %7.4f\n", 
            bus_sol[i, 1], bus_sol[i, 2], bus_sol[i, 3], 
            bus_sol[i, 4], bus_sol[i, 5], bus_sol[i, 6], bus_sol[i, 7])
    end

end
