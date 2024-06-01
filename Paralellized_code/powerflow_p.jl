using MPI
using SparseArrays
using Printf

# Initialize MPI environment
MPI.Init()

# Get the total number of processes and rank of the current process
comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
myid = MPI.Comm_rank(comm)

# Load file (manual change)
if myid == 0
    include("claims.jl")
    include("form_J.jl")
    include("makeYbus.jl")
    include("dPW5bus.jl")
    #include("d1248.jl")
    #include("data2a.jl")
    #include("data48em.jl")
    #include("d19968.jl")
    #include("d39936.jl")
end

MPI.Barrier(comm)

# Broadcast data to all processes
nbus, nline, bus, line = 0, 0, nothing, nothing
if myid == 0
    nbus = size(bus, 1)
    nline = size(line, 1)
    bus_line = hcat(bus, line)
end

# Define `counts` and `displacements` for `Scatterv` and `Gatherv`
counts = [div(nline, nprocs) for i in 1:nprocs]
counts[1:mod(nline, nprocs)] += 1
displacements = [sum(counts[1:i-1]) for i in 1:nprocs]

nbus, nline, bus_line = MPI.Bcast([nbus, nline, bus_line], root=0)

bus = bus_line[:, 1:10]
line = bus_line[:, 11:end]

# ext2int bus indices conversion 
i2e = bus[:, 1]
e2i = sparsevec(i2e, 1:nbus)
bus_int = e2i[bus[:, 1]]
line_int = e2i[line[:, 1]]
line_int_2 = e2i[line[:, 2]]

bus_new = hcat(bus_int, bus[:, 2:end])
line_new = hcat(line_int, line_int_2, line[:, 3:end])

# Calculate Y bus matrix
Y = makeYbus(bus_new, line_new)

# Read bus data
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

# Set up non_ref_array
ref = findall(bus_type .== 1);
PQV = findall(bus_type .>= 2);
PQ  = findall(bus_type .== 3);
PV  = findall(bus_type .== 2);

noref_id = ones(nbus);
noref_id[ref] .= 0;

# Set up load array
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

# Assign some variable values
iter = 0;
flag = 1;
tol = 1e-9;
iter_max = 30;

# Calculate power mismatch
(P, Q, dP, dQ, conv_flag, mis) = claims(V, ang, Pg, Qg, Pl, Ql, Y, noref_id, nogen_id, tol)

while conv_flag == 1 && iter < iter_max
    global iter += 1
    # form the Jacobian matrix
    # Initial values read from original system; will add flat start later;
    (J, S, V_CPLX_diag, S_diag, J11, J12, J21, J22) = form_J(V, ang, Y, ang_red, volt_red)

    # reduced real and reactive power mismatch vectors
    dP_red = ang_red * dP
    dQ_red = volt_red * dQ

    # Scatter the data
    scatterv_dP = MPI.Scatterv(dP_red, counts, displacements, MPI.DOUBLE_PRECISION, root=0)
    scatterv_dQ = MPI.Scatterv(dQ_red, counts, displacements, MPI.DOUBLE_PRECISION, root=0)

    # Calculate the local solution
    local_sol = J \ [scatterv_dP; scatterv_dQ]

    # Gather the solutions
    gather_sol = MPI.Gatherv(local_sol, counts, displacements, MPI.DOUBLE_PRECISION, root=0)

    if myid == 0
        # find angle elements
        dang = ang_red' * gather_sol[1:length(PQV), end]

        # find voltage elements
        dV = gather_sol[length(PQV)+1:length(PQV)+length(PQ),end]' * volt_red;

        # update V and ang 
        V .+= dV'
        ang .+= dang
        # calculate the power mismatch
        (P, Q, dP, dQ, conv_flag, mis) = claims(V, ang, Pg, Qg, Pl, Ql, Y, noref_id, nogen_id, tol)

        # update Qg
        Qg[PV] = Q[PV] + Ql[PV]

        # check Qglimit; will add PV->PQ conversion later.
        if any(Qg .> qg_max)
            Qg[Qg .> qg_max] = qg_max[Qg .> qg_max]
        elseif any(Qg .< qg_min)
            Qg[Qg .< qg_min] = qg_min[Qg .< qg_min]
        end
    end

    # Broadcast new values to all processes
    V = MPI.Bcast(V, root=0)
    ang = MPI.Bcast(ang, root=0)
    Qg = MPI.Bcast(Qg, root=0)

    # Calculate power mismatch
    (P, Q, dP, dQ, conv_flag, mis) = calmis(V, ang, Pg, Qg, Pl, Ql, Y, noref_id, nogen_id, tol)

    # Reduce mis to root process
    MPI.Reduce(mis, mis_tot, op=MPI.SUM, root=0)

    # Check for convergence
    if myid == 0
        # check for convergence
        if abs(mis_tot) < tol
            flag = 0;
        end
    end

    # Broadcast flag to all processes
    flag = MPI.Bcast(flag, root=0)
end

# Finalize MPI environment
MPI.Finalize()
