function calmis(V, ang, Pg, Qg, Pl, Ql, Y, noref_id, nogen_id, tol)
    # Complex bus voltage
    V_CPLX = V .* exp.(im * ang)
    
    # Bus current injection
    I_inj = Y * V_CPLX
    
    # Power outputs
    S = V_CPLX .* conj.(I_inj)
    P = real.(S)
    Q = imag.(S)
    
    dP = Pg .- Pl .- P
    dQ = Qg .- Ql .- Q
    dP = dP .* noref_id # non-ref
    dQ = dQ .* noref_id # non-ref
    dQ = dQ .* nogen_id # load
    mis = maximum(abs.([dP; dQ]))
    
    if mis > tol
        conv_flag = 1
    else
        conv_flag = 0
    end
    
    return P, Q, dP, dQ, conv_flag, mis
end
