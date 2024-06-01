using LinearAlgebra

function form_J(V, ang, Y, ang_red, volt_red)
    # complex V
    V_CPLX = V .* exp.(im * ang)

    # complex power
    S = V_CPLX .* conj.(Y) * conj.(V_CPLX)

    # sparse matrix
    S_diag = spdiagm(S)
    V_CPLX_diag = spdiagm(V_CPLX)
    Vm_diag = spdiagm(abs.(V))

    SS = V_CPLX_diag * conj.(Y) * conj.(V_CPLX_diag)

    S1 = ((S_diag + SS) / Vm_diag) * volt_red'
    S2 = (S_diag - SS) * ang_red'

    # form 4 parts of Jacobian
    J11 = -ang_red * imag.(S2)
    J12 = ang_red * real.(S1)
    J21 = volt_red * real.(S2)
    J22 = volt_red * imag.(S1)

    # build the full J
    J = [J11 J12; J21 J22]
    return J, S, V_CPLX_diag, S_diag, J11, J12, J21, J22
end
