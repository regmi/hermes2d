from hermes2d._hermes2d cimport scalar, FuncReal, GeomReal, WeakForm, \
        int_curl_e_curl_f, int_e_f, int_grad_u_grad_v_ord, int_v, int_v_ord, malloc, ExtDataReal, c_Ord, create_Ord, \
        FuncOrd, GeomOrd, ExtDataOrd, int_u_v, int_u_v_ord, BC_NATURAL, BC_ESSENTIAL, c_BCType, H1Space, cplx

# Problem parameters
cdef double mu_r   = 1.0
cdef double kappa  = 1.0
cdef double lamda = 1.0

# Boundary condition types
cdef c_BCType bc_type_bessel(int marker):
    if marker == 1 or marker == 6:
        return <c_BCType>BC_ESSENTIAL    # perfect conductor
    else:
        return <c_BCType>BC_NATURAL      # impedance

# Essential (Dirichlet) boundary condition values.
cdef scalar essential_bc_values(int ess_bdy_marker, double x, double y):
    return 0

cdef scalar bilinear_form(int n, double *wt, FuncReal **t, FuncReal *u, FuncReal *v, GeomReal
        *e, ExtDataReal *ext):
    return 1.0/mu_r * int_curl_e_curl_f(n, wt, u, v) - sqr(kappa) * int_e_f(n, wt, u, v)
    
cdef scalar bilinear_form_surf_bessel(int n, double *wt, FuncReal **t, FuncReal *u, FuncReal *v,
        GeomReal *e, ExtDataReal *ext):
    cplx ii = cplx(0.0, 1.0)
    return ii * (-kappa) * int_e_tau_f_tau(n, wt, u, v, e)

cdef scalar linear_form_surf_bessel(int n, double *wt, FuncReal **t, FuncReal *v, GeomReal *e,
        ExtDataReal *ext):
    scalar result = 0
    for i in range(n):
        cdef double r = c_sqrt(e->x[i] * e->x[i] + e->y[i] * e->y[i])
        cdef double theta = c_atan2(e->y[i], e->x[i])
        if (theta < 0):
            theta += 2.0*c_pi
        cdef double j13    = jv(-1.0/3.0, r),      j23    = jv(+2.0/3.0, r)
        cdef double cost   = c_cos(theta),         sint   = c_sin(theta)
        cdef double cos23t = c_cos(2.0/3.0*theta), sin23t = c_sin(2.0/3.0*theta)

        cdef double Etau = e->tx[i] * (cos23t*sint*j13 - 2.0/(3.0*r)*j23*(cos23t*sint + sin23t*cost)) +
                      e->ty[i] * (-cos23t*cost*j13 + 2.0/(3.0*r)*j23*(cos23t*cost - sin23t*sint))

        result += wt[i] * cplx(cos23t*j23, -Etau) * ((v->val0[i] * e->tx[i] + v->val1[i] * e->ty[i]))
    return result

# Maximum polynomial order to integrate surface linear form.
cdef c_Ord _linear_form_surf_ord(int n, double *wt, Func<Ord> *u_ext[], Func<Ord> *v, Geom<Ord> *e, ExtData<Ord> *ext):
    return Ord(v->val[0].get_max_order())
  
cdef c_Ord _order_bf(int n, double *wt, FuncOrd **t, FuncOrd *u, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return int_grad_u_grad_v_ord(n, wt, u, v)

cdef c_Ord _order_bf_surf_bessel(int n, double *wt, FuncOrd **t, FuncOrd *u, FuncOrd *v,
        GeomOrd *e, ExtDataOrd *ext):
    return int_u_v_ord(n, wt, u, v)

cdef c_Ord _order_lf(int n, double *wt, FuncOrd **t, FuncOrd *u, GeomOrd
        *e, ExtDataOrd *ext):
    return int_v_ord(n, wt, u)

def set_forms(WeakForm dp):
    dp.thisptr.add_matrix_form(0, &bilinear_form, &_order_bf, H2D_SYM)
    dp.thisptr.add_matrix_form_surf(0, &bilinear_form_surf_bessel, &_order_bf_surf_bessel)
    dp.thisptr.add_vector_form_surf(0, &linear_form_surf_bessel, &_linear_form_surf_ord)

def set_bc(H1Space space):
    space.thisptr.set_bc_types(&bc_type_06)
    space.thisptr.set_essential_bc_values(&essential_bc_values)

