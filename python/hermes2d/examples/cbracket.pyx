from hermes2d._hermes2d cimport scalar, FuncReal, GeomReal, WeakForm, \
        int_grad_u_grad_v, int_grad_u_grad_v_ord, int_v, int_v_ord, malloc, ExtDataReal, c_Ord, create_Ord, \
        FuncOrd, GeomOrd, ExtDataOrd, int_u_v, int_u_v_ord, BC_NATURAL, H2D_SYM, BC_ESSENTIAL, c_BCType, H1Space, \
        c_atan, c_pi, c_sqrt, c_sqr, c_atan2, c_cos
        
# Problem parameters.
cdef double E  = 200e9                 # Young modulus for steel: 200 GPa.
cdef double nu = 0.3                   # Poisson ratio.
cdef double f  = 1e3                  # Load force: 10^3 N.
cdef double lamda = (E * nu) / ((1 + nu) * (1 - 2*nu))
cdef double mu = E / (2*(1 + nu))

# Boundary markers.
cdef int BDY_LEFT = 1
cdef int BDY_TOP = 2

cdef double fn(double x, double y):
    cdef double theta = c_atan2(y,x)
    if theta < 0:
        theta = theta + 2.0 * c_pi
    cdef double r = c_sqrt(x*x + y*y)
    
    cdef double mu
    if theta <= c_pi:
        mu = c_cos(RHO * TAU) * c_cos((theta - c_pi + SIGMA)*TAU)
    else:
        mu = c_cos((c_pi/2. - RHO)*TAU) * c_cos((theta - 3.*c_pi/2. - SIGMA)*TAU)
    return (r**TAU) * mu

# Boundary condition types
cdef c_BCType bc_type_bracket(int marker):
    if (marker == BDY_LEFT):
        return <c_BCType>BC_ESSENTIAL
    else:
        return <c_BCType>BC_NATURAL

# Essential (Dirichlet) boundary condition values.
cdef scalar essential_bc_values(int ess_bdy_marker, double x, double y):
    return 0

cdef scalar bilinear_form_I_III(int n, double *wt, FuncReal **t, FuncReal *u, FuncReal *v, GeomReal
        *e, ExtDataReal *ext):
    return R * int_grad_u_grad_v(n, wt, u, v)

cdef scalar bilinear_form_II_IV(int n, double *wt, FuncReal **t, FuncReal *u, FuncReal *v,
        GeomReal *e, ExtDataReal *ext):
    return 1.0 * int_grad_u_grad_v(n, wt, u, v)
    
cdef c_Ord _order_bf(int n, double *wt, FuncOrd **t, FuncOrd *u, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return int_grad_u_grad_v_ord(n, wt, u, v)

def set_forms(WeakForm dp):
    dp.thisptr.add_matrix_form(0, 0, &bilinear_form_I_III, &_order_bf, H2D_SYM, 0)
    dp.thisptr.add_matrix_form(0, 0, &bilinear_form_II_IV, &_order_bf, H2D_SYM, 1)

def set_bc(H1Space space):
    space.thisptr.set_bc_types(&bc_type_kellogg)
    space.thisptr.set_essential_bc_values(&essential_bc_values)

wf.add_matrix_form(0, 0, callback(bilinear_form_0_0), H2D_SYM);  # note that only one symmetric part is
wf.add_matrix_form(0, 1, callback(bilinear_form_0_1), H2D_SYM);  # added in the case of symmetric bilinear
wf.add_matrix_form(1, 1, callback(bilinear_form_1_1), H2D_SYM);  # forms
wf.add_vector_form_surf(1, callback(linear_form_surf_1), BDY_TOP);
