from hermes2d._hermes2d cimport scalar, FuncReal, GeomReal, WeakForm, \
        int_grad_u_grad_v, int_grad_u_grad_v_ord, int_v, int_v_ord, malloc, ExtDataReal, c_Ord, create_Ord, \
        FuncOrd, GeomOrd, ExtDataOrd, int_u_v, int_u_v_ord, BC_NATURAL, H2D_SYM, BC_ESSENTIAL, c_BCType, H1Space, \
        c_atan, c_pi, c_sqrt, c_sqr, c_atan2, c_cos

# Problem parameters.
cdef double e_0  = 8.8541878176 * 1e-12
cdef double mu_0 = 1.256 * 1e-6
cdef double k = 1.0

# Boundary condition types
cdef c_BCType bc_type_screen(int marker):
    return <c_BCType>BC_ESSENTIAL
    
# Unit tangential vectors to the boundary. 
cdef double2 tau[5] = { { 0, 0}, { 1, 0 },  { 0, 1 }, { -1, 0 }, { 0, -1 } }

# Essential (Dirichlet) boundary condition values.
cdef scalar essential_bc_values(int ess_bdy_marker, double x, double y):
    cdef scalar dx, dy
    return exact0(x, y, dx, dy)*tau[ess_bdy_marker][0] + exact1(x, y, dx, dy)*tau[ess_bdy_marker][1]

cdef scalar bilinear_form(int n, double *wt, FuncReal **t, FuncReal *u, FuncReal *v, GeomReal
        *e, ExtDataReal *ext):
    return R * int_grad_u_grad_v(n, wt, u, v)
    
cdef c_Ord _order_bf(int n, double *wt, FuncOrd **t, FuncOrd *u, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return int_grad_u_grad_v_ord(n, wt, u, v)

def set_forms(WeakForm dp):
    dp.thisptr.add_matrix_form(0, 0, &bilinear_form, &_order_bf, H2D_SYM)

def set_bc(H1Space space):
    space.thisptr.set_bc_types(&bc_type_screen)
    space.thisptr.set_essential_bc_values(&essential_bc_values)
