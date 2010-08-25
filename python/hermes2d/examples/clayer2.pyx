from hermes2d._hermes2d cimport scalar, FuncReal, GeomReal, WeakForm, \
        int_grad_u_grad_v, int_v, malloc, ExtDataReal, c_Ord, create_Ord, \
        FuncOrd, GeomOrd, ExtDataOrd, H1Space, BC_ESSENTIAL, BC_NATURAL, c_BCType, H2D_SYM, \
        c_atan, c_pi, c_exp, c_sqrt, c_sqr, int_F_v, int_grad_u_grad_v_ord
from hermes2d._hermes2d cimport int_F_v_ord


# Problem parameters.
cdef double K = 100                       # Slope of the layer.

# Boundary condition types.
cdef c_BCType bc_type(int marker):
    return <c_BCType>BC_ESSENTIAL
    
# Essential (Dirichlet) boundary condition values.
cdef scalar essential_bc_values(int ess_bdy_marker, double x, double y):
    return 0

# Exact solution to the 1D problem -u'' + K*K*u = K*K in (-1,1) with zero Dirichlet BC.
cdef double uhat(double x):
    return 1. - (c_exp(K*x) + c_exp(-K*x)) / (c_exp(K) + c_exp(-K))

cdef double duhat_dx(double x):
    return -K * (c_exp(K*x) - c_exp(-K*x)) / (c_exp(K) + c_exp(-K))

cdef double dduhat_dxx(double x):
    return -K*K * (c_exp(K*x) + c_exp(-K*x)) / (c_exp(K) + c_exp(-K))

# Exact solution u(x,y) to the 2D problem is defined as the
# Cartesian product of the 1D solutions.
cdef double sol_exact(double x, double y, double& dx, double& dy):
    dx = duhat_dx(x) * uhat(y)
    dy = uhat(x) * duhat_dx(y)
    return uhat(x) * uhat(y)

# Right-hand side.
cdef double rhs(double x, double y):
    return -(dduhat_dxx(x)*uhat(y) + uhat(x)*dduhat_dxx(y)) + K*K*uhat(x)*uhat(y)

cdef scalar bilinear_form(int n, double *wt, FuncReal **t, FuncReal *u, FuncReal *v, GeomReal
        *e, ExtDataReal *ext):
    return int_grad_u_grad_v(n, wt, u, v)

cdef scalar linear_form(int n, double *wt, FuncReal **t, FuncReal *u, GeomReal
        *e, ExtDataReal *ext):
    return -int_F_v(n, wt, rhs, u, e)

cdef c_Ord linear_form_ord(int n, double *wt, FuncOrd **t, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return create_Ord(24)
    
cdef c_Ord _order_bf(int n, double *wt, FuncOrd **t, FuncOrd *u, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return int_grad_u_grad_v_ord(n, wt, u, v)

cdef c_Ord _order_lf(int n, double *wt, FuncOrd **t, FuncOrd *u, GeomOrd
            *e, ExtDataOrd *ext):
    # this doesn't work, unless we redefine rhs using Ord:
    #return int_F_v_ord(n, wt, rhs, u, e).mul_double(-1)
    return create_Ord(20)

def set_forms(WeakForm dp):
    dp.thisptr.add_matrix_form(0, 0, &bilinear_form, &_order_bf, H2D_SYM)
    dp.thisptr.add_vector_form(0, &linear_form, &linear_form_ord)

def set_bc(H1Space space):
    space.thisptr.set_bc_types(&bc_type)
    space.thisptr.set_essential_bc_values(&essential_bc_values)
