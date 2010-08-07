from hermes2d._hermes2d cimport scalar, FuncReal, GeomReal, WeakForm, \
        int_grad_u_grad_v, int_grad_u_grad_v_ord, int_v, int_v_ord, int_F_v, malloc, ExtDataReal, c_Ord, create_Ord, \
        FuncOrd, GeomOrd, ExtDataOrd, int_u_v, int_u_v_ord, BC_NATURAL, H2D_SYM, BC_ESSENTIAL, c_BCType, H1Space, \
        c_atan, c_pi, c_sqrt, c_sqr, c_atan2, c_cos, c_sin
        
# Equation parameters.
cdef double K = c_pi/2
cdef double ALPHA = 2.01

cdef double fn(double x, double y):
    if (x <= 0):
        return c_cos(K*y)
    else:
        return c_cos(K*y) + (x ** ALPHA)

cdef double fndd(double x, double y, double& dx, double& dy):
  if (x <= 0):
      dx = 0
  else:
      dx = ALPHA * (x ** (ALPHA - 1))
  dy = -c_sin(K*y)*K
  return fn(x, y)

cdef scalar rhs(scalar x, scalar y):
    if (x < 0):
        return fn(x, y)*K*K
    else:
        return fn(x, y)*K*K-ALPHA*(ALPHA-1)*(x ** (ALPHA - 2.0)) - K*K*(x ** ALPHA)

# Boundary condition types
cdef c_BCType bc_type_ls(int marker):
    if (marker == 1):
        return <c_BCType>BC_ESSENTIAL
    else:
        return <c_BCType>BC_NATURAL

# Essential (Dirichlet) boundary condition values.
cdef scalar essential_bc_values(int ess_bdy_marker, double x, double y):
    return fn(x, y)

cdef scalar bilinear_form(int n, double *wt, FuncReal **t, FuncReal *u, FuncReal *v, GeomReal
        *e, ExtDataReal *ext):
    return int_grad_u_grad_v(n, wt, u, v)

cdef scalar linear_form(int n, double *wt, FuncReal **t, FuncReal *v,
        GeomReal *e, ExtDataReal *ext):
    return int_F_v(n, wt, rhs, v, e)
    
cdef c_Ord _order_bf(int n, double *wt, FuncOrd **t, FuncOrd *u, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return int_grad_u_grad_v_ord(n, wt, u, v)
    
cdef c_Ord _linear_form_ord(int n, double *wt, FuncOrd **t, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return create_Ord(30)

def set_forms(WeakForm dp):
    dp.thisptr.add_matrix_form(0, 0, &bilinear_form, &_order_bf, H2D_SYM)
    dp.thisptr.add_vector_form(0, &linear_form, &_linear_form_ord)

def set_bc(H1Space space):
    space.thisptr.set_bc_types(&bc_type_ls)
    space.thisptr.set_essential_bc_values(&essential_bc_values)

