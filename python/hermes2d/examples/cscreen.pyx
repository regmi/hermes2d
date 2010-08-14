from hermes2d._hermes2d cimport scalar, FuncReal, GeomReal, WeakForm, \
        int_grad_u_grad_v, int_grad_u_grad_v_ord, int_v, int_v_ord, malloc, ExtDataReal, c_Ord, create_Ord, \
        FuncOrd, GeomOrd, ExtDataOrd, int_u_v, int_u_v_ord, BC_NATURAL, H2D_SYM, BC_ESSENTIAL, c_BCType, H1Space, \
        c_atan, c_pi, c_sqrt, c_sqr, c_atan2, c_cos, double2, cplx

# Problem parameters.
cdef double e_0  = 8.8541878176 * 1e-12
cdef double mu_0 = 1.256 * 1e-6
cdef double k = 1.0

# Boundary condition types
cdef c_BCType bc_type_screen(int marker):
    return <c_BCType>BC_ESSENTIAL
    
# Unit tangential vectors to the boundary. 
cdef double2 tau[5]
tau[0][0] = 0
tau[0][1] = 0
tau[1][0] = 1
tau[1][1] = 0
tau[2][0] = 0
tau[2][1] = 1
tau[3][0] = -1
tau[3][1] = 0
tau[4][0] = 0
tau[4][1] = -1

# Fresnel integral
cdef extern void fresnl(double xxa, double *ssa, double *cca )

cdef scalar Fder(double u):
    cdef cplx abc(0.0, c_pi/4)
    cdef scalar a = abc
    cdef scalar b = cplx(0.0, u*u)
    cdef scalar d = cplx(0.0, 2.0*u)
    cdef double s, c
    fresnl(sqrt(2/c_pi) * u, &s , &c)
    cdef scalar fres = cplx(c,-s)
    cdef scalar fresder = exp(-b)
    
    return 0.5*sqrt(M_PI) * exp(b) * ( d * (exp(-a) - sqrt(2.0)*(fres)) - sqrt(2.0)*fresder*sqrt(2.0/M_PI) )
  
cdef scalar der_Hr(double x, double y):
    cdef double r = sqrt(x*x + y*y)
    cdef double t = c_atan2(y,x)
    cdef scalar a = cplx(0.0, c_pi/4 - k*r)
    cdef scalar i = cplx(0.0,1.0)
    return 1/c_sqrt(c_pi) * exp(a) * ( (-i*k)*(Fn(c_sqrt(2*k*r)*c_sin(t/2 - c_pi/8)) + Fn(c_sqrt(2*k*r)*c_sin(t/2 + c_pi/8))) +
        (Fder(c_sqrt(2*k*r)*c_sin(t/2 - c_pi/8))*(c_sqrt(k)/c_sqrt(2*r)*c_sin(t/2 - c_pi/8)) +
         Fder(c_sqrt(2*k*r)*c_sin(t/2 + c_pi/8))*(c_sqrt(k)/c_sqrt(2*r)*c_sin(t/2 + c_pi/8))))

cdef scalar der_Ht(double x, double y):
    cdef double r = c_sqrt(x*x + y*y)
    cdef double t = c_atan2(y,x)
    cdef scalar a = cplx(0.0, c_pi/4 - k*r)
    return 1/c_sqrt(c_pi) * exp(a) * (Fder(c_sqrt(2*k*r)*c_sin(t/2 - c_pi/8))*(c_sqrt(k*r/2)*c_cos(t/2 - c_pi/8)) + Fder(c_sqrt(2*k*r)*c_sin(t/2 + c_pi/8))*(c_sqrt(k*r/2)*c_cos(t/2 + c_pi/8)))

cdef scalar exact0(double x, double y, scalar& dx, scalar& dy):
    cdef double r = c_sqrt(x*x + y*y)
    cdef double theta = c_atan2(y,x)
    cdef scalar Hr = der_Hr(x,y)
    cdef scalar Ht = der_Ht(x,y)
    cdef scalar i = cplx(0.0,1.0)
    return  -i * (Hr * y/r + Ht * x/(r*r))
  
# Essential (Dirichlet) boundary condition values.
cdef scalar essential_bc_values(int ess_bdy_marker, double x, double y):
    cdef scalar dx, dy
    return exact0(x, y, dx, dy)*tau[ess_bdy_marker][0] + exact1(x, y, dx, dy)*tau[ess_bdy_marker][1]

cdef scalar bilinear_form(int n, double *wt, FuncReal **t, FuncReal *u, FuncReal *v, GeomReal
        *e, ExtDataReal *ext):
    return int_curl_e_curl_f(n, wt, u, v) - int_e_f(n, wt, u, v)
    
cdef c_Ord _order_bf(int n, double *wt, FuncOrd **t, FuncOrd *u, FuncOrd *v, GeomOrd
        *e, ExtDataOrd *ext):
    return int_grad_u_grad_v_ord(n, wt, u, v)

def set_forms(WeakForm dp):
    dp.thisptr.add_matrix_form(0, &bilinear_form, &_order_bf, H2D_SYM)

def set_bc(H1Space space):
    space.thisptr.set_bc_types(&bc_type_screen)
    space.thisptr.set_essential_bc_values(&essential_bc_values)
