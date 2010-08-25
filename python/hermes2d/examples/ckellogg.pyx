from hermes2d._hermes2d cimport scalar, FuncReal, GeomReal, WeakForm, Mesh, Solution, \
        int_grad_u_grad_v, int_grad_u_grad_v_ord, int_v, int_v_ord, malloc, ExtDataReal, c_Ord, create_Ord, \
        FuncOrd, GeomOrd, ExtDataOrd, int_u_v, int_u_v_ord, BC_NATURAL, H2D_SYM, BC_ESSENTIAL, c_BCType, H1Space, \
        c_atan, c_pi, c_sqrt, c_sqr, c_atan2, c_cos, c_sin
        
# Problem parameters.
cdef double R = 161.4476387975881      
cdef double TAU = 0.1                  
cdef double RHO = c_pi/4.              
cdef double SIGMA = -14.9225651045515E2 

# Exact solution
cdef double fn(double x, double y):
    cdef double theta = c_atan2(y,x)
    if theta < 0:
        theta = theta + 2.0 * c_pi
    cdef double r = c_sqrt(x*x + y*y)
    
    cdef double mu
    if theta <= c_pi/2.0:
        mu = c_cos((c_pi/2.0 * SIGMA) * TAU) * c_cos((theta - c_pi/2.0 + RHO) * TAU)
    else:
        if theta <= c_pi:
            mu = c_cos(RHO * TAU) * c_cos((theta - c_pi - SIGMA) * TAU)
        else:
            if theta <= 3.0*(c_pi/2.0):
                mu = c_cos(SIGMA * TAU) * c_cos((theta - c_pi - RHO) * TAU)
            else:
                mu = c_cos((c_pi/2.0 - RHO) * TAU) * c_cos((theta - 3.0 * c_pi/2.0 - SIGMA) * TAU)

    return (r ** TAU) * mu

cdef double fndd(double x, double y, double& dx, double& dy):
    cdef double theta = c_atan2(y,x)
    if (theta < 0):
        theta = theta + 2*c_pi
    cdef double r = c_sqrt(x*x + y*y)
    # x-derivative
    if (theta <= c_pi/2.):
        dx = TAU*x*(r ** (2.*(-1 + TAU/2.))) * c_cos((c_pi/2. - SIGMA)*TAU) * \
        c_cos(TAU*(-c_pi/2. + RHO + theta)) + \
        (TAU*y*(r ** TAU)*c_cos((c_pi/2. - SIGMA)*TAU) * \
        c_sin(TAU*(-c_pi/2. + RHO + theta))/(r*r))
    else:
        if (theta <= c_pi):
            dx = TAU*x * (r ** (2.*(-1 + TAU/2.))) * c_cos(RHO*TAU) * \
            c_cos(TAU*(-c_pi + SIGMA + theta)) + (TAU*y * (r ** TAU) * \
            c_cos(RHO*TAU) * c_sin(TAU*(-c_pi + SIGMA + theta))/(r*r))
        else:
            if (theta <= 3.*c_pi/2.):
                dx = TAU*x * (r ** (2.*(-1 + TAU/2.))) * c_cos(SIGMA*TAU) * \
                c_cos(TAU*(-c_pi - RHO + theta)) + (TAU*y * (r ** TAU) * \
                c_cos(SIGMA*TAU) * c_sin(TAU*(-c_pi - RHO + theta))/(r*r))
            else:
                dx = TAU*x* (r ** (2*(-1 + TAU/2.))) * \
                c_cos((c_pi/2. - RHO)*TAU) * \
                c_cos(TAU*(-3.*c_pi/2. - SIGMA + theta)) + \
                (TAU*y*(r ** TAU) * c_cos((c_pi/2. - RHO)*TAU) * \
                c_sin(TAU*(-3.*c_pi/2. - SIGMA + theta))/(r*r))
      
    # y-derivative
    if (theta <= c_pi/2.):
        dy = TAU*y * (r ** (2*(-1 + TAU/2.))) * \
        c_cos((c_pi/2. - SIGMA)*TAU) * \
        c_cos(TAU*(-c_pi/2. + RHO + theta)) - \
        (TAU * (r ** TAU) * c_cos((c_pi/2. - SIGMA)*TAU) * \
        c_sin(TAU*(-c_pi/2. + RHO + theta))*x/(r*r))
    else:
        if (theta <= c_pi):
            dy = TAU*y* (r ** (2*(-1 + TAU/2.))) * c_cos(RHO*TAU) * \
            c_cos(TAU*(-c_pi + SIGMA + theta)) - \
            (TAU * (r ** TAU) * c_cos(RHO*TAU) * \
            c_sin(TAU*(-c_pi + SIGMA + theta))*x/(r*r))
        else:
            if (theta <= 3.*c_pi/2.):
                dy = TAU*y * (r ** (2*(-1 + TAU/2.))) * c_cos(SIGMA*TAU) * \
                c_cos(TAU*(-c_pi - RHO + theta)) - \
                (TAU * (r ** TAU) * c_cos(SIGMA*TAU) * \
                c_sin(TAU*(-c_pi - RHO + theta))*x/(r*r))
            else:
                dy = TAU*y * (r ** (2*(-1 + TAU/2.))) * \
                c_cos((c_pi/2. - RHO)*TAU) * \
                c_cos(TAU*(-3.*c_pi/2. - SIGMA + theta)) - \
                (TAU * (r ** TAU) * c_cos((c_pi/2. - RHO)*TAU) * \
                c_sin(TAU*((-3.*c_pi)/2. - SIGMA + theta))*x/(r*r))

    return fn(x,y)

#def get_err_exact(Mesh m, Solution s):
#    exact = ExactSolution()
#    exact.thisptr.set_exact(m.thisptr, &fndd)
#    cdef double err_exact = h1_error(&s.thisptr, &exact.thisptr) * 100
#    return err_exact
    
# Boundary condition types
cdef c_BCType bc_type_kellogg(int marker):
    return <c_BCType>BC_ESSENTIAL

# Essential (Dirichlet) boundary condition values.
cdef scalar essential_bc_values(int ess_bdy_marker, double x, double y):
    return fn(x, y)

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

