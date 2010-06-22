from hermes2d._hermes2d cimport scalar, WeakForm, H1Space, EdgePos, \
        FuncReal, GeomReal, ExtDataReal, BC_ESSENTIAL, c_BCType, \
        BC_NATURAL, int_v, c_Ord, create_Ord, FuncOrd, GeomOrd, ExtDataOrd, \
        int_v_ord

CONST_GAMMA = [-0.5, 1.0, -0.5]

# Boundary condition types
# Note: natural means Neumann, Newton, or any other type of condition
# where the solution value is not prescribed.
cdef c_BCType bc_type_05(int marker):
    if marker == 4:
        return <c_BCType>BC_ESSENTIAL
    else:
        return <c_BCType>BC_NATURAL
        
# Function values for essential(Dirichlet) boundary markers
cdef scalar essential_bc_values(int marker, double x, double y):
    return 0.0

cdef c_Ord _order_lf(int n, double *wt, FuncOrd *u, GeomOrd
        *e, ExtDataOrd *ext):
    return int_v_ord(n, wt, u).mul_double(CONST_GAMMA[e.marker-1])

cdef scalar linear_form_surf_05(int n, double *wt, FuncReal *u, GeomReal
        *e, ExtDataReal *ext):
    return CONST_GAMMA[e.marker-1] * int_v(n, wt, u)

def set_forms(WeakForm dp):
    dp.thisptr.add_vector_form_surf(0, &linear_form_surf_05, &_order_lf);

def set_bc(H1Space space):
    space.thisptr.set_bc_types(&bc_type_05)
    space.thisptr.set_essential_bc_values(&essential_bc_values)
