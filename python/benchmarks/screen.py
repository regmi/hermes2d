#  This benchmark problem with known exact solution describes an electromagnetic wave that hits
#  a screen under the angle of 45 degrees, causing a very strong singularity at the tip of the 
#  screen. Convergence graphs saved (both exact error and error estimate, and both wrt. dof number.
#  and cpu time).
#
#  PDE: time-harmonic Maxwell's equations.
#
#  Known exact solution, see the function exact().
#
#  Domain: square domain cut from the midpoint of the left edge to the center (center
#          point of left edge duplicated).
#
#  Meshes: you can either use "screen-quad.mesh" (quadrilateral mesh) or
#          "screen-tri.mesh" (triangular mesh). See the command mesh.load(...) below.
#
#  BC: tangential component of solution taken from known exact solution (essential BC),
#      see function essential_bc_values() below.

# Import modules
from hermes2d import Mesh, MeshView, VectorView, OrderView, H1Shapeset, PrecalcShapeset, H1Space, \
        WeakForm, Solution, ScalarView, LinSystem, DummySolver, RefSystem, \
    H1Adapt, H1ProjBasedSelector, CandList, \
        H2D_EPS_HIGH, H2D_FN_DX, H2D_FN_DY

from hermes2d.examples.cscreen import set_bc, set_forms
from hermes2d.examples import get_screen-quad_mesh

# The following parameters can be changed:

SOLVE_ON_COARSE_MESH = True # If true, coarse mesh FE problem is solved in every adaptivity step.
                                         # If false, projection of the fine mesh solution on the coarse mesh is used. 
INIT_REF_NUM = 1              # Number of initial uniform mesh refinements.
P_INIT = 1                    # Initial polynomial degree. NOTE: The meaning is different from
                                         # standard continuous elements in the space H1. Here, P_INIT refers
                                         # to the maximum poly order of the tangential component, and polynomials
                                         # of degree P_INIT + 1 are present in element interiors. P_INIT = 0
                                         # is for Whitney elements.
THRESHOLD = 0.5            # This is a quantitative parameter of the adapt(...) function and
                                         # it has different meanings for various adaptive strategies (see below).
STRATEGY = 1                  # Adaptive strategy:
                                         # STRATEGY = 0 ... refine elements until sqrt(THRESHOLD) times total
                                         #   error is processed. If more elements have similar errors, refine
                                         #   all to keep the mesh symmetric.
                                         # STRATEGY = 1 ... refine all elements whose error is larger
                                         #   than THRESHOLD times maximum element error.
                                         # STRATEGY = 2 ... refine all elements whose error is larger
                                         #   than THRESHOLD.
                                         # More adaptive strategies can be created in adapt_ortho_h1.cpp.
CAND_LIST = CandList.H2D_HP_ANISO_H # Predefined list of element refinement candidates. Possible values are
                                         # H2D_P_ISO, H2D_P_ANISO, H2D_H_ISO, H2D_H_ANISO, H2D_HP_ISO,
                                         # H2D_HP_ANISO_H, H2D_HP_ANISO_P, H2D_HP_ANISO.
                                         # See User Documentation for details.
MESH_REGULARITY = -1          # Maximum allowed level of hanging nodes:
                                         # MESH_REGULARITY = -1 ... arbitrary level hangning nodes (default),
                                         # MESH_REGULARITY = 1 ... at most one-level hanging nodes,
                                         # MESH_REGULARITY = 2 ... at most two-level hanging nodes, etc.
                                         # Note that regular meshes are not supported, this is due to
                                         # their notoriously bad performance.
CONV_EXP = 1.0             # Default value is 1.0. This parameter influences the selection of
                                         # cancidates in hp-adaptivity. See get_optimal_refinement() for details.
ERR_STOP = 2.0             # Stopping criterion for adaptivity (rel. error tolerance between the
                                         # fine mesh and coarse mesh solution in percent).
NDOF_STOP = 50000             # Adaptivity process stops when the number of degrees of freedom grows
                                         # over this limit. This is to prevent h-adaptivity to go on forever.

# Load the mesh.
mesh = Mesh()
mesh.load(get_screen-quad_mesh())   # quadrilaterals
#mesh.load(get_screen-tri_mesh())   # triangles

# Perform initial mesh refinements
for i in range(INIT_REF_NUM):
    mesh.refine_all_elements()

# Create an Hcurl space with default shapeset
space = HcurlSpace(mesh, P_INIT)
set_bc(space)

# Initialize the weak formulation
wf = WeakForm()
set_forms(wf)

# Initialize views.
Xview_r = ScalarView("Electric field X - real",   0, 0, 300, 280)
Yview_r = ScalarView("Electric field Y - real", 310, 0, 300, 280)
Xview_i = ScalarView("Electric field X - imag", 620, 0, 300, 280)
Yview_i = ScalarView("Electric field Y - imag", 930, 0, 300, 280)
ord = OrderView("Polynomial Orders", 0, 335, 300, 280)

# Initialize refinement selector.
selector = H1ProjBasedSelector(CAND_LIST, CONV_EXP, H2DRS_DEFAULT_ORDER)

# Initialize the coarse mesh problem.
ls = LinSystem(wf)
ls.set_spaces(space)space)

# Adaptivity loop:
it = 1
done = False
sln_coarse = Solution()
sln_fine = Solution()

while(not done):
    print("\n---- Adaptivity step %d ----\n" % it)
    it += 1
    
    # Assemble and solve the fine mesh problem.
    rs = RefSystem(ls)
    rs.assemble()
    rs.solve_system(sln_fine, lib="hermes")

    # Either solve on coarse mesh or project the fine mesh solution 
    # on the coarse mesh.
    if SOLVE_ON_COARSE_MESH:
        ls.assemble()
        ls.solve_system(sln_coarse)
    else:
        ls.project_global(sln_fine, sln_coarse)

    # Calculate error wrt. exact solution.
    ex = Solution()
    ex.set_exact(mesh, exact);
    err_exact = 100 * ex.hcurl_error(sln_coarse, ex)

    # Visualization.
    real = RealFilter(sln_coarse)
    imag = ImagFilter(sln_coarse)
    Xview_r.set_min_max_range(-3.0, 1.0)
    #Xview_r.show_scale(false)
    Xview_r.show(real, H2D_EPS_NORMAL, H2D_FN_VAL_0)
    Yview_r.set_min_max_range(-4.0, 4.0)
    #Yview_r.show_scale(false)
    Yview_r.show(real, H2D_EPS_NORMAL, H2D_FN_VAL_1)
    Xview_i.set_min_max_range(-1.0, 4.0)
    #Xview_i.show_scale(false)
    Xview_i.show(imag, H2D_EPS_NORMAL, H2D_FN_VAL_0)
    Yview_i.set_min_max_range(-4.0, 4.0)
    #Yview_i.show_scale(false)
    Yview_i.show(imag, H2D_EPS_NORMAL, H2D_FN_VAL_1)
    ord.show(space)

    # Calculate error estimate wrt. fine mesh solution.
    hp = HcurlAdapt(ls)
    hp.set_solutions(sln_coarse, sln_fine)
    err_est_adapt = hp.calc_error() * 100
    err_est_hcurl = hcurl_error(sln_coarse, sln_fine) * 100
    
    # If err_est too large, adapt the mesh.
    if (err_est < ERR_STOP):
        done = True
    else:
        done = hp.adapt(selector, THRESHOLD, STRATEGY, MESH_REGULARITY)

        if (ls.get_num_dofs() >= NDOF_STOP):
            done = True