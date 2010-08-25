#  With large K, this is a singularly perturbed problem that exhibits an extremely
#  thin and steep boundary layer. Singularly perturbed problems are considered to
#  be very difficult, but you'll see that Hermes can solve them easily even for large
#  values of K.
#
#  PDE: -Laplace u + K*K*u = K*K.
#
#  Domain: square, see the file square.mesh.
#
#  BC:  Homogeneous Dirichlet.

from hermes2d import Mesh, MeshView, VectorView, OrderView, H1Shapeset, PrecalcShapeset, H1Space, \
        WeakForm, Solution, ScalarView, LinSystem, DummySolver, RefSystem, \
    H1Adapt, H1ProjBasedSelector, CandList, \
        H2D_EPS_HIGH, H2D_FN_DX, H2D_FN_DY

from hermes2d.examples.csingular_perturbation import set_bc, set_forms
from hermes2d.examples import get_sp_square_mesh

#  The following parameters can be changed:

SOLVE_ON_COARSE_MESH = False # If true, coarse mesh FE problem is solved in every adaptivity step.
                                         # If false, projection of the fine mesh solution on the coarse mesh is used. 
INIT_REF_NUM = 1              # Number of initial mesh refinements (the original mesh is just one element)
INIT_REF_NUM_BDY = 3          # Number of initial mesh refinements towards the boundary
P_INIT = 1                    # Initial polynomial degree of all mesh elements.
THRESHOLD = 0.3            # This is a quantitative parameter of the adapt(...) function and
                                         # it has different meanings for various adaptive strategies (see below).
STRATEGY = 0                  # Adaptive strategy:
                                         # STRATEGY = 0 ... refine elements until sqrt(THRESHOLD) times total
                                         #   error is processed. If more elements have similar errors, refine
                                         #   all to keep the mesh symmetric.
                                         # STRATEGY = 1 ... refine all elements whose error is larger
                                         #   than THRESHOLD times maximum element error.
                                         # STRATEGY = 2 ... refine all elements whose error is larger
                                         #   than THRESHOLD.
                                         # More adaptive strategies can be created in adapt_ortho_h1.cpp.
CAND_LIST = CandList.H2D_HP_ANISO  # Predefined list of element refinement candidates. Possible values are
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
ERR_STOP = 0.1             # Stopping criterion for adaptivity (rel. error tolerance between the
                                         # fine mesh and coarse mesh solution in percent).
NDOF_STOP = 100000            # Adaptivity process stops when the number of degrees of freedom grows
                                         # over this limit. This is to prevent h-adaptivity to go on forever.
                                         
H2DRS_DEFAULT_ORDER = -1     # A default order. Used to indicate an unkonwn order or a maximum support order

# Load the mesh.
mesh = Mesh()
mesh.load(get_sp_square_mesh())

# Perform initial mesh refinements
for i in range(INIT_REF_NUM):
    mesh.refine_all_elements()
mesh.refine_towards_boundary(1, INIT_REF_NUM_BDY)

# Create an H1 space with default shapeset
space = H1Space(mesh, P_INIT)
set_bc(space)

# Initialize the weak formulation
wf = WeakForm()
set_forms(wf)

# Initialize views.
sview = ScalarView("Coarse mesh solution")
#OrderView  oview("Polynomial orders")

# Initialize refinement selector.
selector = H1ProjBasedSelector(CAND_LIST, CONV_EXP, H2DRS_DEFAULT_ORDER)

# Initialize the coarse mesh problem.
ls = LinSystem(wf)
ls.set_spaces(space)

# Adaptivity loop:
it = 1
done = False
sln_coarse = Solution()
sln_fine = Solution()

while(not done):
    print("\n---- Adaptivity step %d ----\n" % it)
    it += 1

    # Assemble and solve the fine mesh problem.
    print("Solving on fine mesh.") 
    rs = RefSystem(ls)
    rs.assemble()
    rs.solve_system(sln_fine, lib="hermes")

    # Either solve on coarse mesh or project the fine mesh solution
    # on the coarse mesh.
    if SOLVE_ON_COARSE_MESH:
        print("Solving on coarse mesh.")
        ls.assemble()
        ls.solve_system(sln_coarse)
    else:
        print("Projecting fine mesh solution on coarse mesh.")
        ls.project_global(sln_fine, sln_coarse)
    
    # View the solution and mesh.
    sview.show(sln_coarse)
    mesh.plot(space)

    # Calculate error estimate wrt. fine mesh solution.
    print("Calculating error (est).")
    hp = H1Adapt(ls)
    hp.set_solutions([sln_coarse], [sln_fine])
    err_est = hp.calc_error() * 100

    # If err_est too large, adapt the mesh.
    if (err_est < ERR_STOP):
        done = True
    else:
        done = hp.adapt(selector, THRESHOLD, STRATEGY, MESH_REGULARITY)
        if (ls.get_num_dofs() >= NDOF_STOP):
            done = True

# Show the fine mesh solution - the final result.
#sview.set_title("Final solution")
sview.show_mesh(False)
sview.show(sln_fine)
