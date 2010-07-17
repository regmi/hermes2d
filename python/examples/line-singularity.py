#  This is a simple elliptic problem with known exact solution where one
#  can compare isotropic and anisotropic refinements.
#
#  PDE: -Laplace u - K*K*u = f.
#  where f is dictated by the exact solution.
#
#  Exact solution: u(x,y) = cos(K*y)    for x < 0,
#                  u(x,y) = cos(K*y) + pow(x, alpha)   for x > 0   where alpha > 0.
#
#  Domain: square, see the file singpert.mesh.
#
#  BC:  Homogeneous Dirichlet.
#
#  The following parameters can be changed:

SOLVE_ON_COARSE_MESH = True # If true, coarse mesh FE problem is solved in every adaptivity step.
                                         # If false, projection of the fine mesh solution on the coarse mesh is used. 
INIT_REF_NUM = 0              # Number of initial mesh refinements (the original mesh is just one element)
P_INIT = 2                    # Initial polynomial degree of all mesh elements.
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
CAND_LIST = CandList.H2D_HP_ANISO # Predefined list of element refinement candidates. Possible values
                                         # are H2D_P_ISO, H2D_P_ANISO, H2D_H_ISO, H2D_H_ANISO, H2D_HP_ISO,
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
ERR_STOP = 0.0001          # Stopping criterion for adaptivity (rel. error tolerance between the
                                         # fine mesh and coarse mesh solution in percent).
NDOF_STOP = 100000            # Adaptivity process stops when the number of degrees of freedom grows
                                         # over this limit. This is to prevent h-adaptivity to go on forever.

# Load the mesh.
mesh = Mesh()
mesh.load(get_ls_square_quad_mesh())

# Perform initial mesh refinements
for i in range(INIT_REF_NUM):
    mesh.refine_all_elements()

# Create an H1 space with default shapeset
space = H1Space(mesh, P_INIT)
set_bc(space)

# Enumerate degrees of freedom.
ndof = assign_dofs(space);

# Initialize the weak formulation
wf = WeakForm()
set_forms(wf)

# Initialize views.
sview = ScalarView("Coarse mesh solution", 0, 0, 440, 350)
oview = OrderView("Coarse mesh", 450, 0, 400, 350)

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

    # View the solution and mesh.
    sview.show(sln_coarse)
    mesh.plot(space)

    # Calculate error estimate wrt. fine mesh solution.
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
sview.show(sln_fine)
