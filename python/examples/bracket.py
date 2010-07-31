# This example uses adaptive multimesh hp-FEM to solve a simple problem
# of linear elasticity. Note that since both displacement components
# have similar qualitative behavior, the advantage of the multimesh 
# discretization is less striking than for example in the tutorial 
# example 11-adapt-system.
#
# PDE: Lame equations of linear elasticity, treated as a coupled system
#      of two PDEs.
#
# BC: u_1 = u_2 = 0 on Gamma_1
#     du_2/dn = f on Gamma_2
#     du_1/dn = du_2/dn = 0 elsewhere

from hermes2d import Mesh, MeshView, VectorView, OrderView, H1Shapeset, PrecalcShapeset, H1Space, \
        WeakForm, Solution, ScalarView, LinSystem, DummySolver, RefSystem, \
    H1Adapt, H1ProjBasedSelector, CandList, \
        H2D_EPS_HIGH, H2D_FN_DX, H2D_FN_DY

from hermes2d.examples.cbracket import set_bc, set_forms
from hermes2d.examples import get_bracket_mesh

# The following parameters can be changed: 

SOLVE_ON_COARSE_MESH = True # If true, coarse mesh FE problem is solved in every adaptivity step.
                                         # If false, projection of the fine mesh solution on the coarse mesh is used. 
P_INIT = 2                    # Initial polynomial degree of all mesh elements.
MULTI = True                 # MULTI = true  ... use multi-mesh,
                                         # MULTI = false ... use single-mesh.
                                         # Note: In the single mesh option, the meshes are
                                         # forced to be geometrically the same but the
                                         # polynomial degrees can still vary.
THRESHOLD = 0.3            # This is a quantitative parameter of the adapt(...) function and
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
CAND_LIST = CandList.H2D_HP_ANISO # Predefined list of element refinement candidates. Possible values are
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
ERR_STOP = 0.5             # Stopping criterion for adaptivity (rel. error tolerance between the
                                         # fine mesh and coarse mesh solution in percent).
NDOF_STOP = 60000             # Adaptivity process stops when the number of degrees of freedom grows over
                                         # this limit. This is mainly to prevent h-adaptivity to go on forever.

# Load the mesh.
xmesh = Mesh()
ymesh = Mesh()
xmesh.load(get_bracket_mesh())


# Initial mesh refinements.
xmesh.refine_element(1)
xmesh.refine_element(4)

# Create initial mesh for the vertical displacement component.
# This also initializes the multimesh hp-FEM.
ymesh.copy(xmesh)

# Create H1 spaces with default shapesets.
xdisp = H1Space(mesh, P_INIT)
if MULTI:
    ydisp = H1Space(ymesh, P_INIT)
else:
    ydisp = H1Space(xmesh, P_INIT)
set_bc(xdisp, ydisp)

# Initialize the weak formulation.
wf = WeakForm(2)
set_forms(wf)

# Initialize views.
xoview = OrderView()
yoview = OrderView ()
sview = ScalarView()

# Initialize refinement selector.
selector = H1ProjBasedSelector(CAND_LIST, CONV_EXP, H2DRS_DEFAULT_ORDER)

# Initialize the coarse mesh problem.
#LinSystem ls(&wf, Tuple<Space*>(&xdisp, &ydisp));
ls = LinSystem(wf)
ls.set_spaces(space)

# Adaptivity loop:
it = 1
done = False
x_sln_coarse = Solution()
y_sln_coarse = Solution()
x_sln_fine = Solution()
y_sln_fine = Solution()

while(not done):
    print("\n---- Adaptivity step %d ----\n" % it)

# Assemble and solve the fine mesh problem.
    rs = RefSystem(ls)
    rs.assemble()
    rs.solve_system(sln_fine, lib="hermes")
    #rs.solve(Tuple<Solution*>(&x_sln_fine, &y_sln_fine));

    # Either solve on coarse mesh or project the fine mesh solution 
    # on the coarse mesh.
    if SOLVE_ON_COARSE_MESH:
        ls.assemble()
        ls.solve_system(sln_coarse)
        #ls.solve(Tuple<Solution*>(&x_sln_coarse, &y_sln_coarse));
    else:
        ls.project_global(sln_fine, sln_coarse)
        #ls.project_global(Tuple<MeshFunction*>(&x_sln_fine, &y_sln_fine), Tuple<Solution*>(&x_sln_coarse, &y_sln_coarse))

    # View the solution and mesh.
    stress_coarse = VonMisesFilter(x_sln_coarse, y_sln_coarse, mu, lamda);
    sview.set_min_max_range(0, 3e4)
    sview.show(stress_coarse)
    xoview.show(xdisp)
    yoview.show(ydisp)

# Skip visualization time. 
cpu_time.tick(HERMES_SKIP);

# Calculate element errors and total error estimate.
info("Calculating error (est).");
H1Adapt hp(&ls);
hp.set_solutions(Tuple<Solution*>(&x_sln_coarse, &y_sln_coarse), Tuple<Solution*>(&x_sln_fine, &y_sln_fine));
hp.set_error_form(0, 0, bilinear_form_0_0<scalar, scalar>, bilinear_form_0_0<Ord, Ord>);
hp.set_error_form(0, 1, bilinear_form_0_1<scalar, scalar>, bilinear_form_0_1<Ord, Ord>);
hp.set_error_form(1, 0, bilinear_form_1_0<scalar, scalar>, bilinear_form_1_0<Ord, Ord>);
hp.set_error_form(1, 1, bilinear_form_1_1<scalar, scalar>, bilinear_form_1_1<Ord, Ord>);
double err_est = hp.calc_error(H2D_TOTAL_ERROR_REL | H2D_ELEMENT_ERROR_REL) * 100;

# Report results.
info("ndof_x_coarse: %d, ndof_x_fine: %d", 
     ls.get_num_dofs(0), rs.get_num_dofs(0));
info("ndof_y_coarse: %d, ndof_y_fine: %d", 
     ls.get_num_dofs(1), rs.get_num_dofs(1));
info("ndof: %d, err_est: %g%%", ls.get_num_dofs(), err_est);

# Add entry to DOF convergence graph.
graph_dof.add_values(ls.get_num_dofs(), err_est);
graph_dof.save("conv_dof.dat");

# Add entry to CPU convergence graph.
graph_cpu.add_values(cpu_time.accumulated(), err_est);
graph_cpu.save("conv_cpu.dat");

# If err_est too large, adapt the mesh.
if (err_est < ERR_STOP) done = true;
else {
  info("Adapting the coarse mesh.");
  done = hp.adapt(&selector, THRESHOLD, STRATEGY, MESH_REGULARITY);
  if (ls.get_num_dofs() >= NDOF_STOP) done = true;
}

as++;
}
while (!done);
verbose("Total running time: %g s", cpu_time.accumulated());

# Show the fine mesh solution - the final result
VonMisesFilter stress_fine(&x_sln_fine, &y_sln_fine, mu, lambda);
sview.set_title("Fine mesh solution");
sview.set_min_max_range(0, 3e4);
sview.show_mesh(false);
sview.show(&stress_fine);

# Wait for all views to be closed.
View::wait();
return 0;
}
