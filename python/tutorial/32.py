# This example shows how to use the L2 finite element space and L2 shapeset.
# As a sample problem, a continuous function x^3 + y^3 is projected onto the
# L2 finite element space in the L2 norm. When zero-order is used, the result
# is a piecewice constant function. The class BaseView will show you the basis
# functions.

# Import modules
from hermes2d import Mesh, MeshView, VectorView, OrderView, H1Shapeset, PrecalcShapeset, H1Space, L2Space \
        WeakForm, Solution, ScalarView, LinSystem, DummySolver, RefSystem, \
    H1Adapt, H1ProjBasedSelector, CandList, \
        H2D_EPS_HIGH, H2D_FN_DX, H2D_FN_DY

from hermes2d.examples.c32 import set_bc, set_forms
from hermes2d.examples import get_square_mesh

# The following parameters can be changed:

INIT_REF_NUM = 1    # Number of initial uniform mesh refinements.
P_INIT = 3          # Polynomial degree of mesh elements.

# Projected function.
def F(x, y, dx, dy):
    return - (x ** 4) * (y ** 5) 
    dx = 0 # not needed for L2-projection
    dy = 0 # not needed for L2-projection

# Load the mesh.
mesh = Mesh()
mesh.load(get_square_mesh())

# Perform initial mesh refinements
for i in range(INIT_REF_NUM):
    mesh.refine_all_elements()

# Create an L2 space with default shapeset.
space = L2Space(mesh, P_INIT)
set_bc(space)

# View basis functions.
#BaseView bview()
#bview.show(space)
#View::wait(H2DV_WAIT_KEYPRESS)

# Assemble and solve the finite element problem.
wf_dummy = WeakForm()
ls = LinSystem(wf_dummy)
ls.set_spaces(space)
sln = Solution()
proj_norm_l2 = 0
ls.project_global(F, sln, proj_norm_l2);

# Visualize the solution.
view1 = ScalarView()
view1.show(sln)
