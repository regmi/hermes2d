import os

def get_example_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "domain.mesh")
    return os.path.normpath(mesh)

def get_sample_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "sample.mesh")
    return os.path.normpath(mesh)

def get_cylinder_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "cylinder4.mesh")
    return os.path.normpath(mesh)

def get_cathedral_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "cathedral.mesh")
    return os.path.normpath(mesh)

def get_motor_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "motor.mesh")
    return os.path.normpath(mesh)

def get_07_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "07.mesh")
    return os.path.normpath(mesh)

def get_bracket_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "bracket.mesh")
    return os.path.normpath(mesh)

def get_12_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "12.mesh")
    return os.path.normpath(mesh)
    
def get_square_quad_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "square_quad.mesh")
    return os.path.normpath(mesh)

def get_ls_square_quad_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "ls_square_quad.mesh")
    return os.path.normpath(mesh)

def get_ls_square_quad_mesh():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "siso_square_quad.mesh")
    return os.path.normpath(mesh)

def get_square_quad_mesh_smooth_iso():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "square_quad.mesh")
    return os.path.normpath(mesh)

def get_square_quad_mesh_smooth_aniso-x():
    """
    Returns an example mesh, that is distributed with hermes2d.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mesh = os.path.join(this_dir, "smooth_aniso-x_square_quad.mesh")
    return os.path.normpath(mesh)
