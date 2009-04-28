// This file is part of Hermes2D.
//
// Copyright 2005-2008 Jakub Cerveny <jakub.cerveny@gmail.com>
// Copyright 2005-2008 Lenka Dubcova <dubcova@gmail.com>
// Copyright 2005-2008 Pavel Solin <solin@unr.edu>
//
// Hermes2D is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// Hermes2D is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Hermes2D.  If not, see <http://www.gnu.org/licenses/>.

// $Id: solution.h 1086 2008-10-21 09:05:44Z jakub $

#ifndef __HERMES2D_SOLUTION_H
#define __HERMES2D_SOLUTION_H

#include "function.h"
#include "space.h"
#include "refmap.h"

class PrecalcShapeset;


/// \brief Represents a function defined on a mesh.
///
/// MeshFunction is a base class for all classes representing an arbitrary function
/// superimposed on a mesh (ie., domain). These include the Solution, ExactSolution
/// and Filter classes, which define the concrete behavior and the way the function
/// is (pre)calculated. Any such function can later be visualized. 
/// 
/// (This is an abstract class and cannot be instantiated.)
///
class MeshFunction : public ScalarFunction
{
public:
  
  MeshFunction();
  virtual ~MeshFunction();

  virtual void set_quad_2d(Quad2D* quad_2d);
  virtual void set_active_element(Element* e);

  Mesh*   get_mesh() const { return mesh; }
  RefMap* get_refmap() { update_refmap(); return refmap; }

  virtual scalar get_pt_value(double x, double y, int item = FN_VAL_0) = 0;
  
protected:
  
  int mode;
  Mesh* mesh;
  RefMap* refmap;

public:

  /// For internal use only.
  void force_transform(MeshFunction* mf)
    { ScalarFunction::force_transform(mf->get_transform(), mf->get_ctm()); }
  void update_refmap()
    { refmap->force_transform(sub_idx, ctm); }

};


/// \brief Represents the solution of a PDE.
///
/// The Solution class represents the solution of a PDE. Given a space and a solution vector,
/// it calculates the appropriate linear combination of basis functions at the specified
/// element and integration points.
///
/// TODO: write how to obtain solution values, maybe include inherited methods from Function as comments.
///
class Solution : public MeshFunction
{
public:
  
  Solution();
  virtual ~Solution();
  virtual void free();

  void assign(Solution* sln);
  Solution& operator = (Solution& sln) { assign(&sln); return *this; }
  void copy(const Solution* sln);
  
  void set_exact(Mesh* mesh, scalar   (*exactfn)(double x, double y, scalar& dx , scalar& dy));
  void set_exact(Mesh* mesh, scalar2& (*exactfn)(double x, double y, scalar2& dx, scalar2& dy));

  void set_const(Mesh* mesh, scalar c);
  void set_const(Mesh* mesh, scalar c0, scalar c1); // two-component (Hcurl) const

  void set_zero(Mesh* mesh);
  void set_zero_2(Mesh* mesh); // two-component (Hcurl) zero

  /// Sets solution equal to Dirichlet lift only, solution vector = 0
  void set_dirichlet_lift(Space* space, PrecalcShapeset* pss);
  
  /// Enables or disables transformation of the solution derivatives (H1 case)
  /// or values (vector (Hcurl) case). This means FN_DX_0 and FN_DY_0 or
  /// FN_VAL_0 and FN_VAL_1 will or will not be returned premultiplied by the reference
  /// mapping matrix. The default is enabled (true).
  void enable_transform(bool enable = true);

  /// Saves the complete solution (i.e., including the internal copy of the mesh and
  /// element orders) to a binary file. On Linux, if `compress` is true, the file is
  /// compressed with gzip and a ".gz" suffix added to the file name.
  void save(const char* filename, bool compress = true);
  
  /// Loads the solution from a file previously created by Solution::save(). This completely 
  /// restores the solution in the memory. The file name has to include the ".gz" suffix,
  /// in which case the file is piped through gzip to decompress the data (Linux only).
  void load(const char* filename);

  /// Returns solution value or derivatives at element e, in its reference domain point (xi1, xi2).
  /// 'item' controls the returned value: 0 = value, 1 = dx, 2 = dy, 3 = dxx, 4 = dyy, 5 = dxy.
  /// NOTE: This function should be used for postprocessing only, it is not effective 
  /// enough for calculations.
  scalar get_ref_value(Element* e, double xi1, double xi2, int component = 0, int item = 0);

  /// Returns solution value or derivatives (correctly transformed) at element e, in its reference 
  /// domain point (xi1, xi2). 'item' controls the returned value: 0 = value, 1 = dx, 2 = dy, 
  /// 3 = dxx, 4 = dyy, 5 = dxy.  
  /// NOTE: This function should be used for postprocessing only, it is not effective 
  /// enough for calculations.
  scalar get_ref_value_transformed(Element* e, double xi1, double xi2, int a, int b);

  /// Returns solution value or derivatives at the physical domain point (x, y).
  /// 'item' controls the returned value: FN_VAL_0, FN_VAL_1, FN_DX_0, FN_DX_1, FN_DY_0,....
  /// NOTE: This function should be used for postprocessing only, it is not effective 
  /// enough for calculations.
  virtual scalar get_pt_value(double x, double y, int item = FN_VAL_0);

  /// Returns the number of degrees of freedom of the solution.
  /// Returns -1 for exact or constant solutions.
  int get_num_dofs() const { return num_dofs; };

  /// Multiplies the function represented by this class by the given coefficient.
  void multiply(scalar coef);

  
public:

  /// Internal. Used by LinSystem::solve(). Should not be called directly
  virtual void set_fe_solution(Space* space, PrecalcShapeset* pss, scalar* vec, double dir = 1.0);

  /// Internal.
  virtual void set_active_element(Element* e);


protected:

  enum { SLN, EXACT, CNST, UNDEF } type;

  bool own_mesh;
  bool transform;

  void* tables[4][4];   ///< precalculated tables for last four used elements
  Element* elems[4][4];
  int cur_elem, oldest[4];

  scalar* mono_coefs;  ///< monomial coefficient array
  int* elem_coefs[2];  ///< array of pointers into mono_coefs
  int* elem_orders;    ///< stored element orders
  int num_coefs, num_elems;
  int num_dofs;

  scalar   (*exactfn1)(double x, double y, scalar& dx,  scalar& dy);
  scalar2& (*exactfn2)(double x, double y, scalar2& dx, scalar2& dy);
  scalar   cnst[2];
  scalar   exact_mult;

  virtual void precalculate(int order, int mask);
  
  scalar* dxdy_coefs[2][6];
  scalar* dxdy_buffer;

  double** calc_mono_matrix(int o, int*& perm);
  void init_dxdy_buffer();
  void free_tables();

  Element* e_last; ///< last visited element when getting solution values at specific points
  
};


/// \brief Represents and exact solution of a PDE.
///
/// ExactSolution represents an arbitrary user-specified function defined on a domain (mesh),
/// typically an exact solution to a PDE. This can be used to compare an approximate solution
/// with an exact solution (see DiffFilter).
///
/// Please note that the same functionality can be obtained by using Solution::set_exact().
/// This class is provided merely for convenience.
/// 
class ExactSolution : public Solution
{
public:
  
  ExactSolution(Mesh* mesh, scalar   (*exactfn)(double x, double y, scalar& dx , scalar& dy))
    { set_exact(mesh, exactfn); }
     
  ExactSolution(Mesh* mesh, scalar2& (*exactfn)(double x, double y, scalar2& dx, scalar2& dy))
    { set_exact(mesh, exactfn); }

};



#endif
