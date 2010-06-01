// This file is part of Hermes2D.
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

#ifndef __H2D_REFSYSTEM_H
#define __H2D_REFSYSTEM_H

#include "linsystem.h"

class Mesh;
class ExactSolution;

///
///
///
class H2D_API RefSystem : public LinSystem
{
public:
  RefSystem(LinSystem* base, int order_increase = 1, int refinement = 1);
  virtual ~RefSystem();

  /// Do not call in this class
  void set_spaces(int n, ...);
  /// Do not call in this class
  void set_pss(int n, ...);

  /// Sets different order increases for components
  /// The length of array "order_increase" must be equal to the number of equations
  void set_order_increase(int* order_increase);

  /// Creates reference (fine) meshes and spaces and assembles the
  /// reference system.
  void assemble(bool rhsonly = false);

  /// Copies and refines meshes from coarse mesh linsystem
  void prepare();

  bool solve_exact(scalar (*exactfn)(double x, double y, scalar& dx, scalar& dy), Solution* sln);

  /// Frees reference spaces and meshes. Called
  /// automatically on desctruction.
  void free_meshes_and_spaces();

  /// Gets pointer to reference space [i]
  Space* get_ref_space(int i = 0) {return ref_spaces[i];}

  /// Gets number of DOF in reference space 'i'
  int get_num_dofs(int i) {return ref_spaces[i]->get_num_dofs();}

  /// Gets total number of DOF in the reference system.
  int get_num_dofs();

  /// Gets pointer to reference mesh [i]
  Mesh* get_ref_mesh(int i=0) {return ref_meshes[i];}

protected:

  LinSystem* base;
  int order_inc[1000];   // FIXME: this should be done using dynamic allocation.
  int refinement;

  Mesh**  ref_meshes;
  Space** ref_spaces;

};



#endif
