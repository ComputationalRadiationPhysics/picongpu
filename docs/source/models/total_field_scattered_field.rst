.. _model-TFSF:

Total Field/Scattered Field
===========================

.. sectionauthor:: Sergei Bastrakov

This section describes the Total Field/Scattered Field (TF/SF) technique for adding incident field to a simulation.
TF/SF is used for simulating a field produced by external sources and coming towards the simulation volume from any boundaries and directions.
PIConGPU implements this technique as incident field.
There are also :ref:`other ways to generate a laser <usage-workflows-addLaser>`.

First, we present the general idea of TF/SF and describe its formulation for the standard Yee field solver.
Then we show how it is generalized for any FDTD-type solver, this general version is implemented in PIConGPU.
Finally, some aspects of using the PIConGPU incident field are covered.

Introduction
------------

When applying TF/SF, we operate on a domain where field values are updated by an explicit FDTD-type Maxwell's solver.
In case a simulation uses a field absorber, the absorber operates externally to this area and independently of TF/SF (no special coupling needed).

This section uses the same Yee grid and notation as the :ref:`FDTD description <model-AOFDTD>`.
The description and naming is generally based on [Potter2017]_ [Taflove2005]_, however there are some differences in notation (also those two sources use different Yee grid layouts).

The field produced in the domain by external sources (such as incoming laser) is called *incident field* and denoted :math:`\vec E^{inc}(x, y, z, t)` and :math:`\vec B^{inc}(x, y, z, t)`.
*Total field (TF)* is the combined field from the internally occuring field dynamic and the incident field, we denote it :math:`\vec E^{tot}(x, y, z, t)` and :math:`\vec B^{tot}(x, y, z, t)`.
Thus, total field is a "normal" result of a simulation with the laser.
Finally, the *scattered field (SF)* is the difference of these two fields:

.. math::

   \vec E^{scat}(x, y, z, t) &= \vec E^{tot}(x, y, z, t) - \vec E^{inc}(x, y, z, t)

   \vec B^{scat}(x, y, z, t) &= \vec B^{tot}(x, y, z, t) - \vec B^{inc}(x, y, z, t)

Scattered field does not have the laser effect, and transparently propagates the waves outgoing from the TF area towards domain boundaries.
However, note that Maxwell's equations hold in both the TF and SF regions.
Special handling is needed only near the boundary betweeen the regions.

The field values are represented as a Yee grid of :math:`\vec E` and :math:`\vec B` values in the domain, TF/SF does not require additionally stored data.
However, it changes the interpretation of grid values: the domain is virtually subdivided into the internal area where the stored grid values are TF and external area with SF values.

The subvidision is done with an axis-aligned *Huygens surface* :math:`S`.
The Huygens surface is chosen so that no Yee grid nodes lay on it.
So it could be located at an arbitrary position that does not collide with cell- and half-cell boundaries.

In PIConGPU, the position of the Huygens surface is defined relative to the internal boundary of the field absorber.
The surface is shifted inwards relative to each boundary by a user-defined gap in full cells and an additional 0.75 cells.
The equations presented in  this section hold for this 0.75 shift and the Yee grid layout used in PIConGPU.
In principle, a similar derivation can be done in any case, but the resulting expression would not generally match index-by-index. 

We use TF/SF formulation to generate fields at the Huygens surface that propagate inwards the simulation area and act as application of :math:`\vec E^{inc}(x, y, z, t)`, :math:`\vec B^{inc}(x, y, z, t)`.
The fields would mostly propagate in this direction (and not equally to both sides), and as a theoretical limit only inwards.
However, in practice due to numerical dispersion a small portion, such as :math:`10^{-3}` of the amplitude, would also propagate externally.

The TF/SF technique can be interpreted in two ways.
First, as manipulations on discrete FDTD equations given the virtual separation of the domain described above.
This "math" view is used in the following description for the Yee and general-case FDTD solvers.

Alternatively, TF/SF can be treated as applying the equivalence theorem on the Huygens surface.
Then it is equivalent to impressing the electric and magnetic current sheets, :math:`\vec J` and :math:`\vec K` respectively, on :math:`S`:

.. math::

   \vec J \rvert_S &= - \vec n \times \frac{1}{\mu_0} \vec B^{inc} \rvert_S

   \vec K \rvert_S &= \vec n \times \vec E^{inc} \rvert_S

where :math:`\vec n` is a unit normal vector on :math:`S` pointing towards the internal volume.
Both points of view in principle result in the same scheme.

TF/SF for Standard Yee Solver
-----------------------------

Single Boundary
^^^^^^^^^^^^^^^

First we describe application of a field source only at the :math:`x_{min}` boundary.
For this case, suppose the source is given along a plane :math:`x = x_{min} + gap + 0.75 \Delta x` and acts along the whole domain in :math:`y` and :math:`z`.
The source affects transversal field components :math:`E_y`, :math:`E_z`, :math:`B_y`, :math:`B_z`.
Components :math:`E_x`, :math:`B_x` are not affected by TF/SF and we do not consider them in the following description.
We also omit the :math:`\vec J` term which is applied as usual in the field solver and is also not affected by TF/SF.

The Yee solver update equations for the affected field components and terms are as follows:

.. math::
   \frac{E_y\rvert_{i, j+1/2, k}^{n+1} - E_y\rvert_{i, j+1/2, k}^{n}}{c^2 \Delta t} =&
    \frac{B_x\rvert_{i, j+1/2, k+1/2}^{n+1/2} - B_x\rvert_{i, j+1/2, k-1/2}^{n+1/2}}{\Delta z}

   & - \frac{B_z\rvert_{i+1/2, j+1/2, k}^{n+1/2} - B_z\rvert_{i-1/2, j+1/2, k}^{n+1/2}}{\Delta x}

   \frac{E_z\rvert_{i, j, k+1/2}^{n+1} - E_z\rvert_{i, j, k+1/2}^{n}}{c^2 \Delta t} =&
    \frac{B_y\rvert_{i+1/2, j, k+1/2}^{n+1/2} - B_y\rvert_{i-1/2, j, k+1/2}^{n+1/2}}{\Delta x}

   & - \frac{B_x\rvert_{i, j+1/2, k+1/2}^{n+1/2} - B_x\rvert_{i, j-1/2, k+1/2}^{n+1/2}}{\Delta y}

   \frac{B_y\rvert_{i+1/2, j, k+1/2}^{n+3/2} - B_y\rvert_{i+1/2, j, k+1/2}^{n+1/2}}{\Delta t} =&
    \frac{E_z\rvert_{i+1, j, k+1/2}^{n+1} - E_z\rvert_{i, j, k+1/2}^{n+1}}{\Delta x}
    - \frac{E_x\rvert_{i+1/2, j, k+1}^{n+1} - E_x\rvert_{i+1/2, j, k}^{n+1}}{\Delta z}

   \frac{B_z\rvert_{i+1/2, j+1/2, k}^{n+3/2} - B_z\rvert_{i+1/2, j+1/2, k}^{n+1/2}}{\Delta t} =&
    \frac{E_x\rvert_{i+1/2, j+1, k}^{n+1} - E_x\rvert_{i+1/2, j, k}^{n+1}}{\Delta y}
    - \frac{E_y\rvert_{i+1, j+1/2, k}^{n+1} - E_y\rvert_{i, j+1/2, k}^{n+1}}{\Delta x}

When using TF/SF technique, first a usual Yee field solver update is applied to the whole grid, regardless of TF and SF regions.
Then a separate stage that we call *incident field solver* is run to modify the calculated values where necessary.
The combined effect of the Yee- and incident field solvers is that Maxwell's equations hold on the whole grid and the correct incident field is generated.
We now proceed to describe how are these values identified and what is the modification necessary.

As mentioned above, values like :math:`E_y\rvert_{i, j+1/2, k}^{n+1}` are stored for the whole Yee grid.
Whether they represent the total or the scattered field, depends on the position of the node relative to the Huygens surface.
To avoid confusion, we use the :math:`E_y\rvert_{i, j+1/2, k}^{n+1}` notation for stored grid values, and :math:`E_y^{other}\left( i \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t \right)` to denote fields at the same time and space position, but not stored long-term.

Since the Maxwell's equations hold in both the TF and SF regions, all Yee solver updates involving only grid values from the same region produced correct values that do not need any further modification.
A correction is only needed for grid values that were calculated using a mix of TF and SF values.
Since the standard Yee solver uses a 2-point central derivative operator, those are a single layer of :math:`\vec E` and :math:`\vec B` values located near :math:`S`.

Taking into account the 0.75 shift inwards used by PIConGPU, denote the :math:`x` position of :math:`S`  as :math:`x_S = (i_S + 0.75) \Delta x`.
Then the grid values to be modified by the incident field solver are :math:`E_y\rvert_{i_S+1, j+1/2, k}^{n+1}`, :math:`E_z\rvert_{i_S+1, j, k+1/2}^{n+1}`, :math:`B_y\rvert_{i_S+1/2, j, k+1/2}^{n+3/2}`, and :math:`B_z\rvert_{i_S+1/2, j+1/2, k}^{n+3/2}`.
(All grid values to the right side of those were calculated using only TF values and all grid values on the left side were calculated using only SF values.)

Consider the update of :math:`E_y\rvert_{i_S+1, j+1/2, k}^{n+1}` performed by a standard Yee solver for each :math:`j, k`.
All terms but :math:`B_z\rvert_{i_S+1/2, j+1/2, k}^{n+1/2}` in this update are in the TF region.
Thus, this value has to be modified by the incident field solver in order to preseve the Maxwell's equations.

To derive the modification necessary, consider a hypothetical Maxwell's-preserving update at this point where all participating values were TF:

.. math::

   & \frac{E_y^{tot}\left( (i_S+1) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t \right) - E_y\rvert_{i_S+1, j+1/2, k}^{n}}{c^2 \Delta t} =
   
   & \frac{B_x\rvert_{i_S+1, j+1/2, k+1/2}^{n+1/2} - B_x\rvert_{i_S+1, j+1/2, k-1/2}^{n+1/2}}{\Delta z} -

   & \frac{B_z\rvert_{i_S+3/2, j+1/2, k}^{n+1/2} - B_z^{tot}\left( (i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t \right)}{\Delta x}

Since :math:`B_z\rvert_{i_S+1/2, j+1/2, k}^{n+1/2}` is an SF and by definition of TF and SF,

.. math::

   & B_z^{tot}\left( (i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t \right) =

   & B_z\rvert_{i_S+1/2, j+1/2, k}^{n+1/2} + B_z^{inc}\left( (i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t \right)

Substituting it into the update equation and regrouping the terms yields:
   
.. math::   
   & E_y^{tot}((i_S+1) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t) = E_y\rvert_{i_S+1, j+1/2, k}^{n}

   & + c^2 \Delta t \left(
   \frac{B_x\rvert_{i_S+1, j+1/2, k+1/2}^{n+1/2} - B_x\rvert_{i_S+1, j+1/2, k-1/2}^{n+1/2}}{\Delta z} - \right.

   & \left. \frac{B_z\rvert_{i_S+3/2, j+1/2, k}^{n+1/2} - (B_z\rvert_{i_S+1/2, j+1/2, k}^{n+1/2} + B_z^{inc}((i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t))}{\Delta x} \right)
   
   & = E_y\rvert_{i_S+1, j+1/2, k}^{n} + c^2 \Delta t \left(
   \frac{B_x\rvert_{i_S+1, j+1/2, k+1/2}^{n+1/2} - B_x\rvert_{i_S+1, j+1/2, k-1/2}^{n+1/2}}{\Delta z} - \right.
   
   & \left. \frac{B_z\rvert_{i_S+3/2, j+1/2, k}^{n+1/2} - B_z\rvert_{i_S+1/2, j+1/2, k}^{n+1/2}}{\Delta x} \right)
   
   & + \frac{c^2 \Delta t}{\Delta x} B_z^{inc}((i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t)
   
   & = E_y\rvert_{i_S+1, j+1/2, k}^{n+1} + \frac{c^2 \Delta t}{\Delta x} B_z^{inc}((i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t)

Thus, in the incident field stage we have to apply the following update to the grid value calculated by a normal Yee solver
:

.. math::   

    E_y\rvert_{i_S+1, j+1/2, k}^{n+1} += \frac{c^2 \Delta t}{\Delta x} B_z^{inc}((i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t)

Grid value :math:`E_z\rvert_{i_S+1, j, k+1/2}^{n+1}` is also located in the TF region and with a similar derivation the update for it is

.. math::   

    E_z\rvert_{i_S+1, j, k+1/2}^{n+1} += - \frac{c^2 \Delta t}{\Delta x} B_y^{inc}((i_S+1/2) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1/2) \Delta t)

Values :math:`B_y\rvert_{i_S+1/2, j, k+1/2}^{n+3/2}`, and :math:`B_z\rvert_{i_S+1/2, j+1/2, k}^{n+3/2}` are in the SF region.
For them the Yee solver update includes one term from the TF region, :math:`E_z\rvert_{i_S, j, k+1/2}^{n+1}` and :math:`E_y\rvert_{i_S, j+1/2, k}^{n+1}` respectively.
Making a similar replacement of an SF value as a difference between a TF value and the incident field value and regrouping, the following update must be applied:

.. math::   

    & B_y\rvert_{i_S+1/2, j, k+1/2}^{n+3/2} += - \frac{\Delta t}{\Delta x} E_z^{inc}((i_S+1) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1) \Delta t)
    
    & B_z\rvert_{i_S+1/2, j+1/2, k}^{n+3/2} += \frac{\Delta t}{\Delta x} E_y^{inc}((i_S+1) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t)

The derivation for the :math:`x_{max}` boundary can be done in a similar fashion.
Denote the position of :math:`S` as :math:`x_S = (i_{S, max} + 0.25) \Delta x`.
Note that our 0.75 cells inwards shift of :math:`S` is symmetrical in terms of distance.
It implies that the Yee grid incides along :math:`x` are not fully symmetric between the two sides of each bondary.
The update scheme for :math:`x_{max}` is:

.. math::   

    & E_y\rvert_{i_{S, max}, j+1/2, k}^{n+1} += - \frac{c^2 \Delta t}{\Delta x} B_z^{inc}((i_{S, max}+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t)

    & E_z\rvert_{i_{S, max}, j, k+1/2}^{n+1} += \frac{c^2 \Delta t}{\Delta x} B_y^{inc}((i_{S, max}+1/2) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1/2) \Delta t)

    & B_y\rvert_{i_{S, max}+1/2, j, k+1/2}^{n+3/2} += \frac{\Delta t}{\Delta x} E_z^{inc}((i_{S, max}+1) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1) \Delta t)

    & B_z\rvert_{i_{S, max}+1/2, j+1/2, k}^{n+3/2} += - \frac{\Delta t}{\Delta x} E_y^{inc}((i_{S, max}+1) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t)

Multiple Boundaries
^^^^^^^^^^^^^^^^^^^

In the general case, :math:`S` is comprised of several axis-aligned boundary hyperplanes, 6 planes in 3D, and 4 lines in 2D.

The scheme described above is symmetric for all axes.
In case incident field is coming from multiple boundaries, the updates are in principle the same.
They can generally be treated as sequential application of the single-boundary case.

Applying TF/SF for each boundary affects the derivatives in the normal direction relative to the boundary.
For the standard Yee solver, a single layer of :math:`\vec E` and :math:`\vec B` values along the boundary is affected.
Along other directions, we update all grid values that are internal relative to the Huygens surface.
In case a "corner" grid node is near several boundaries, it is updated in all the respective applications of TF/SF.

General Case FDTD
-----------------

The same principle as for the Yee solver can be applied to any FDTD-type field solver.
Same as above, we consider the case of :math:`x_{min}` boundary and :math:`E_y` field component.
The other boundaries and components are treated symmetrically.

We now apply a general-case spatial-only finite-difference operator to calculate derivatives along :math:`x`.
Such operators on the Yee grid naturally have an antisymmetry of coefficients in :math:`x` relative to the evaluation point.
The antisymmetry is not critical for the following description, but is present in the FDTD solvers implemented and allow simplifying the formulations, and so we assume it.
For :math:`dB_z/dx` such an operator has the following general form:

.. math::

   & \partial_x B_z(i\Delta x, (j+1/2)\Delta y, k\Delta z, (n+1/2)\Delta t) = 

   & \frac{1}{\Delta x} \sum_{ii=0}^{m_x} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \alpha_{ii, jj, kk} \left( B_z\rvert_{i+(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} - B_z\rvert_{i-(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} \right)

Note that there is also typically a symmetry of coefficients along :math:`y` and :math:`z`: :math:`\alpha_{ii, jj, kk} = \alpha_{ii, -jj, kk}`, :math:`\alpha_{ii, jj, kk} = \alpha_{ii, jj, -kk}` but it is not significant for TF/SF.
The derivative operator used by the standard Yee solver has :math:`m_x = m_y = m_z = 0, \alpha_{0, 0, 0} = 1`.

Same as before, denote the :math:`x` position of :math:`S` as :math:`x_S = (i_S + 0.75) \Delta x`.
In order to stay within the grid, we require that :math:`i_S \geq m_x`.
The incident field solver has to update the grid values of :math:`E_y` for which calculating :math:`dB_z/dx` involves a mix of TF and SF values.
These values can be present in both the TF and SF regions around :math:`S`:

.. math::

   & E_{TF} = \{ E_y\rvert_{i_S+1+ii, j+1/2, k}^{n+1} : ii = 0, 1, \ldots, m_x \}

   & E_{SF} = \{ E_y\rvert_{i_S+1-ii, j+1/2, k}^{n+1} : ii = 1, 2, \ldots, m_x \}

Take a node in the TF region :math:`E_y\rvert_{i_0, j+1/2, k}^{n+1} \in E_{TF}` (:math:`i_0 = i_S+1+ii_0` for some :math:`ii_0 \in [0, m_x]`).
During the FDTD update of this node, the :math:`dB_z/dx` operator is calculated:

.. math::

   & \partial_x B_z(i_0\Delta x, (j+1/2)\Delta y, k\Delta z, (n+1/2)\Delta t) = 

   & \frac{1}{\Delta x} \sum_{ii=0}^{m_x} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \alpha_{ii, jj, kk} \left( B_z\rvert_{i_0+(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} - B_z\rvert_{i_0-(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} \right)

We split the outer sum over :math:`ii` into two parts:

.. math::

   & \partial_x B_z(i_0\Delta x, (j+1/2)\Delta y, k\Delta z, (n+1/2)\Delta t) =
 
   &  \frac{1}{\Delta x} \sum_{ii=0}^{i_0-i_S-2} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \alpha_{ii, jj, kk} \left( B_z\rvert_{i_0+(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} - B_z\rvert_{i_0-(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} \right) +
   
   &  \frac{1}{\Delta x} \sum_{ii=i_0-i_S-1}^{m_x} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \alpha_{ii, jj, kk} \left( B_z\rvert_{i_0+(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} - B_z\rvert_{i_0-(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} \right)

The first sum over :math:`ii \in [0, i_0-i_S-2]` only uses :math:`B_z` grid values in the TF region (the minimal index in :math:`x` used is :math:`B_z\rvert_{i_S+3/2, j+jj+1/2, k+kk}^{n+1/2}` for :math:`ii = i_0-i_S-2`).
Note that if :math:`i_0-i_S-2 < 0`, this sum has no terms and is equal to 0; the same applies for the following sums.
Since the :math:`E_y` value in question is also a TF, these terms do not require any action by incident field solver.
The remaining sum over :math:`ii \in [i_0-i_S-1, m_x]` contains differences of a TF value and an SF value.
Each of the latter ones requires a term in the incident field solver update of :math:`E_y\rvert_{i_0, j+1/2, k}^{n+1}`.

Performing the same kind of substitution and regrouping demonstrated above for the standard Yee solver yields

.. math::

   & E_y^{tot}(i_0 \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t) =  E_y\rvert_{i_0, j+1/2, k}^{n+1} +
   
   & \frac{c^2 \Delta t}{\Delta x} \sum_{ii=i_0-i_S-1}^{m_x} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \left( \alpha_{ii, jj, kk} \cdot \right.
   
   & \left. B_z^{inc}((i_0-(ii+1/2)) \Delta x, (j+jj+1/2) \Delta y, (k+kk) \Delta z, (n+1/2) \Delta t) \right)
   
Thus, we apply the following update for each grid value :math:`E_y\rvert_{i_0, j+1/2, k}^{n+1} \in E_{TF}`:

.. math::

   & E_y\rvert_{i_0, j+1/2, k}^{n+1} +=

   & \frac{c^2 \Delta t}{\Delta x} \sum_{ii=i_0-i_S-1}^{m_x} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \left( \alpha_{ii, jj, kk} \cdot \right.
   
   & \left. B_z^{inc}((i_0-(ii+1/2)) \Delta x, (j+jj+1/2) \Delta y, (k+kk) \Delta z, (n+1/2) \Delta t) \right)

For values in SF the treatment is similar.
For a node :math:`E_y\rvert_{i_0, j+1/2, k}^{n+1} \in E_{SF}` (:math:`i_0 = i_S+1-ii_0` for some :math:`ii_0 \in [1, m_x]`) we apply :math:`dB_z/dx` operator and split the outer sum the same way:

.. math::

   & \partial_x B_z(i_0\Delta x, (j+1/2)\Delta y, k\Delta z, (n+1/2)\Delta t) =

   &  \frac{1}{\Delta x} \sum_{ii=0}^{i_S-i_0} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \alpha_{ii, jj, kk} \left( B_z\rvert_{i_0+(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} - B_z\rvert_{i_0-(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} \right) +
   
   &  \frac{1}{\Delta x} \sum_{ii=i_S+1-i_0}^{m_x} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \alpha_{ii, jj, kk} \left( B_z\rvert_{i_0+(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} - B_z\rvert_{i_0-(ii+1/2), j+jj+1/2, k+kk}^{n+1/2} \right)

The first sum only has values in the SF region, and the second sum contains differences of TF and SF values.
Note that now :math:`E_y\rvert_{i_0, j+1/2, k}^{n+1}` is in the SF region and so we express the whole update as for SF:

.. math::

   & E_y^{scat}(i_0 \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t) = E_y\rvert_{i_0, j+1/2, k}^{n+1} +
   
   & \frac{c^2 \Delta t}{\Delta x} \sum_{ii=i_S+1-i_0}^{m_x} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \left( \alpha_{ii, jj, kk} \cdot \right.
   
   & \left. B_z^{inc}((i_0+(ii+1/2)) \Delta x, (j+jj+1/2) \Delta y, (k+kk) \Delta z, (n+1/2) \Delta t) \right)

Thus, we apply the following update for each grid value :math:`E_y\rvert_{i_0, j+1/2, k}^{n+1} \in E_{SF}`:

.. math::

   & E_y\rvert_{i_0, j+1/2, k}^{n+1} +=

   & \frac{c^2 \Delta t}{\Delta x} \sum_{ii=i_S+1-i_0}^{m_x} \sum_{jj=-m_y}^{m_y} \sum_{kk=-m_z}^{m_z} 
   \left( \alpha_{ii, jj, kk} \cdot \right.
   
   & \left. B_z^{inc}((i_0+(ii+1/2)) \Delta x, (j+jj+1/2) \Delta y, (k+kk) \Delta z, (n+1/2) \Delta t) \right)

Other field components, axes and directions are treated in a similar way.

Example: 4th Order FDTD
^^^^^^^^^^^^^^^^^^^^^^^

For example, consider the :ref:`4th order FDTD <model-AOFDTD>` and :math:`x_{min}` boundary.
Its derivative operator has :math:`m_x = 1`, :math:`m_y = m_z = 0`, :math:`\alpha_{0, 0, 0} = 27/24`, :math:`\alpha_{1, 0, 0} = -1/24`.
Three layers of :math:`E_y` are updated, the first in the SF region and the latter two in the TF region:

.. math::

   & E_y\rvert_{i_S, j+1/2, k}^{n+1} += \frac{c^2 \Delta t}{\Delta x} \left( -\frac{1}{24} B_z^{inc}\left( (i_S+3/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t \right) \right)

   & E_y\rvert_{i_S + 1, j+1/2, k}^{n+1} += \frac{c^2 \Delta t}{\Delta x} \left( \frac{27}{24} B_z^{inc}\left( (i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t \right) \right.
   
   & \left. -\frac{1}{24} B_z^{inc}\left( (i_S-1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t \right) \right)

   & E_y\rvert_{i_S + 2, j+1/2, k}^{n+1} += \frac{c^2 \Delta t}{\Delta x} \left( -\frac{1}{24}  B_z^{inc}\left( (i_S+1/2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1/2) \Delta t \right) \right)


Updates of :math:`E_z` are done in a similar fashion:

.. math::

   & E_z\rvert_{i_S, j, k+1/2}^{n+1} += -\frac{c^2 \Delta t}{\Delta x} \left( -\frac{1}{24} B_y^{inc}\left( (i_S+3/2) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1/2) \Delta t \right) \right)

   & E_z\rvert_{i_S + 1, j, k+1/2}^{n+1} += -\frac{c^2 \Delta t}{\Delta x} \left( \frac{27}{24} B_y^{inc}\left( (i_S+1/2) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1/2) \Delta t \right) \right.
   
   & \left. -\frac{1}{24} B_y^{inc}\left( (i_S-1/2) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1/2) \Delta t \right) \right)

   & E_z\rvert_{i_S + 2, j, k+1/2}^{n+1} += -\frac{c^2 \Delta t}{\Delta x} \left( -\frac{1}{24}  B_y^{inc}\left( (i_S+1/2) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1/2) \Delta t \right) \right)

Three layers of :math:`B_y` are updated, the first two in the SF region and the last one in the TF region:

.. math::

   & B_y\rvert_{i_S-1/2, j, k+1/2}^{n+3/2} += -\frac{\Delta t}{\Delta x} \left( -\frac{1}{24} E_z^{inc}\left( (i_S+1) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1) \Delta t \right) \right)

   & B_y\rvert_{i_S + 1/2, j, k+1/2}^{n+3/2} += -\frac{\Delta t}{\Delta x} \left( \frac{27}{24} E_z^{inc}\left( (i_S+1) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1) \Delta t \right) \right.
   
   & \left. -\frac{1}{24} E_z^{inc}\left( (i_S+2) \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1) \Delta t \right) \right)

   & B_y\rvert_{i_S + 3/2, j, k+1/2}^{n+3/2} += -\frac{\Delta t}{\Delta x} \left( -\frac{1}{24}  E_z^{inc}\left( i_S \Delta x, j \Delta y, (k+1/2) \Delta z, (n+1) \Delta t \right) \right)

Finally, updates of :math:`B_z` are as follows:

.. math::

   & B_z\rvert_{i_S-1/2, j+1/2, k}^{n+3/2} += \frac{\Delta t}{\Delta x} \left( -\frac{1}{24}  E_y^{inc}\left( (i_S+1) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t \right) \right)

   & B_z\rvert_{i_S + 1/2, j+1/2, k}^{n+3/2} += \frac{\Delta t}{\Delta x} \left( \frac{27}{24} E_y^{inc}\left( (i_S+1) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t \right) \right.
   
   & \left. -\frac{1}{24} E_y^{inc}\left( (i_S+2) \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t \right) \right)

   & B_z\rvert_{i_S + 3/2, j+1/2, k}^{n+3/2} += \frac{\Delta t}{\Delta x} \left( -\frac{1}{24}  E_y^{inc}\left( i_S \Delta x, (j+1/2) \Delta y, k \Delta z, (n+1) \Delta t \right) \right)

Usage
-----

The TF/SF field generation can be configured in :ref:`incidentField.param <usage-params-core>`.
The position of the Huygens surface is set as a gap relative to the interface of the field absorber and internal area.
Note that using field solvers other than Yee requires a positive gap along the boundaries with a non-zero source, gap value depending on the stencil width along the boundary axis.
This is checked at run time.

Consider a case when both :math:`E^{inc}(x, y, z, t)` and  :math:`\vec B^{inc}(x, y, z, t)` are theoretically present, but only one of them is known in explicit form.

In this case one can try using TF/SF with only the modified known field set as incident and the other one set to 0.
The interpretation of the result is assisted by the equivalence theorem, and in particular Love and Schelkunoff equivalence principles [Harrington2001]_ [Balanis2012]_.
Having :math:`\vec E^{inc}(x, y, z, t) = \vec 0` means only electric current :math:`\vec J` would be impressed on :math:`S`.
Taking into account no incident fields in the SF region, the region is effectively a perfect magnetic conductor.
Likewise, having :math:`\vec B^{inc}(x, y, z, t) = \vec 0` corresponds to only magnetic current and effectively a perfect electric conductor in the SF region.
To generate the expected field amplitude inside the area, the only non-zero source field has to be adjusted.
In the simple plane wave case, the adjustment is to set the amplitude of the present field twice as large, as demonstrated in [Rengarajan2000]_.
In the general case, it appears unclear how to calculate such an adjustment.
Also note, that within the plain wave approximation the unknown field could have alternatively been calculated from the known one.

References
----------
.. [Potter2017]
        M. Potter, J.-P. Berenger
        *A Review of the Total Field/Scattered Field Technique for the FDTD Method*
        FERMAT, Volume 19, Article 1 (2017)

.. [Taflove2005]
        A. Taflove
        *Computational electrodynamics: the finite-difference time-domain method*
        Artech house (2005)

.. [Harrington2001]
        R.F. Harrington
        *Time-Harmonic Electromagnetic Fields*
        McGraw-Hill (2001)

.. [Balanis2012]
        C.A. Balanis
        *Advanced Engineering Electromagnetics*
        John Wiley & Sons (2012)

.. [Rengarajan2000]
        S.R. Rengarajan, Y. Rahmat-Samii
        *The field equivalence principle: illustration of the establishment of the non-intuitive null fields*
        IEEE Antennas and Propagation Magazine, Volume 42, No. 4 (2000)
