.. _usage-plugins-binningPlugin:

#######
Binning
#######

This binning plugin is a flexible binner for particles and field properties.
Users can

- Define their own axes
- Define their own quantity which is binned
- Choose which species are used for particle binning
- Choose which fields are used for field binning
- Choose how frequently they want the binning to be executed
- Choose how many times the binned quantity should be combined over time before dumping
- Write custom output to file, for example other quantities related to the simulation which the user is interested in
- Execute multiple binnings at the same time
- Pass extra parameters as a tuple, if additional information is required by the kernels to do binning.
- Define a host-side hook to execute code before binning at each notify step, for example to fill ``FieldTmp``.

User Input
==========
Users can set up their binning in the ``binningSetup.param`` file. After setup, PIConGPU needs to be recompiled.

.. attention::

  Unlike other plugins, the binning plugin doesn't provide any runtime configuration. To set up binning, users need to define it in the param file and then recompile.

A binner is created using the `addParticleBinner()` or `addFieldBinner()` functions, which describe the configuration options available to the user to set up particle and field binning respectively.
Multiple binnings can be run at the same time by simply calling `addParticleBinner()` or `addFieldBinner()` multiple times with different parameters.

.. doxygenclass:: picongpu::plugins::binning::BinningCreator
  :members: addParticleBinner, addFieldBinner

The most important parts of defining a binning are the axes (the axes of the histogram which define the bins) and the deposited quantity (the quantity to be binned).
Both of these are described using the "Functor Description".


Functor Description
-------------------
The basic building block for the binning plugin is the Functor Description object, and it is used to describe both axes and the deposited quantity.
It describes the properties which we find interesting and how we can calculate/get this property from the particle or field.
A functor description is created using createFunctorDescription.

.. doxygenfunction:: picongpu::plugins::binning::createFunctorDescription(FunctorType functor, std::string const& name, std::array<double, numUnits> units)


Functor
^^^^^^^

The functor is run on the GPU and constitutes the means by which the user can execute code on device while binning. This may be used to calculate the deposited quantity or to calculate the axis values.

The functor needs to follow the signature shown below. This provides the user access to the particle or field object and with information about the :ref:`domain <usage/plugins/binningPlugin:Domain Info>`.

The return type is defined by the user.

The first two arguments are always the worker and the domainInfo. For particle binning this is followed by the particle object, and for field binning this is followed by the field object (if any). If extra data is passed, this is also passed as arguments.
It is the responsibility of the user to ensure that their functors have an appropriate number and types of arguments to match the provided tuples. A few examples are shown below.

For particles:

.. code-block:: c++

  auto myParticleFunctor = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& particle) -> returnType
  {
    // fn body
    return myParameter;
  };

For particles with two extra data paramters:

.. code-block:: c++

  auto myParticleWithExtraDataFunctor = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& particle, auto const& extraData1, auto const& extraData2) -> returnType
  {
    // fn body
    return myParameter;
  };

For one field:

.. code-block:: c++

  auto myFieldFunctor = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& field) -> returnType
  {
    // fn body
    return myParameter;
  };

For two fields with one extra data parameter:

.. code-block:: c++

  auto myFieldWithExtraDataFunctor = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& field1, auto const& field2, auto const& extraData) -> returnType
  {
    // fn body
    return myParameter;
  };

.. note::

  Fields and extra data may be tuples with multiple elements which are unpacked and passed to the user-defined functors. Therefore, it is the responsibility of the user to ensure that their functors have an appropriate number of arguments to match the provided tuples.

Dimensionality and units
^^^^^^^^^^^^^^^^^^^^^^^^
Users can specify the units of their functor output using a 7 dimensional array. Each element of the array corresponds to an SI base unit, and the value stored in that index is the exponent of the unit.
The dimensional base quantities are defined in ``SIBaseUnits_t`` following the international system of quantities (ISQ).
If no units are given, the quantity is assumed to be dimensionless.

.. literalinclude:: ../../../../share/picongpu/tests/compile2/include/picongpu/param/binningSetup.param
   :language: c++
   :start-after: doc-include-start: units
   :end-before: doc-include-end: units
   :dedent:

.. doxygenenum:: picongpu::SIBaseUnits::SIBaseUnits_t


Domain Info
-----------
The binning plugin passes the domain info object to functors as a parameter to give the user access to some useful PIConGPU quantities inside the functor. 
Importantly, it enables the user to find the location of the particle or field in the simulation domain.

The ``DomainInfo`` class contains:

.. doxygenclass:: picongpu::plugins::binning::DomainInfoBase
  :members:

The global and local offsets can be understood by looking at the `PIConGPU domain definitions <https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions>`_.

Particle binning
^^^^^^^^^^^^^^^^

For particle binning, we can use the ``getParticlePosition`` function. 

.. code-block:: cpp

    getParticlePosition<DomainOrigin, PositionPrecision, PositionUnits>()

The particle position has the default precision set to cell precision, and the position is returned in cell units by default.

To request a different origin, precision, or unit system, use the template parameters as described below.

Field binning
^^^^^^^^^^^^^

For field binning, ``DomainInfo`` additionally stores the ``localCellIndex`` within the supercell we can use the ``getCellIndex`` function.

.. code-block:: cpp

    getCellIndex<DomainOrigin, PositionUnits>()

This returns the current cell index being binned relative to the selected origin.

To obtain the exact position of a field inside the cell, relative to the cell index, use the ``FieldPosition`` trait.

Domain origin
^^^^^^^^^^^^^
  
The reference origin for returned positions is selected with ``DomainOrigin``:

The available origins are:

- ``TOTAL``
Absolute origin of the simulation. This includes regions that are no longer part of the current global volume, for example because they have moved out of the sliding window.

- ``GLOBAL``
Origin of the current sliding window, i.e. the currently simulated domain across all GPUs, excluding guard cells.

- ``LOCAL``
Origin of the local domain on the current GPU, excluding guard cells.

- ``MOVING_WINDOW``
Origin relative to the sliding window origin. This origin starts moving only once the sliding window moves and is not discretized to the cell grid.

- ``LOCAL_WITH_GUARDS``
Origin of the local domain on the current GPU, including guard cells. This setting is in particular used to access field data for the current cell with `getCellIndex`.
  
Position precision
^^^^^^^^^^^^^^^^^^
  
For particle positions, the precision is selected with ``PositionPrecision``.

The available precisions are:

- ``CELL``
Returns the particle position at cell precision, i.e. as a cell index.

- ``SUB_CELL``
Returns the particle position with sub-cell precision, i.e. as the cell index plus the particle position inside the cell in the range ``[0,1)``. The result is therefore a floating-point position in units of cells.

Position units
^^^^^^^^^^^^^^

The output units for positions are selected with ``PositionUnits``.

The available units are:

- ``SI``
  Returns the position in SI units.

- ``PIC``
  Returns the position in PIC units.

- ``CELL``
  Returns the position in units of cells. The result is integral for ``PositionPrecision::CELL`` and floating point for ``PositionPrecision::SUB_CELL``.

.. note::

    Using positions in SI/PIC units introduces floating point numerical errors and may be especially problematic when using the ``TOTAL`` origin with moving window, because floating-point precision decreases as the distance from the origin increases.

Axis
----
Axis is a combination of a :ref:`functor description <usage/plugins/binningPlugin:Functor Description>` and an  :ref:`axis splitting <usage/plugins/binningPlugin:Axis Splitting>`.
These are brought together by createAxis functions, depending on what kind of an axis you want.
The name used in the functor description is used as the name of the axis for openPMD.

.. attention::

  The return type of the functor as specified in the functor description is required to be the same as the type of the range (min, max).

Currently implemented axis types
  - Linear Axis
  - Log Axis

.. doxygenclass:: picongpu::plugins::binning::axis::LinearAxis

.. - Equally spaced bins between min and max. Total number of bins equal to n_bins.
..            axis::createLinear(cellY_splitting, cellPositionYDescription);

.. doxygenclass:: picongpu::plugins::binning::axis::LogAxis

.. - Logarithmically spaced bins between min and max. Total number of bins equal to n_bins.
..            axis::createLog(cellY_splitting, cellPositionYDescription);


.. attention::

  The log axis suffers from floating point precision errors, so for certain combinations of min, max, nBins and values to be binned the bin index calculated might be off by one. If using an integral log axis, be careful of what you are doing.


Binning can be done over an arbitrary number of axes, by creating a tuple of all the axes. Limited by memory depending on number of bins in each axis.

Axis Splitting
^^^^^^^^^^^^^^
Defines the axis range and how it is split into bins. Bins are defined as closed open intervals from the lower edge to the upper edge.
In the future, this plugin will support other ways to split the domain, eg. using the binWidth or by auto-selecting the parameters.

.. doxygenclass:: picongpu::plugins::binning::axis::AxisSplitting
  :members:


Range
"""""

.. doxygenclass:: picongpu::plugins::binning::axis::Range
  :members:

.. note::

  Axes are passed to addParticleBinner or addFieldBinner grouped in a tuple. This is just a collection of axis objects and is of arbitrary size.
  Users can make a tuple for axes by using the ``createTuple()`` function and passing in the axis objects as arguments.

Species
-------
PIConGPU species which should be used in particle binning.
Species can be instances of a species type or a particle species name as a PMACC_CSTRING. For example,

.. code-block:: c++

  auto electronsObj = PMACC_CSTRING("e"){};

Optionally, users can specify a filter to be used with the species. This is a predicate functor, i.e. it is a functor with a signature as described above and which returns a boolean. If the filter returns true it means the particle is included in the binning.
They can then create a ``FilteredSpecies`` object which contains the species and the filter.

.. literalinclude:: ../../../../share/picongpu/tests/compile2/include/picongpu/param/binningSetup.param
   :language: c++
   :start-after: doc-include-start: filter
   :end-before: doc-include-end: filter
   :dedent:

.. note::

  Species are passed to addParticleBinner in the form of a tuple. This is just a collection of Species and FilteredSpecies objects (the tuple can be a mixture of both) and is of arbitrary size.
  Users can make a species tuple by using the ``createSpeciesTuple()`` function and passing in the objects as arguments.

Fields
------
PIConGPU fields which should be used in field binning.
Fields must be instances of the ``FieldInfo`` type.

.. doxygenstruct:: picongpu::plugins::binning::FieldInfo
  :members:

For example,

.. literalinclude:: ../../../../share/picongpu/tests/compile2/include/picongpu/param/binningSetup.param
   :language: c++
   :start-after: doc-include-start: fieldTuple
   :end-before: doc-include-end: fieldTuple
   :dedent:

Fields are passed to addFieldBinner in the form of a tuple. This is just a collection of ``FieldInfo`` and is of arbitrary size.
Users can make a fields tuple by using the ``createTuple()`` function and passing in the objects as arguments.
The functors receive the fields in the form of the field data box, which is the field data on the current GPU including the guard cells, in the order they are passed in the tuple.

.. literalinclude:: ../../../../share/picongpu/tests/compile2/include/picongpu/param/binningSetup.param
   :language: c++
   :start-after: doc-include-start: fieldBox
   :end-before: doc-include-end: fieldBox
   :dedent:

.. note::

  It is possible to have an empty tuple for fields when doing field binning, in which case the functor will be called with no fields. This may be useful if you are passing in extra data and want field traversal over it.

.. note::

  It is possible to have field information available while doing particle binning as well. Users can simply pass in ``FieldInfo`` objects in the extra data tuple, and the functor will be called with the field information as well.

Deposited Quantity
------------------
Quantity to be deposited is simply a :ref:`functor description <usage/plugins/binningPlugin:Functor Description>`.
The signature of quantity functors is

.. code-block:: c++

  auto myQuantityFunctor = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, ...) -> returnType

This option makes it evident that the binning is more than just about creating histograms. While histograms are a common use case, the binning plugin allows for using various binary operations to combine various quantities within bins and not just noting the frequencies of occurrences. This means that users can define custom quantities to be combined in each bin, such as charge, energy, momentum, or any other property of interest. The flexibility of the functor description enables users to specify exactly what and how they want to combine data in the bins.
For example, you might want to combine the total charge of particles within each bin, or the average kinetic energy of particles in a specific region. The deposited quantity functor provides the mechanism to calculate and return these values, which are then combined in the corresponding bins during the simulation.
By default the deposited quantity is added for each bin, but this is configurable by the user by setting a binary operation for the :ref:`reduce <usage/plugins/binningPlugin:Reduction>`.

Extra Data
----------
Users can pass extra data to the functors if additional information is required by the kernels to do binning.
Extra data is passed to ``binningCreator``'s ``addParticleBinner`` or ``addFieldBinner`` functions in the form of a tuple. This tuple is just a collection of objects and is of arbitrary size.
Users can make a tuple of the extra data by using the ``createTuple()`` function and passing in the objects as arguments.

.. note::

  Ensure that the functors have an appropriate number of arguments to match the provided tuples. The extra data may be tuples with multiple elements which are unpacked and passed to the user-defined functors.

Host-side hook
---------------
Users can define a host-side hook to execute code before binning starts for a notify step.
This hook is set using the ``setHostSideHook`` method of the ``BinningCreator`` class and takes a callable (lambda/functor/std::function) which returns void.

.. code-block:: c++

  []() -> void{
    // host-side code to be executed before binning
  }

For example, this can be used to pre-process data on the host side, such as filling temporary fields before they are used in field binning.

Notify period
-------------
Set the periodicity of the output. Follows the period syntax defined :ref:`here <usage/plugins:period syntax>`. Notifies every time step by default.

Dump Period
-----------
Defines the number of notify steps to reduce over. Note that this is not a reduction over actual PIC iterations, but over the notify periods.
For example a value of 10 means that after every 10 notifies, an reduced file will be written out.
If PIConGPU exits before executing 10 notifies, then there will be no output.
The plugin dumps on every notify if this is set to either 0 or 1. This is the default behaviour.


Binning Particles Leaving the Simulation Volume
-----------------------------------------------

.. doxygenenum:: picongpu::plugins::binning::ParticleRegion

By default, only particles within the simulation volume are binned by the particle binner. However, users can modify this behavior to include or exclusively bin particles that are leaving the global simulation volume.
This can be configured using the ``enableRegion`` and ``disableRegion`` options with the regions defined by the ``ParticleRegion`` enum.

.. attention::

  Users must carefully configure the notify period when using the binning plugin for leaving particles. The plugin bins particles leaving the global simulation volume at every timestep (except 0) after particles are pushed, regardless of the notify period.
  If the plugin is not notified at every timestep, this can cause discrepancies between the binning process and histogram dumps, which follow the notify period.
  Additionally, the binning plugin is first notified at timestep 0, allowing users to bin initial conditions. However, leaving particles are first binned at timestep 1, after the initial particle push.
  Therefore, users should consider setting the notify period’s start at timestep 1, depending on their specific needs.

writeOpenPMDFunctor
-------------------
Users can also write out custom output to file, for example other quantities related to the simulation which the user is interested in.
This is a lambda with the following signature, and is set using the ``setWriteOpenPMDFunctor`` method.

.. code-block:: c++

  [=](::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh) -> void

.. note::

  Make sure to capture by copy only, as the objects defined in the param file are not kept alive


OpenPMD JSON Configuration
--------------------------
Users can set a json configuration string for the OpenPMD output format using the ``setOpenPMDJsonCfg`` setter method.

Configuration Options Summary
-----------------------------
The following table summarizes the configuration options available for the binner object returned by `addParticleBinner()` or `addFieldBinner()`. These methods allow for fine-tuning the behavior of the binning plugin.

.. list-table:: Binning Configuration Options
   :widths: 30 50 20
   :header-rows: 1

   * - Option
     - Description
     - Default Value
   * - ``setNotifyPeriod``
     - Sets the periodicity of the output. Follows the period syntax defined :ref:`here <usage/plugins:period syntax>`.
     - ``"1"`` (every time step)
   * - ``setDumpPeriod``
     - Defines the number of notify steps to reduce over before dumping. A value of 0 or 1 means dump on every notify.
     - ``0u``
   * - ``setHostSideHook``
     - Sets a host-side hook (lambda/functor) to execute code before binning at each notify step.
     - An empty lambda ``[]{}``.
   * - ``enableRegion`` / ``disableRegion``
     - Configures binning for particles inside (``Bounded``) or leaving (``Leaving``) the simulation volume. This is only for particle binning.
     - ``Bounded`` region is enabled.
   * - ``setOpenPMDWriteFunctor``
     - Sets a functor to write custom data to the openPMD output file.
     - An empty functor.
   * - ``setOpenPMDJsonCfg``
     - Sets a JSON configuration string for the openPMD output format.
     - ``"{}"``
   * - ``setOpenPMDExtension``
     - Sets the file extension for the openPMD output.
     - Default OpenPMD extension (e.g., ``.bp`` or ``.h5``).
   * - ``setOpenPMDInfix``
     - Sets the infix for file names in openPMD output. The default uses ``%06T`` for the time step.
     - ``"_%06T."``

OpenPMD Output
==============
The binning outputs are stored in HDF5 files in ``simOutput/binningOpenPMD/`` directory.

The files are named as ``<binnerOutputName>_<timestep>.h5``.

The OpenPMD mesh is call "Binning".

The outputs in written in SI units.

The output histogram has 2 bins more in each dimension than the user-defined ``nBins`` in that dimension, to deal with under and overflow.

The number of bin edges written out for an axis is one more than the user-defined ``nBins``. These represent the bins in [min,max]. Since there are actually ``nBins + 2`` bins, two edges are not written out.
These are the first and last edge, corresponding to the overflow bins, and they have the value of -inf and + inf.


=========================== =========== ==========================================================
Attribute                   Hierarchy   Description
=========================== =========== ==========================================================
``unitSI``                  Mesh        Scaling factor for the deposited quantity to convert to SI
``<axisName>_bin_edges``    Mesh        The edges of the bins of an axis in SI units
``<axisName>_units``        Mesh        The units of an axis
``reduceCounter``           Iteration   Reduction operations per dump period
=========================== =========== ==========================================================

Reduction
=========
The binning plugin provides flexible options for reducing data within bins. This is useful when a dump period is set. Instead of simply adding values in a bin, users can utilize other PMacc operations as well. This includes operations such as addition, minimum, maximum, bitwise AND, bitwise OR, and bitwise XOR.
This flexibility enables users to tailor the reduction process to their specific needs. For instance, you might want to track the minimum or maximum value of a certain property within each bin.
A list of available PMacc operations can be found under `/include/pmacc/math/operation`. Only operations defining the `getMPI_Op` function and the `AlpakaAtomicOp` and `NeutralElement` traits are supported.

.. note::

  The operation is only viable if it has a corresponding MPI reduction operation, a neutral element, and a cooresponding alpaka operation. Only binary atomics on arithmetic types are supported. In particular CAS operations are not supported.


The reduce operation is passed as a template parameter to the `createBinner` functions.
For example, to use the maximum operation for reduction, you would pass the corresponding PMacc operation as a template parameter:

.. literalinclude:: ../../../../share/picongpu/tests/compile2/include/picongpu/param/binningSetup.param
   :language: c++
   :start-after: doc-include-start: atomics
   :end-before: doc-include-end: atomics
   :dedent:

Similarly, you can use other PMacc operations such as `pmacc::math::operation::Min`, `pmacc::math::operation::Add`, `pmacc::math::operation::BitwiseAnd`, `pmacc::math::operation::BitwiseOr`, `pmacc::math::operation::BitwiseXor`, etc.


Example binning plugin usage: Laser Wakefield electron spectrometer
===================================================================

The :ref:`LWFA example <LWFA-example>`  contains a sample binning plugin setup to calculate an in-situ electron spectrometer.
The kinetic energy of the electrons :math:`E = (\gamma - 1) m_o c^2` is plotted on axis 1 and the direction of the electrons :math:`\theta = \mathrm{atan}(p_x/p_y)` is plotted on axis 2.
The charge :math:`Q` moving in the bin direction :math:`\theta` at the bin energy :math:`E` is calculated for each bin.
Such spectrometers are a common tool in plasma based electron acceleration experiments [Kurz2018]_.

.. note::

  Please note that if you specify the SI units of an axis, e.g. via ``energyDimension`` in the LWFA example,
  PIConGPU automatically converts to the internal unit system, but then also expects SI-compliant values for the axis range
  (in the case of energy Joules).


To read the electron spectrometer data in python, one could load and plot it like this:

.. code:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib.colors import LogNorm
  import openpmd_api as io
  import scipy.constants as const

  # access openPMD series of eSpec
  series = io.Series("./LWFA/simOutput/binningOpenPMD/eSpec_%T.h5", access=io.Access_Type.read_only)

  last_iter = list(series.iterations)[-1]
  it = series.iterations[last_iter]
  espec_h = it.meshes['Binning'][io.Mesh_Record_Component.SCALAR]

  # load data
  espec = espec_h[:,:]
  series.flush()

  # convert to SI units and make positve (electrons have a negative charge)
  espec *= espec_h.get_attribute('unitSI') * -1

  # get axes (they are already in the correct SI unit)
  E_bins = espec_h.get_attribute('Energy_bin_edges')
  theta_bins = espec_h.get_attribute('pointingXY_bin_edges')

  # convert C/J/rad -> C/MeV/mrad
  convert_C_per_Joule_per_rad_to_pC_per_MeV_per_mrad = 1./1e-12 * const.elementary_charge/1e6 * 1/1e3

  # plot
  plt.pcolormesh(np.array(E_bins) / const.elementary_charge / 1e6,
                      np.array(theta_bins) / 0.001,
                      espec[1:-1, 1:-1] * convert_C_per_Joule_per_rad_to_pC_per_MeV_per_mrad,
                      norm=LogNorm(), cmap=plt.cm.inferno)
  cb = plt.colorbar()

  plt.xlabel(r"$E \, \mathrm{[MeV]}$", fontsize=18)
  plt.xticks(fontsize=14)

  plt.ylabel(r"$\theta \, \mathrm{[mrad]}$", fontsize=18)
  plt.yticks(fontsize=14)

  cb.set_label(r"$\frac{\mathrm{d}^2 Q}{\mathrm{d} E \mathrm{d}\theta} \, \mathrm{[pC/MeV/mrad]}$", fontsize=20)
  for i in cb.ax.get_yticklabels():
      i.set_fontsize(14)

  plt.tight_layout()
  plt.show()


References
----------

.. [Kurz2018]
        T. Kurz, J.P. Couperus, J.M. Krämer, et al.
        *Calibration and cross-laboratory implementation of scintillating screens for electron bunch charge determination*,
        Review of Scientific Instruments (2018),
        https://doi.org/10.1063/1.5041755

More examples of binning plugin usage
=====================================

Further examples of the binning plugin can be found in the atomic physics example at ``share/picongpu/examples/AtomicPhysics/include/picongpu/param/binningSetup.param`` and the ``share/picongpu/tests/compile2`` test.
