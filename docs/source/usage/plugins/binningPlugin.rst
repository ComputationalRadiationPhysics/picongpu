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
	- Choose if the binning should be time averaging or normalized by bin volume
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

.. doxygenfunction:: picongpu::plugins::binning::createFunctorDescription


Functor
^^^^^^^
The functor needs to follow the signature shown below. This provides the user access to the particle or field object and with information about the :ref:`domain <usage/plugins/binningPlugin:Domain Info>`.

The return type is defined by the user.

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

For fields:

.. code-block:: c++

	auto myFieldFunctor = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& field) -> returnType
	{
		// fn body
		return myParameter;
	};

For fields with extra data:

.. code-block:: c++

	auto myFieldWithExtraDataFunctor = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& field1, auto const& field2, auto const& extraData) -> returnType
	{
		// fn body
		return myParameter;
	};

.. note::

	Fields and extra data may be tuples with multiple elements which are unpacked and passed to the user-defined functors. Therefore, it is the responsibility of the user to ensure that their functors have an appropriate number of arguments to match the provided tuples.

Domain Info
"""""""""""
Enables the user to find the location of the particle or field (in cells) in the simulation domain. Contains

For particle binning, the ``DomainInfo`` class contains:

.. doxygenclass:: picongpu::plugins::binning::DomainInfoBase
	:members:

For field binning, the ``DomainInfo`` class additionally contains:

.. doxygenclass:: picongpu::plugins::binning::DomainInfo< BinningType::Field >
	:members:

The global and local offsets can be understood by looking at the `PIConGPU domain definitions <https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions>`_.

For particle binning, the particle position is obtained at cell center precision by default. To get sub-cell precision or SI units, use optional template parameters with ``getParticlePosition<DomainOrigin, PositionPrecision, PositionUnits>``.

For field binning, the field ``DomainInfo`` additionally holds the localCellIndex in the supercell and has a method to ``getCellIndex<DomainOrigin, PositionUnits>`` to get the current cell index being binned relative to an origin (global, total, local).


Dimensionality and units
^^^^^^^^^^^^^^^^^^^^^^^^
Users can specify the units of their functor output using a 7 dimensional array. Each element of the array corresponds to an SI base unit, and the value stored in that index is the exponent of the unit.
The dimensional base quantities are defined in ``SIBaseUnits_t`` following the international system of quantities (ISQ).
If no units are given, the quantity is assumed to be dimensionless.

.. code-block:: c++

	std::array<double, numUnits> momentumDimension{};
	momentumDimension[SIBaseUnits::length] = 1.0;
	momentumDimension[SIBaseUnits::mass] = 1.0;
	momentumDimension[SIBaseUnits::time] = -1.0;

.. doxygenenum:: picongpu::SIBaseUnits::SIBaseUnits_t


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
They can then create a FilteredSpecies object which contains the species and the filter.

.. code-block:: c++

	auto myFilter = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& particle) -> bool
	{
		bool binningEnabled = true;
		// fn body
		return binningEnabled;
	};

	auto fitleredElectrons = FilteredSpecies{electronsObj, myFilter};

.. note::

			Species are passed to addParticleBinner in the form of a tuple. This is just a collection of Species and FilteredSpecies objects (the tuple can be a mixture of both) and is of arbitrary size.
			Users can make a species tuple by using the ``createSpeciesTuple()`` function and passing in the objects as arguments.

Fields
------
PIConGPU fields which should be used in field binning.
Fields can be instances of a field type. For example,

.. code-block:: c++

	DataConnector& dc = Environment<>::get().DataConnector();
	auto fieldsTuple = createTuple(
	dc.get<FieldB>(FieldB::getName())->getDeviceDataBox(),
	dc.get<FieldTmp>(FieldTmp::getUniqueId(0))->getDeviceDataBox());

Fields are passed to addFieldBinner in the form of a tuple. This is just a collection of field objects and is of arbitrary size.
Users can make a fields tuple by using the ``createTuple()`` function and passing in the objects as arguments.

.. note::

	It is possible to have an empty tuple for fields when doing field binning, in which case the functor will be called with no fields. This may be useful if you are passing in extra data and want field traversal over it.

Deposited Quantity
------------------
Quantity to be deposited is simply a :ref:`functor description <usage/plugins/binningPlugin:Functor Description>`.
The signature of quantity functors is

.. code-block:: c++

	auto myQuantityFunctor = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, ...) -> returnType

Extra Data
----------
Users can pass extra data to the functors if additional information is required by the kernels to do binning.
	This extra data is created using the `createTuple()` function and then passed into the binning creator.
Extra data is passed to ``addParticleBinner`` or ``addFieldBinner`` functions in the form of a tuple. This tuple is just a collection of objects and is of arbitrary size.
Users can make a tuple of the extra data by using the ``createTuple()`` function and passing in the objects as arguments.

For example, to pass extra data to a particle binner:

.. code-block:: c++

	auto extraData = createTuple(extraData1, extraData2);
	addParticleBinner(axisTuple, speciesTuple, depositionFunctor, extraData);

Similarly, for a field binner:

.. code-block:: c++

	auto extraData = createTuple(extraData1, extraData2);
	addFieldBinner(axisTuple, fieldsTuple, depositionFunctor, extraData);

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
Defines the number of notify steps to accumulate over. Note that this is not accumulating over actual PIC iterations, but over the notify periods.
If time averaging is enabled, this is also the period to do time averaging over.
For example a value of 10 means that after every 10 notifies, an accumulated file will be written out.
If PIConGPU exits before executing 10 notifies, then there will be no output.
The plugin dumps on every notify if this is set to either 0 or 1. This is the default behaviour.

Time Averaging
--------------
When dumping the accumulated output, whether or not to divide by the dump period, i.e. do a time averaging. Enabled by default.

.. attention::

	The user needs to set a dump period to enable time averaging.

Normalize by Bin Volume
-----------------------
Since it is possible to have non-uniformly sized axes, it makes sense to normalize the binned quantity by the bin volume to enable a fair comparison between bins. Enabled by default.


Binning Particles Leaving the Simulation Volume
-----------------------------------------------

.. doxygenenum:: picongpu::plugins::binning::ParticleRegion

By default, only particles within the simulation volume are binned by the particle binner. However, users can modify this behavior to include or exclusively bin particles that are leaving the global simulation volume.
This can be configured using the ``enableRegion`` and ``disableRegion`` options with the regions defined by the ``ParticleRegion`` enum.

.. attention::

Users must carefully configure the notify period when using the binning plugin for leaving particles. The plugin bins particles leaving the global simulation volume at every timestep (except 0) after particles are pushed, regardless of the notify period.
If the plugin is not notified at every timestep, this can cause discrepancies between the binning process and time-averaged data or histogram dumps, which follow the notify period.
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

OpenPMD Output
==============
The binning outputs are stored in HDF5 files in ``simOutput/binningOpenPMD/`` directory.

The files are named as ``<binnerOutputName>_<timestep>.h5``.

The OpenPMD mesh is call "Binning".

The outputs in written in SI units.

If normalization is enabled, the output is normalized by the bin volume.

The output histogram has 2 bins more in each dimension than the user-defined ``nBins`` in that dimension, to deal with under and overflow.

The number of bin edges written out for an axis is one more than the user-defined ``nBins``. These represent the bins in [min,max]. Since there are actually ``nBins + 2`` bins, two edges are not written out.
These are the first and last edge, corresponding to the overflow bins, and they have the value of -inf and + inf.


=========================== ==========================================================
Attribute                   Description
=========================== ==========================================================
``unitSI``                  Scaling factor for the deposited quantity to convert to SI
``<axisName>_bin_edges``    The edges of the bins of an axis in SI units
``<axisName>_units``        The units of an axis
=========================== ==========================================================


Example binning plugin usage: Laser Wakefield electron spectrometer
==================================================================

The :ref:`LWFA example <LWFA-example>`  contains a sample binning plugin setup to calculate an in-situ electron spectrometer.
The kinetic energy of the electrons :math:`E = (\gamma - 1) m_o c^2` is plotted on axis 1 and the direction of the electrons :math:`\theta = \mathrm{atan}(p_x/p_y)` is plotted on axis 2.
The charge :math:`Q` moving in the bin direction :math:`\theta` at the bin energy :math:`E` is calculated for each bin.
The charge is normalized to the bin volume :math:`\Delta E \cdot \Delta \theta`.
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

Further examples of the binning plugin can be found in the default ``binningSetup.param`` and the param file associated with the atomic physics example.
