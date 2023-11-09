.. _usage-plugins-binningPlugin:
#######
Binning
#######

This binning plugin is a flexible binner for particles properties.
Users can 
    - Define their own axes
    - Define their own quantity which is binned
    - Choose which species which are used for the binning
    - Choose how frequently they want the binning to be executed
    - Choose if the binning should be time averaging or normalized by bin volume
    - Write custom output to file, for example other quantites related to the simulation which the user is interested in
    - Execute multiple binnings at the same time

User Input
==========
Users can set up their binning in the ``binningSetup.param`` file. After setup, PIConGPU needs to be recompiled.

.. attention::

   Unlike other plugins, the binning plugin doesn't provide any runtime configuration. To set up binning, users need to define it in the param file and then recompile.

A binner is created using the ``addBinner()`` function, which describes the configuration options available to the user to set up the binning.
Multiple binnings can be run at the same time by simply calling ``addBinner()`` multiple times with different parameters.

.. doxygenclass:: picongpu::plugins::binning::BinningCreator
    :members: addBinner

A most important parts of defining a binning are the axes (the axes of the histogram which define the bins) and the deposited quantity (the quantity to be binned). 
Both of these are described using the "Functor Description".


Functor Description
-------------------
The basic building block for the binning plugin is the Functor Description object, and it is used to describe both axes and the deposited quantity.
It describes the particle properties which we find interesting and how we can calculate/get this property from the particle.
A functor description is created using createFunctorDescription.

.. doxygenfunction:: picongpu::plugins::binning::createFunctorDescription


Functor 
^^^^^^^
The functor needs to follow the signature shown below. This provides the user access to the particle object and with information about the :ref:`domain <usage/plugins/binningPlugin:Domain Info>`.

.. code-block:: c++

    auto myFunctor = [] ALPAKA_FN_ACC(auto const& domainInfo, auto const& worker, auto const& particle) -> returnType
    {
        // fn body
        return myParameter;
    };

The return type is defined by the user. 

Domain Info
"""""""""""
Enables the user to find the location of the particle (in cells) in the simulation domain. Contains 

.. doxygenclass:: picongpu::plugins::binning::DomainInfo
    :members:

The global and local offsets can be understood by lookng at the `PIConGPU domain definitions <https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions>`_.



Dimensionality and units
^^^^^^^^^^^^^^^^^^^^^^^^
Users can specify the units of their functor output using a 7 dimensional array. Each element of the array corresponds an SI base unit, and the value stored in that index is the exponent of the unit.
The dimensional base quantities are defined as in ``SIBaseUnits_t`` following the international system of quantities (ISQ).
If no units are given, the quantity is assumed to be dimensionless.

.. code-block:: c++

    std::array<double, 7> momentumDimension{};
    momentumDimension[SIBaseUnits::length] = 1.0;
    momentumDimension[SIBaseUnits::mass] = 1.0;
    momentumDimension[SIBaseUnits::time] = -1.0;

.. doxygenenum:: picongpu::traits::SIBaseUnits::SIBaseUnits_t


Axis
----
Axis is a combination of a :ref:`functor description <usage/plugins/binningPlugin:Functor Description>` and an  :ref:`axis splitting <usage/plugins/binningPlugin:Axis Splitting>`
These are brought together by createAxis functions, depending on what kind of an axis you want.
The name used in the functor description is used as the name of the axis for openPMD. 

.. attention::

   The return type of the functor as specified in the functor description is required to be the same as the type of the range (min, max).

Currently implemented axis types 
    - Linear Axis 

.. doxygenclass:: picongpu::plugins::binning::axis::LinearAxis

.. - Equally spaced bins between min and max. Total number of bins equal to n_bins.
..            axis::createLinear(cellY_splitting, cellPositionYDescription);


Binning can be done over an arbitrary number of axes, by creating a tuple of all the axes. Limited by memory depending on number of bins in each axis.

Axis Splitting
^^^^^^^^^^^^^^
Defines the axis range and how it is split into bins.
In the future this plugin will support other ways to split the domain, eg. using the binWidth or by auto-selecting the parameters.

.. doxygenclass:: picongpu::plugins::binning::axis::AxisSplitting
    :members:


Range
"""""

.. doxygenclass:: picongpu::plugins::binning::axis::Range
    :members:

Species
-------
PIConGPU species which should be used in binning. 
Species can be instances of a species type or a particle species name as a PMACC_CSTRING. For example, 

.. code-block:: c++
    
    auto electronsObj = PMACC_CSTRING("e"){};

.. note::

   Some parameters (axes and species) are given in the form of tuples. These are just a collection of objects and are of arbitrary size. 
   Users can make a tuple by using the ``createTuple()`` function and passing in the objects as arguments.


Deposited Quantity
------------------
Quantity to be deposited is simply a :ref:`functor description <usage/plugins/binningPlugin:Functor Description>`. 


Notify period
-------------
Set the periodicity of the output. Follows the period syntax defined :ref:`here <usage/plugins:period syntax>`.

Dump Period
-----------
Defines the number of notify steps to accumulate over. Note that this is not accumulating over actual PIC iterations, but over the notify periods.   
If time averaging is enabled, this is also the period to do time averaging over.
For example a value of 10 means that after every 10 notifies, an accumulated file will be written out. 
If PIConGPU exits before executing 10 notifies, then there will be no output.
The plugin dumps on every notify if this is set to either 0 or 1.

Time Averaging
--------------
When dumping the accumulated output, whether or not to divide by the dump period, i.e. do a time averaging.

.. attention::

    The user needs to set a dump period to enable time averaging. 

Normalize by Bin Volume
-----------------------
Since it is possible to have non-uniformly sized axes, it makes sense to normalize the binned quantity by the bin volume to enable a fair comparison between bins.


writeOpenPMDFunctor
-------------------
Users can also write out custom output to file, for example other quantites related to the simulation which the user is interested in.
This is a lambda with the following signature. 

.. code-block:: c++

    [=](::openPMD::Series& series, ::openPMD::Iteration& iteration, ::openPMD::Mesh& mesh) -> void 

.. note::

   Make sure to capture by copy only, as the objects defined in the param file are not kept alive



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
