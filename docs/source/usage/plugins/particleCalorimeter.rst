.. _usage-plugins-particleCalorimeter:

Particle Calorimeter
--------------------

A binned calorimeter of the amount of kinetic energy per solid angle and energy-per-particle.

The solid angle bin is solely determined by the particle's momentum vector and not by its position, so we are emulating a calorimeter at infinite distance.

The calorimeter takes into account all existing particles as well as optionally all particles which have already left the global simulation volume.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`openPMD API <install-dependencies>` is compiled in.

.param file
^^^^^^^^^^^

The spatial calorimeter resolution can be customized and in :ref:`speciesDefinition.param <usage-params-plugins>`.
Therein, a species can be also be marked for detecting particles leaving the simulation box.

.cfg file
^^^^^^^^^

All options are denoted exemplarily for the photon (``ph``) particle species here.


================================== =========================================================================================
PIConGPU command line option       Description
================================== =========================================================================================
``--ph_calorimeter.period``        The ouput periodicity of the plugin.
                                   A value of ``100`` would mean an output at simulation time step *0, 100, 200, ...*.
``--ph_calorimeter.file``          Output file suffix. Put unique name if same ``species + filter`` is used multiple times.
``--ph_calorimeter.ext``           openPMD filename extension. This determines the backend to be used by openPMD.
                                   Default is ``.h5`` for HDF5 output.
``--ph_calorimeter.filter``        Use filtered particles. All available filters will be shown with ``picongpu --help``
``--ph_calorimeter.numBinsYaw``    Specifies the number of bins used for the yaw axis of the calorimeter.
                                   Defaults to ``64``.
``--ph_calorimeter.numBinsPitch``  Specifies the number of bins used for the pitch axis of the calorimeter.
                                   Defaults to ``64``.
``--ph_calorimeter.numBinsEnergy`` Specifies the number of bins used for the energy axis of the calorimeter.
                                   Defaults to ``1``, i.e. there is no energy binning.
``--ph_calorimeter.minEnergy``     Minimum detectable energy in keV.
                                   Ignored if ``numBinsEnergy`` is ``1``.
                                   Defaults to ``0``.
``--ph_calorimeter.maxEnergy``     Maximum detectable energy in keV.
                                   Ignored if ``numBinsEnergy`` is ``1``.
                                   Defaults to ``1000``.
``--ph_calorimeter.logScale``      En-/Disable logarithmic energy binning.  Allowed values: ``0`` for disable, ``1`` enable.
``--ph_calorimeter.openingYaw``    opening angle yaw of the calorimeter in degrees.
                                   Defaults to the maximum value: ``360``.
``--ph_calorimeter.openingPitch``  opening angle pitch of the calorimeter in degrees.
                                   Defaults to the maximum value: ``180``.
``--ph_calorimeter.posYaw``        yaw coordinate of the calorimeter position in degrees.
                                   Defaults to the +y direction: ``0``.
``--ph_calorimeter.posPitch``      pitch coordinate of the calorimeter position in degrees.
                                   Defaults to the +y direction: ``0``.
================================== =========================================================================================

Coordinate System
^^^^^^^^^^^^^^^^^

.. image:: ../../../images/YawPitch.png
   :alt: orientation of axes

Yaw and pitch are `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_ defining a point on a sphere's surface, where ``(0,0)`` points to the ``+y`` direction here. In the vicinity of ``(0,0)``, yaw points to ``+x`` and pitch to ``+z``.

**Orientation detail:** Since the calorimeters' three-dimensional orientation is given by just two parameters (``posYaw`` and ``posPitch``) there is one degree of freedom left which has to be fixed.
Here, this is achieved by eliminating the Euler angle roll.
However, when ``posPitch`` is exactly ``+90`` or ``-90`` degrees, the choice of roll is ambiguous, depending on the yaw angle one approaches the singularity.
Here we assume an approach from ``yaw = 0``.

Tuning the spatial resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the spatial bin size is chosen by dividing the opening angle by the number of bins for yaw and pitch respectively.
The bin size can be tuned by customizing the mapping function in ``particleCalorimeter.param``.


Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

each energy bin times each coordinate bin allocates two counter (``float_X``) permanently and on each accelerator for active and outgoing particles.

Host
""""

as on accelerator.

Output
^^^^^^

The calorimeters are stored in openPMD-files in the ``simOutput/<species>_calorimeter/`` directory.
The file names are ``<species>_calorimeter_<sfilter>_<timestep>.<file_ending>``.

Depending on whether energy binning is enabled the dataset is two or three dimensional.
The dataset has the following attributes:


================== =============================================
Attribute          Description
================== =============================================
``unitSI``         scaling factor for energy in calorimeter bins
``maxYaw[deg]``    half of the opening angle yaw.
``maxPitch[deg]``  half of the opening angle pitch.
``posYaw[deg]``    yaw coordinate of the calorimeter.
``posPitch[deg]``  pitch coordinate of the calorimeter.
                   If energy binning is enabled:
``minEnergy[keV]`` minimal detectable energy.
``maxEnergy[keV]`` maximal detectable energy.
``logScale``       boolean indicating logarithmic scale.
================== =============================================

The output in each bin is given in Joule.
Divide by energy value of the bin for a unitless count per bin.

The output uses a custom geometry.
Since the openPMD API does currently not (yet) support reading from datasets with a custom-name geometry, this plugin leaves the default geometry ``"cartesian"`` instead of specifying something like ``"calorimeter"``.
If the output is 2D, cells are defined by `[pitch, yaw]` in degrees.
If the output is 3D, cells are defined by `[energy bin, pitch, yaw]` where the energy bin is given in keV. Additionally, if `logScale==1`, then the energy bins are on a logarithmic scale whose start and end can be read from the custom attributes `minEnergy[keV]` and `maxEnergy[keV]` respectively.

.. note::

   This plugin is a multi plugin. 
   Command line parameters can be used multiple times to create e.g. dumps with different dumping period.
   In the case where an optional parameter with a default value is explicitly defined the parameter will be always passed to the instance of the multi plugin where the parameter is not set.
   e.g. 

   .. code-block:: bash

      --ph_calorimeter.period 128 --ph_calorimeter.file calo1 --ph_calorimeter.filter all
      --ph_calorimeter.period 1000 --ph_calorimeter.file calo2 --ph_calorimeter.filter all --ph_calorimeter.logScale 1 --ph_calorimeter.minEnergy 1

   creates two plugins:
 
   #. calorimeter for species ph each 128th time step **with** logarithmic energy binning.
   #. calorimeter for species ph each 1000th time step **without** (this is the default) logarithmic energy binning.

.. attention::

   When using the plugin multiple times for the same combination of ``species`` and ``filter``, you *must* provide a unique ``file`` suffix.
   Otherwise output files will overwrite each other, since only ``species``, ``filter`` and ``file`` suffix are encoded in it.

   An example use case would be two (or more) calorimeters for the same species and filter but with differing position in space or different binning, range, linear and log scaling, etc.

Analysis Tools
^^^^^^^^^^^^^^

The first bin of the energy axis of the calorimeter contains all particle energy less than the minimal detectable energy whereas the last bin contains all particle energy greater than the maximal detectable energy.
The inner bins map to the actual energy range of the calorimeter.

To easily access the data, you can use our python module located in ``lib/python/picongpu/extra/plugins/data/calorimeter.py``

.. code:: python

import numpy as np
import matplotlib.pyplot as plt

from calorimeter import particleCalorimeter

# setup access to data
calObj = particleCalorimeter("./simOutput/e_calorimeter/e_calorimeter_all_%T.bp")

# last bin contains overflow 
selected_energy_bin = -1

plt.title("selected energy: >{:.1f} keV".format(calObj.getEnergy()[selected_energy_bin]), fontsize=18)

plt.pcolormesh(calObj.getYaw(), calObj.getPitch(), calObj.getData(2000)[selected_energy_bin, :, :])

plt.xlabel(calObj.detector_params["axisLabels"][-1], fontsize=18)
plt.ylabel(calObj.detector_params["axisLabels"][-2], fontsize=18)

cb = plt.colorbar()
cb.set_label("energy", fontsize=18)

plt.show()
