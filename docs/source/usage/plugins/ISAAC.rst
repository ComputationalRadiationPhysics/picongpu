.. _usage-plugins-ISAAC:

ISAAC
-----

This is a plugin for the in-situ library ISAAC for a live rendering and steering of PIConGPU simulations.
For Installation notes see :ref`our install notes <install-dependencies>`. 

.cfg file
^^^^^^^^^

========================= =========================================================================
Command line option       Description
========================= =========================================================================
``--isaac.period N``      Sets up, that every _N_ th timestep an image will be rendered.
                          This parameter can be changed later with the controlling client.
``--isaac.name NAME``     Sets the _NAME_ of the simulation, which is shown at the client.
``--isaac.url URL``       _URL_ of the needed and running isaac server.
                          Host names and IPs are supported.
``--isaac.port PORT``     _PORT_ of the isaac server.
                          The default value is ``2458`` (for the in-situ plugins), but may be needed to changed for tunneling reasons or if more than one server shall run on the very same hardware.
``--isaac.width WIDTH``   Setups the _WIDTH_ and _HEIGHT_ of the created image(s).
``--isaac.height HEIGHT`` Default is ``1024x768``.
``--isaac.direct_pause``  If activated ISAAC will pause directly after the simulation started.
                          Useful for presentations or if you don't want to miss the beginning of the simulation.
``isaac.quality QUALITY`` Sets the _QUALITY_ of the images, which are compressed right after creation.
                          Values between ``1`` and ``100`` are possible.
                          The default is ``90``, but ``70`` does also still produce decent results.
========================= =========================================================================

Running and steering a simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First of all you need to build and run the isaac server somewhere.
Probably the best idea is to use the login or head node of your cluster, otherwise PIConGPU but also the clients will not be able to find it.
More informations about building and starting ISAAC can be found here: https://github.com/ComputationalRadiationPhysics/isaac/blob/master/INSTALL.md.

The ISAAC Plugin has an `isaac.param`, which specifies which fields and particles are rendered.
This can be edited (in your local paramSet), but at runtime also an arbitrary amount of fields (in ISAAC called _sources_) can be deactivated.
At default every field and every known species are rendered.

The most important settings for ISAAC are ``--isaac.period``, ``--isaac.name`` and ``--isaac.url``. A possibly addition for your submission tbg file could be ``--isaac.period 1 --isaac.name !TBG_jobName --isaac.url YOUR_SERVER``, where the tbg variables ``!TBG_jobName`` is used as name and ``YOUR_SERVER`` needs to be set up by yourself.
See more options here: https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/docs/TBG_macros.cfg .

Functor Chains
^^^^^^^^^^^^^^

One of the most important features of ISAAC are the **Functor Chains**.
As most sources (including fields and species) may not be suited for a direct rendering or even full negative (like the electron density field), the functor chains enable you to change the domain of your field source-wise. A date will be read from the field, the functor chain applied and then **only the x-component** used for the classification and later rendering of the scene.
Multiply functors can be applied successive with the Pipe symbol ``|``.
The possible functors are at default:

* **mul** for a multiplication with a constant value.
  For vector fields you can choose different value per component, e.g. ``mul(1,2,0)``, which will multiply the x-component with 1, the y-component with 2 and the z-component with 0.
  If less parameters are given than components exists, the last parameter will be used for all components without an own parameter.
* **add** for adding a constant value, which works the same as ``mul(...)``.
* **sum** for summarizing all available components.
  Unlike ``mul(...)`` and ``add(...)`` this decreases the dimension of the data to ``1``, which is a scalar field.
  You can exploit this functor to use a different component than the x-component for the classification, e.g. with ``mul(0,1,0) | sum``.
  This will first multiply the x- and z-component with 0, but keep the y-component and then merge this to the x-component.
* **length** for calculating the length of a vector field.
  Like `sum` this functor reduces the dimension to a scalar field, too. However ``mul(0,1,0) | sum`` and ``mul(0,1,0) | length`` do not do the same.
  As ``length`` does not know, that the x- and z-component are 0 an expensive square root operation is performed, which is slower than just adding the components up.
* **idem** does nothing, it just returns the input data.
  This is the default functor chain.

Beside the functor chains the client allows to setup the weights per source (values greater than 6 are more useful  for PIConGPU than the default weights of 1), the classification via transfer functions, clipping, camera steering and to switch the render mode to iso surface rendering.
Furthermore interpolation can be activated.
However this is quite slow and most of the time not needed for non-iso-surface rendering.

Example renderings
^^^^^^^^^^^^^^^^^^

.. image:: https://raw.githubusercontent.com/ComputationalRadiationPhysics/isaac/master/example_renderings/picongpu_wakefield_8.png
   :alt: Laser Wakefield

.. image:: https://raw.githubusercontent.com/ComputationalRadiationPhysics/isaac/master/example_renderings/picongpu_kelvin_helmholtz_4.png
   :alt: Kelvin Helmholtz Instability

.. image:: https://raw.githubusercontent.com/ComputationalRadiationPhysics/isaac/master/example_renderings/picongpu_weibel_1.png
   :alt: Weibel Instability

