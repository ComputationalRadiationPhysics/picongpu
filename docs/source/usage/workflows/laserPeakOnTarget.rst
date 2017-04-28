.. _usage-workflows-laserPeakOnTarget:

Setting the Laser Initialization Cut-Off
----------------------------------------

.. sectionauthor:: Axel Huebl

Laser profiles for simulation are modeled with a temporal envelope.
A common model assumes a Gaussian intensity distribution over time which by definition never sets to zero, so it needs to be cut-off to a reasonable range.

In :ref:`laser.param <usage-params-core>` each profile implements the cut-off to start (and end) initializing the laser profile via a parameter ``PULSE_INIT`` :math:`t_\text{init}` (sometimes also called ``RAMP_INIT``).
:math:`t_\text{init}` is given in units of the ``PULSE_LENGTH`` :math:`\tau` which is implemented *laser-profile dependent* (but usually as :math:`\sigma_I` of the standard Gaussian of intensity :math:`I=E^2`).

For a fixed target in distance :math:`d` to the lower :math:`y=0` boundary of the simulation box, the maximum intensity arrives at time:

.. math::

   t_\text{laserPeakOnTarget} = \frac{t_\text{init} \cdot \tau}{2} + \frac{d}{c_0}

or in terms of discrete time steps :math:`\Delta t`:

.. math::

   \text{step}_\text{laserPeakOnTarget} = \frac{t_\text{laserPeakOnTarget}}{\Delta t}.

.. note::
   Moving the spatial plane of initialization of the laser pulse via ``initPlaneY`` does not change the formula above.
   The implementation covers this spatial offset during initialization.
