.. SPDX-FileCopyrightText: PIConGPU contributors
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _model-lasers:

Analytic Expressions for the 3D Laser Profiles
==============================================

.. sectionauthor:: Klaus Steiniger


Among others, PIConGPU offers the ``GaussianPulse`` and ``DispersivePulse`` profiles to model complex, Gaussian-like laser pulse profiles in simulations.

The ``GaussianPulse`` profile allows modelling standard Gaussian pulses defined by a Gaussian distribution of laser energy along the longitudinal and transverse directions.
Furthermore, it allows including higher-order transverse modes via Laguerre-Gaussian modes [Pausch2022]_.

The ``DispersivePulse`` profile allows modelling standard Gaussian pulses which feature dispersions, i.e. frequency specific modifications of the laser's spectral phase, up to third order.
That is, time delay (TD), angular dispersion (AD), spatial dispersion (AD), group-delay dispersion (GDD), and third-order dispersion (TOD) are self-consistently taken into account in the propagation of these pulses.
The profile assumes a Gaussian shape for the laser's spectrum, see the computation of ``envFreqExp`` in ``DispersivePulse.hpp``.
The respective in-focus values of these dispersions can be provided as parameters.
However, the electric field values in time domain are computed from the field's values in frequency domain by a discrete Fourier transform.
Therefore, it is possible to use any other shape of the laser's spectrum by modifying the profile's source code.
See ``amp()`` function in `include/picongpu/fields/incidentField/profiles/DispersivePulse.hpp <https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/include/picongpu/fields/incidentField/profiles/DispersivePulse.hpp>`_.

A very concise description of the equations used in the ``GaussianPulse`` and ``DispersivePulse`` profiles are provided in the following.
Please also refer to the in-code documentation of these and the other profiles in order to obtain more profile-specific information.
Furthermore, refer to :ref:`Workflows / Adding Laser <usage-workflows-addLaser>` in order to see how to add a laser to a PIConGPU simulation.



Definitions
-----------
Following [Steiniger2024]_, the electric field in frequency-space domain is assumed to be polarized along :math:`x` and defined as

.. math::
  \hat{\vec E}(\vec r, \Omega) = \hat E_\mathrm{A}(\vec r, \Omega) e^{-\imath \varphi(\vec r, \Omega)}\vec{\mathrm e}_x\,,

where :math:`\Omega=2\pi\nu` is the angular frequency for frequency :math:`\nu` and :math:`\vec r` the position considered, :math:`\hat E_\mathrm{A}` is the spectral amplitude and
:math:`\varphi=\tfrac{\Omega}{c} \vec{\mathrm e}_\Omega \cdot \vec r`
the spectral phase of the pulse,
with :math:`\vec{\mathrm e}_\Omega` being the propagation direction of frequency :math:`\Omega`.
Dispersions are assumed to occur only in the plane spanned by the direction of pulse propagation :math:`\vec{\mathrm e}_z`, being equal to the direction of propagation of the central frequency :math:`\Omega_0`, and polarization :math:`\vec{\mathrm e}_x`, i.e. :math:`\vec{\mathrm e}_\Omega \cdot \vec{\mathrm e}_y = 0 \forall \Omega`.
This implies that the spectral phase does not vary along the :math:`y`-direction in focus, i.e.

.. math::
  \varphi |_{z=0} = \varphi(x,\Omega) |_{z=0} = k_x(\Omega) \cdot x\,,

where :math:`k_x = \tfrac{\Omega}{c} \vec{\mathrm e}_\Omega \cdot \vec{\mathrm e}_x = -\tfrac{\Omega}{c}\sin\theta` with :math:`\theta=\theta(\Omega)` being the angle enclosed by the propagation directions of frequency :math:`\Omega` and the central laser frequency :math:`\Omega_0`.



In the following, the spectral amplitude is assumed to be separable in a spectrum :math:`\epsilon_\Omega`, as well as transverse envelopes :math:`\epsilon_x` and :math:`\epsilon_y`

.. math::
  \hat E_{\mathrm A} = \epsilon_\Omega \epsilon_x \epsilon_y\,.


.. math::
  \epsilon_\Omega(\Omega)& = e^{-\frac{\tau_0^2}{4}(\Omega-\Omega_0)^2}\,, &
    \tau_0& = \tau_\mathrm{FWHM,I} / \sqrt{2 \ln 2} \\
  \epsilon_x(x)& = e^{-\frac{\left[x-x_0(\Omega)\right]^2}{w_{0,x}^2}}\,, &
    w_{0,x}& = w_{\mathrm{FWHM,I},x} / \sqrt{2 \ln 2} \\
  \epsilon_y(y)& = e^{-\frac{y^2}{w_{0,y}^2}}\,, &
    w_{0,y}& = w_{\mathrm{FWHM,I},y} / \sqrt{2 \ln 2} \\

:math:`x_0=x_0(\Omega)` is the center position of the spatial distribution of frequency :math:`\Omega` along the polarization direction in the focus.


3D field in frequency-space domain
----------------------------------

Prerequisite: Huygens' integral in the Fresnel approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[Siegman1986]_ shows on p. 633 and p. 636 formulas for the propagation of paraxial beams in 3D and 2D, respectively.
They have an equal kernel

.. math::
  \hat K_s(s^\prime, z, \Omega) = \sqrt{\frac{\Omega}{2\pi c}}
    \frac{e^{\imath\frac{\pi}{4}}}{\sqrt{z}}
    e^{-\imath \frac{\Omega}{2 c z} (s-s^\prime)^{2}}\,,

such that propagation can be computed as

.. math::
  & \text{(2D)}& \hat E_x(x, z, \Omega)& =
      e^{-\imath\frac{\Omega}{c}z}
      \int\limits_{-\infty}^{\infty}
        \hat K_x(x^\prime, z, \Omega) \hat{E}(x^\prime, z=0, \Omega) \, \mathrm dx^\prime
    \qquad \clubsuit \label{eq::Fresnel2D} \\
  & \text{(3D)}& \hat{E}(x, y, z, \Omega)& =
      e^{-\imath\frac{\Omega}{c}z}
      \int\limits_{-\infty}^{\infty} \int\limits_{-\infty}^{\infty}
        \hat K_y(y^\prime, z, \Omega) \hat K_x(x^\prime, z, \Omega)
        \hat{E}(x^\prime, y^\prime, z=0, \Omega) \, \mathrm dx^\prime \mathrm dy^\prime \\
  & \text{(3D)}& \hat E(x, y, z, \Omega)& =
      \epsilon_\Omega e^{-\imath\frac{\Omega}{c}z}
      \int\limits_{-\infty}^{\infty}
        \hat K_x(x^\prime, z, \Omega) \epsilon_{x^\prime}
        \mathrm e^{-\imath \varphi(x^\prime, \Omega)}\, \mathrm dx^\prime
      \int\limits_{-\infty}^{\infty}
        \hat K_y(y^\prime, z, \Omega) \epsilon_{y^\prime} \,\mathrm dy^\prime \\
  & & & = \hat E_x(x, z, \Omega)
      \int\limits_{-\infty}^{\infty}
        \hat K_y(y^\prime, z, \Omega) \epsilon_{y^\prime} \,\mathrm dy^\prime\,. \qquad \bigstar \label{eq::Fresnel3Dsep} \\
  & & & \qquad \text{(provided $\hat E$ is separable and dispersion-free along $y$)}


Derivation
^^^^^^^^^^
[Steiniger2024]_ computed the 2D :math:`\hat E_x(x,z,\Omega)` part of eq. :math:`\bigstar`, i.e. computed :math:`\clubsuit` and thereby omitted the last integral in :math:`\bigstar`.
The result of this last integral can be read off from the known solution for the 2D part, being provided in eq. (6) of [Steiniger2024]_.

.. math::
  \hat E_x(x, z, \Omega) = &\
    \epsilon_\Omega
    \left[ 1 + \frac{z^2}{z_{\mathrm{R},x}^2} \right]^{-1/4}
    e^{-\left[
        x - \left( x_0 - \frac{c}{\Omega_0 w_{0,x}}\alpha z \right)
      \right]^2
      \left[ \frac{1}{w_x(z)^2} + \imath \frac{\Omega}{2c} R_x^{-1}(z) \right]
    }
    \\ &\ \times
    e^{-\imath \frac{\Omega}{c}z}
    e^{\imath \alpha \frac{x}{w_{0,x}} }
    e^{\imath \frac{\alpha^2}{4}\frac{z}{z_{\mathrm R,x}} }
    e^{\imath \frac{1}{2} \arctan\frac{z}{z_{\mathrm R,x}} }
    e^{-\imath \frac{1}{2}GDD_\mathrm{foc}(\Omega-\Omega_0)^2}
    e^{-\imath \frac{1}{6}TOD_\mathrm{foc}(\Omega-\Omega_0)^3}\\
  z_{\mathrm R,s} = &\ \frac{\Omega_0 w_{0,s}^2}{2c} \\
  w_s(z) = &\ w_{0,s} \sqrt{1 + \frac{z^2}{z_{\mathrm R,s}^2}} \\
  R_s^{-1}(z) = &\ \frac{z}{z^2 + z_{\mathrm R,s}^2} \\
  \alpha(\Omega) = &\
    \frac{w_0}{c}\left[
      \Omega_0 \theta^\prime (\Omega-\Omega_0)
      + \frac{1}{2}\left( 2 \theta^\prime + \Omega_0 \theta^{\prime\prime} \right) (\Omega-\Omega_0)^2
      \right. \\
    &\ \qquad \left. + \frac{1}{6}\left(
        3\theta^{\prime\prime} + \Omega_0\theta^{\prime\prime\prime} - \Omega_0 {\theta^\prime}^3
      \right) (\Omega-\Omega_0)^3
    \right]

Note, :math:`k_x \approx - \alpha / w_{0,x}`.

The sought result for :math:`\int \hat K_y \epsilon_{y^\prime}\,\mathrm dy^\prime` is obtained from the 2D solution :math:`\hat E_x(x,z,\Omega)` by dropping :math:`\epsilon_\Omega` and :math:`\mathrm e^{-\imath(\Omega/c)z}`, plus letting :math:`\alpha \rightarrow 0` and :math:`x_0 \rightarrow 0`.
Hence,

.. math::
  \hat E_y(y,z,\Omega)& := \int\limits_{-\infty}^{\infty}
        \hat K_y(y^\prime, z, \Omega) \epsilon_{y^\prime} \,\mathrm dy^\prime \\
  & = \sqrt{\frac{\Omega}{2\pi c}} \frac{e^{\imath\frac{\pi}{4}}}{\sqrt{z}}
      \int\limits_{-\infty}^{\infty} e^{-\frac{{y^\prime}^2}{w_{0,y^\prime}^2}}
        e^{-\imath \frac{\Omega}{2 c z} (y-y^\prime)^{2}} \,\mathrm dy^\prime \\
  & = \left[ 1 + \frac{z^2}{z_{\mathrm R,y}^2} \right]^{-1/4}
    \mathrm e^{-y^2\left[\frac{1}{w_y(z)^2} + \imath \frac{\Omega}{2c}R_y^{-1}(z)\right]}
    \mathrm e^{\imath\frac{1}{2}\arctan\frac{z}{z_{\mathrm R,y}}}\,.


Result
^^^^^^
According to the definitions above

.. math::
  \hat E(x,y,z,\Omega) = \hat E_x \cdot \hat E_y\,.


3D field in time-space domain
-----------------------------
Derivation
^^^^^^^^^^
For the Fourier transform to time-space domain, again the results of [Steiniger2024]_ can be reused since the modifications due to the presence of :math:`\hat E_y` are easy to incorporate.
In fact, when applying the transformation :math:`\Omega \rightarrow \tfrac{1}{\tau_0}\Omega^\prime + \Omega_0`, as is done in the reference,

.. math::
  \hat E_y(y,z,\frac{1}{\tau_0}\Omega^\prime + \Omega_0) =
    \left[ 1 + \frac{z^2}{z_{\mathrm R,y}^2} \right]^{-1/4}
    \mathrm e^{-\frac{y^2}{w_y(z)^2}} \mathrm e^{-\imath \Omega_0 \frac{y^2 R^{-1}_y(z)}{2c}}
    \mathrm e^{\imath\frac{1}{2}\arctan\frac{z}{z_{\mathrm R,y}}}
    \mathrm e^{-\imath \frac{y^2 R_y^{-1}(z)}{2c}\frac{1}{\tau_0}\Omega^\prime}

only the last exponent depends on frequency :math:`\Omega^\prime` and needs to be taken into account in the Fourier transform.
This last exponential simply contributes a term to :math:`\gamma_4`, which is located between eqs. (13) and (14) in [Steiniger2024]_, in the form of

.. math::
  -\frac{y^2 R_y^{-1}(z)}{2c}\frac{1}{\tau_0}\,.


Result
^^^^^^
In total, the 3D time-space domain field :math:`E(x,y,z,t)` of the ``DispersivePulse`` is obtained by taking the existing 2D solution :math:`E(x,z,t)`, cf. eq. (14) in [Steiniger2024]_, and applying

.. math::
  \gamma_4^\prime& = \gamma_4 - \frac{y^2 R_y^{-1}(z)}{2c}\frac{1}{\tau_0} \\
  E(x,y,z,t)& = E(x,z,t,) |_{\gamma_4 \rightarrow \gamma_4^\prime} \cdot
    \left[ 1 + \frac{z^2}{z_{\mathrm R,y}^2} \right]^{-1/4}
    \mathrm e^{-\frac{y^2}{w_y(z)^2}} \mathrm e^{-\imath \Omega_0 \frac{y^2 R^{-1}_y(z)}{2c}}
    \mathrm e^{\imath\frac{1}{2}\arctan\frac{z}{z_{\mathrm R,y}}}\,.


Special case without dispersion: The ``GaussianPulse``
-----------------------------------------------------------

In that case

.. math::
  \gamma_1& = 1 \\
  \gamma_2& = 0 \\
  \gamma_3& = 0 \\
  \gamma_4^\prime& = \left[
      t - \frac{z}{c} - \frac{1}{2c}\left(x^2 R_x^{-1}(z)+y^2 R_y^{-1}(z)\right)
    \right]\frac{1}{\tau_0}


.. math::
    E(x,y,z,t) = &\
        \frac{1}{\tau_0\sqrt{\pi}}
        \left( 1 + \frac{z^2}{z_{\mathrm R,x}^2} \right)^{-1/4}
        \left( 1 + \frac{z^2}{z_{\mathrm R,y}^2} \right)^{-1/4} \\
    &\ \times
        e^{\imath \Omega_0 \gamma_4^\prime \tau_0}
        e^{\imath \frac{1}{2} \left(\arctan\frac{z}{z_{\mathrm R,x}} + \arctan\frac{z}{z_{\mathrm R,y}}\right)}
        e^{-\left(\frac{x^2}{w_x^2} + \frac{y^2}{w_y^2}\right)}
        e^{-{\gamma_4^\prime}^2}\,.


References
----------
.. [Pausch2022]
        R. Pausch et al.
        *Modeling Beyond Gaussian Laser Pulses in Particle-in-Cell Simulations - The Impact of Higher Order Laser Modes*
        2022 IEEE Advanced Accelerator Concepts Workshop (AAC), Long Island, NY, USA (2022).
        https://doi.org/10.1109/AAC55212.2022.10822876

.. [Siegman1986]
        Siegman, Anthony E.
        *Lasers*,
        University science books (1986).

.. [Steiniger2024]
        K. Steiniger et al.
        *Distortions in focusing laser pulses due to spatio-temporal couplings: an analytic description*,
        High Power Laser Science and Engineering 12 (2024).
        https://doi.org/10.1017/hpl.2023.96
