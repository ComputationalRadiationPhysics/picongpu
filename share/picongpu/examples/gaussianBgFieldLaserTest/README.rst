Ultrashort, strongly focused Gaussian Laser crossing a Laser-Wakefield Accelerator
==================================================================================

.. sectionauthor:: Alexander Debus <a.debus (at) hzdr.de>

Laser-wakefield accelerator setup for a laser pulse based on PIConGPUs "background field" mechanism. The laser pulse model includes first-order corrections for ultrashort, fs-scale durations and strong focusing according to [Hua2004]. This alternative laser can replace the usual laser and has the advantage that one insert the laser into the simulation from any direction within the y-z-plane. Also one can use an arbitrary number of different laser pulses.

This is a demonstration setup for quick testing of this feature with bad resolution and unphysically high plasma gradients. In addition to the regular laser, the setup includes a background-field laser, which does not propagate along the usual y-axis, but crosses the simulation volume along the transverse z-axis, such as a probe beam. The configured incident angle (here: 90°) is measured agains the y-axis. For demonstration purposes, the probe beam features the same intensity as the initial drive laser. 

References
----------

.. [Hua2004]
        J. F. Hua, Y. K. Ho, Y. Z. Lin, Z. Chen, Y. J. Xie, S. Y. Zhang, Z. Yan, J. J. Xu
        High-order corrected fields of ultrashort, tightly focused laser pulses
        Applied Physics Letters (2004),
        https://dx.doi.org/10.1063/1.1811384
