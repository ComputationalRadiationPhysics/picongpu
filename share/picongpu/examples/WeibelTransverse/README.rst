WeibelTransverse: Transverse Weibel Instability
===============================================

.. sectionauthor:: Julian J. Lenz

Two homogeneous, counter-streaming beams of equal mass and opposite charge are
initialized in a periodic box. The species called ``PIC_Ions`` drifts along -z
and ``PIC_Electrons`` along +z, both with a Lorentz factor of 1.021
(:math:`\beta \approx 0.2`). The ion mass ratio is 1, so the pair is
equivalent to an electron-positron plasma. Electrons additionally receive a
temperature of 5 eV. A quiet start with two macro-particles per cell in x and
y and one in z is used.

The counter-stream drives the transverse Weibel (filamentation) instability,
which grows magnetic fields transverse to the beams. The box has
192 x 512 x 12 cells of 18 nm and is periodic in all directions. The reference
configuration ``etc/picongpu/4.cfg`` uses four GPUs (1 x 4 x 1) and runs for
2500 steps. It writes PNG snapshots of both species in the y-z and y-x planes,
energy histograms, particle counts and field energies.

Usage
-----

.. code-block:: bash

   pic-create $PIC_EXAMPLES/WeibelTransverse $HOME/picInputs/weibelTransverse
   cd $HOME/picInputs/weibelTransverse
   pic-build
   tbg -s bash -c etc/picongpu/4.cfg -t $PICSRC/etc/picongpu/bash/mpiexec.tpl $SCRATCH/runs/weibelTransverse
