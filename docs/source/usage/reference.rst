.. _usage-reference:

Reference
=========

.. sectionauthor:: Axel Huebl

PIConGPU is a more than decade-long scientific project with many people contributing to it.
In order to credit the work of others, we expect you to cite our latest paper describing PIConGPU when publishing and/or presenting scientific results.

In addition to that and out of good scientific practice, you should document the version of PIConGPU that was used and any modifications you applied.
A list of releases alongside a DOI to reference it can be found here:

https://github.com/ComputationalRadiationPhysics/picongpu/releases


Citation
--------

BibTeX code:

.. include:: REFERENCE.md
   :code: TeX
   :start-line: 1
   :end-line: -1


Acknowledgements
----------------

In many cases you receive support and code base maintainance from us or the PIConGPU community without directly justifying a full co-authorship.
Additional to the citation, please consider adding an acknowledgement of the following form to reflect that:

    We acknowledge all contributors to the open-source code PIConGPU for enabling our simulations.

or:

    We acknowledge [list of specific persons that helped you] and all further contributors to the open-source code PIConGPU for enabling our simulations.

Community Map
-------------

PIConGPU comes without a registration-wall, with open and re-distributable licenses and without any strings attached.
We therefore *rely on you* to show our community, diversity and usefulness, e.g. to funding agencies.

    Please consider adding yourself to our `community map <https://github.com/ComputationalRadiationPhysics/picongpu-communitymap>`_!

Thank you and enjoy PIConGPU and our community!

.. raw:: html

   <iframe src="https://computationalradiationphysics.github.io/picongpu-communitymap/" style="width:100%; height: 650px;" frameborder="0"></iframe>


Publications
------------

The following publications are sorted by topics.
Papers covering multiple topics will be listed multiple times.
In the end, a list of all publications in chronological order is given with more details.
If you want to add your publication to the list as well please feel free to contact us or open a pull request directly.

Application of PIConGPU in various physics scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following, a list of publications describing using PIConGPU is given in various cases.

Laser plasma electron acceleration
""""""""""""""""""""""""""""""""""
 - Laser wakefield acceleration (LWFA) with self-truncated ionization-injection (STII) [Couperus2017]_
 - PhD thesis on experimental aspects of LWFA with STII [Couperus2018]_,
 - PhD thesis on theoretical ascpects of self-focusing during LWFA with STII [Pausch2019]_
 - Hybrid laser-driven/beam-driven plasms acceleration [Kurz2021]_
 - Acceleration in carbon nanotubes [Perera2020]_
 - TeV/m catapult acceleration in graphene layers [Bontoiu2023]_

Laser plasma ion acceleration
"""""""""""""""""""""""""""""
 - Proton acceleration from cryogenic hydrogen jets [Obst2017]_
 - Mass-Limited, Near-Critical, micron-scale, spherical targets [Hilz2018]_
 - PhD thesis on theoretical aspects of mass-limited, near-critical, micron-scale, spherical targets [Huebl2019]_
 - All-optical stuctering of laser-accelerated protons [Obst-Huebl2018]_
 - PhD thesis on laser-driven proton beam structering [Obst-Huebl2019]_
 - Laser-ion multi-species acceleration [Huebl2020]_

Laser plasma light sources and diagnostics
""""""""""""""""""""""""""""""""""""""""""
 - PhD thesis on radiation from LWFA [Pausch2019]_
 - Laser-driven x-ray and proton sources [Ostermayr2020]_
 - Betatron x-ray diagnostic for beam decoherence [Koehler2019]_, [Koehler2021]_


Astrophysics
""""""""""""
 - Simulating the Kelvin Helmholtz instability (KHI) [Bussmann2013]_
 - Visualizing the KHI [Huebl2014]_
 - Theoretical model of the radiation evolution during the KHI [Pausch2017]_
 - PhD thesis covering the KHI radiation [Pausch2019]_

Machine Learning
""""""""""""""""


Methods used in PIConGPU software
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following, a list of references is given, sorted by topics, that describe PIConGPU as a software.
References to publications of implemented methods that were developed by other groups are also given for completeness.
These references are marked by an asterisk.


General code references
"""""""""""""""""""""""

 - First publication on PIConGPU [Burau2010]_
 - Currently main reference [Bussmann2013]_
 - Most up-to-date reference [Huebl2019]_

Field solvers
"""""""""""""
 - Yee field solver\ :sup:`\*` [Yee1966]_
 - Lehe field solver\ :sup:`\*` [Lehe2013]_
 - High-Order Finite Difference solver\ :sup:`\*` [Ghrist2000]_

Particle pushers
""""""""""""""""
 - Boris pusher\ :sup:`\*` [Boris1970]_
 - Vay pusher\ :sup:`\*` [Vay2008]_
 - Reduced Landau-Lifshitz pusher\ :sup:`\*` [Vranic2016]_
 - Higuera-Cary pusher\ :sup:`\*` [Higuera2017]_

Current deposition
""""""""""""""""""
 - Esirkepov method\ :sup:`\*` [Esirkepov2001]_

Ionization-physics extensions
"""""""""""""""""""""""""""""
 - Barrier suppression ionization (BSI)\ :sup:`\*` [ClementiRaimondi1963]_, [ClementiRaimondi1967]_, [MulserBauer2010]_
 - Tunneling ionization - Keldysh\ :sup:`\*` [Keldysh]_
 - Tunneling ionization - Ammosov-Delone-Krainov (ADK)\ :sup:`\*` [DeloneKrainov1998]_, [BauerMulser1999]_
 - Master thesis - model implementation [Garten2015]_
 - Collisional ionization [FLYCHK2005]_, [More1985]_
 - ionization current\ :sup:`\*` [Mulser1998]_

Binary_collisions
"""""""""""""""""
 - fundamental alogorithm\ :sup:`\*` [Perez2012]_
 - improvements and corrections\ :sup:`\*` [Higginson2020]_

QED code extensions
"""""""""""""""""""
 - Various methods applicable in PIC codes\ :sup:`\*` [Gonoskov2015]_
 - Diploma thesis - model implementation [Burau2016]_

Diagnostic methods
""""""""""""""""""
 - classical radiation: [Pausch2012]_, [Pausch2014]_, [Pausch2018]_, [Pausch2019]_
 - phase space analysis: [Huebl2014]_
 - beam emittance: [Rudat2019]_
 - coherent transistion radiation (CTR): [Carstens2019]_

Visualization
"""""""""""""
 - first post-processing implementation: [Zuehl2011]_, [Ungethuem2012]_
 - first in-situ visualization: [Schneider2012a]_, [Schneider2012b]_
 - Kelvin-Helmholtz instabilty: [Huebl2014]_
 - in-situ visualization with ISAAC: [Matthes2016]_, [Matthes2016b]_
 - in-situ particle rendering: [Meyer2018]_

Input/Output
""""""""""""
 - parallel HDF5, ADIOS1, compression, data reduction and I/O performance model [Huebl2017]_

HPC kernels and benchmarks
""""""""""""""""""""""""""
 - proceedings of the SC'13 [Bussmann2013]_

Theses
^^^^^^^^
 - Diploma thesis: first post-processing rendering [Zuehl2011]_
 - Diploma thesis: first in-situ rendering [Schneider2012b]_
 - Diploma thesis: In-situ radiation calculation [Pausch2012]_
 - Diploma thesis: Algorithms, LWFA injection, Phase Space analysis [Huebl2014]_
 - Master thesis: Ionization methods [Garten2015]_
 - Diploma thesis: QED scattering processes [Burau2016]_
 - Diploma thesis: In-situ live visualization [Matthes2016]_
 - PhD thesis: LWFA injection using STII (mainly experiment) [Couperus2018]_
 - Bachelor thesis: In-situ live visualization [Meyer2018]_
 - Master thesis: Beam emittance and automated parameter scans [Rudat2019]_
 - PhD thesis: Radiation during LWFA and KHI, radiative corrections [Pausch2019]_
 - PhD thesis: LWFA betatron radiation (mainly experiment) [Koehler2019]_
 - PhD thesis: LWFA Coherent transistion radiation diagnostics (CTR) (mainly experiment) [Zarini2019]_
 - PhD thesis: Laser ion acceleration (mainly experiment) [Obst-Huebl2019]_
 - PhD thesis: Exascale simulations with PIConGPU, laser ion acceleration  [Huebl2019]_
 - Bachelor thesis: Synthetic coherent transistion radiation [Carstens2019]_


List of PIConGPU references in chronological order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [Burau2010]
       Burau, H., et al.,
       *A fully relativistic particle-in-cell code for a GPU cluster.*,
       IEEE Transactions on Plasma Science, 38(10 PART 2), 2831–2839 (2010),
       https://doi.org/10.1109/TPS.2010.2064310

.. [Zuehl2011]
       Zühl, L.,
       *Visualisierung von Laser-Plasma-Simulationen*,
       Diploma Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Informatiker" (2011),
       https://www.hzdr.de/db/Cms?pOid=35687

.. [Schneider2012a]
       Schneider, B.,
       *Gestengesteuerte visuelle Datenanalyse einer Laser-Plasma-Simulation*,
       Student Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2012),
       https://www.hzdr.de/db/Cms?pOid=37242

.. [Schneider2012b]
       Schneider, B.,
       *In Situ Visualization of a Laser-Plasma Simulation*,
       Diploma Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Informatiker" (2012),
       https://www.hzdr.de/db/Cms?pOid=40353

.. [Pausch2012]
       Pausch, R.,
       *Electromagnetic Radiation from Relativistic Electrons as Characteristic Signature of their Dynamics*,
       Diploma Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2012),
       https://doi.org/10.5281/zenodo.843510

.. [Ungethuem2012]
       Ungethüm, A.,
       *Simulation and visualisation of the electro-magnetic field around a stimulated electron*,
       Student Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2012),
       https://www.hzdr.de/db/Cms?pOid=38508

.. [Bussmann2013]
       Bussmann, M. et al.,
       *Radiative signatures of the relativistic Kelvin-Helmholtz instability*,
       SC ’13 Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 5-1-5–12),
       https://doi.org/10.1145/2503210.2504564

.. [Huebl2014]
       Huebl, A. et al.,
       *Visualizing the Radiation of the Kelvin-Helmholtz Instability*,
       IEEE Transactions on Plasma Science 42.10 (2014),
       https://doi.org/10.1109/TPS.2014.2327392

.. [Pausch2014]
       Pausch, R., Debus, A., Widera, R. et al.,
       *How to test and verify radiation diagnostics simulations within particle-in-cell frameworks*,
       Nuclear Instruments and Methods in Physics Research, Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 740, 250–256 (2014),
       https://doi.org/10.1016/j.nima.2013.10.073

.. [Huebl2014]
        Huebl, A.,
        *Injection Control for Electrons in Laser-Driven Plasma Wakes on the Femtosecond Time Scale*,
        Diploma Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2014),
        https://doi.org/10.5281/zenodo.15924

.. [Garten2015]
        Garten, M.,
        *Modellierung und Validierung von Feldionisation in parallelen Particle-in-Cell-Codes*,
        Master Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2015),
        https://doi.org/10.5281/zenodo.202500

.. [Burau2016]
        Burau, H.,
        *Entwicklung und Überprüfung eines Photonenmodells für die Abstrahlung durch hochenergetische Elektronen*,
        Diploma Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2016),
        https://doi.org/10.5281/zenodo.192116

.. [Matthes2016]
        Matthes, A.,
        *In-Situ Visualisierung und Streaming von Plasmasimulationsdaten*,
        Diploma Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Informatiker" (2016),
        https://doi.org/10.5281/zenodo.55329

.. [Matthes2016b]
        Matthes, A., Huebl, A., Widera, R., Grottel, S., Gumhold, S., Bussmann, M.,
        *In situ, steerable, hardware-independent and data-structure agnostic visualization with ISAAC*,
        Supercomputing Frontiers and Innovations, [S.l.], v. 3, n. 4, p. 30-48, oct. 2016,
        https://doi.org/10.14529/jsfi160403

.. [Pausch2017]
       Pausch, R., Bussmann, M., Huebl, A., Schramm, U., Steiniger, K., Widera, R. and Debus, A.,
       *Identifying the linear phase of the relativistic Kelvin-Helmholtz instability and measuring its growth rate via radiation*,
       Phys. Rev. E 96, 013316 - Published 26 July 2017,
       https://doi.org/10.1103/PhysRevE.96.013316

.. [Couperus2017]
       Couperus, J. P. et al.,
       *Demonstration of a beam loaded nanocoulomb-class laser wakefield accelerator*,
       Nature Communications volume 8, Article number: 487 (2017),
       https://doi.org/10.1038/s41467-017-00592-7

.. [Huebl2017]
        Huebl, A. et al.,
        *On the Scalability of Data Reduction Techniques in Current and Upcoming HPC Systems from an Application Perspective*,
        ISC High Performance Workshops 2017, LNCS 10524, pp. 15-29 (2017),
        https://doi.org/10.1007/978-3-319-67630-2_2

.. [Obst2017]
        Obst, L., Göde, S., Rehwald, M. et al.,
        *Efficient laser-driven proton acceleration from cylindrical and planar cryogenic hydrogen jets*,
        Sci Rep 7, 10248 (2017),
        https://doi.org/10.1038/s41598-017-10589-3

.. [Pausch2018]
       Pausch, R., Debus, A., Huebl, A. at al.,
       *Quantitatively consistent computation of coherent and incoherent radiation in particle-in-cell codes — A general form factor formalism for macro-particles*,
       Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 909, 419–422 (2018),
       https://doi.org/10.1016/j.nima.2018.02.020

.. [Couperus2018]
        Couperus, J. P.,
        *Optimal beam loading in a nanocoulomb-class laser wakefield accelerator*,
        PhD Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2018),
        https://doi.org/10.5281/zenodo.1463710

.. [Meyer2018]
        Meyer, F.,
        *Entwicklung eines Partikelvisualisierers für In-Situ-Simulationen*,
        Bachelor Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2018),
        https://doi.org/10.5281/zenodo.1423296

.. [Hilz2018]
        Hilz, P, et al.,
        *Isolated proton bunch acceleration by a petawatt laser pulse*,
        Nature Communications, 9(1), 423 (2018),
        https://doi.org/10.1038/s41467-017-02663-1

.. [Obst-Huebl2018]
        Obst-Huebl, L., Ziegler, T., Brack, FE. et al.,
        *All-optical structuring of laser-driven proton beam profiles*,
        Nat Commun 9, 5292 (2018),
        https://doi.org/10.1038/s41467-018-07756-z

.. [Rudat2019]
       Rudat, S.,
       *Laser Wakefield Acceleration Simulation as a Service*,
       Master Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://doi.org/10.5281/zenodo.3529741

.. [Pausch2019]
       Pausch, R.,
       *Synthetic radiation diagnostics as a pathway for studying plasma dynamics from advanced accelerators to astrophysical observations*,
       PhD Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://doi.org/10.5281/zenodo.3616045

.. [Koehler2019]
       Köhler, A.,
       *Transverse Electron Beam Dynamics in the Beam Loading Regime*,
       PhD Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://doi.org/10.5281/zenodo.3342589

.. [Zarini2019]
       Zarini, O.,
       *Measuring sub-femtosecond temporal structures in multi-ten kiloampere electron beams*,
       PhD Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://nbn-resolving.org/urn:nbn:de:bsz:d120-qucosa2-339775

.. [Obst-Huebl2019]
       Obst-Hübl, L.,
       *Achieving optimal laser-proton acceleration through multi-parameter interaction control*,
       PhD Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://doi.org/10.5281/zenodo.3252952

.. [Huebl2019]
       Huebl, A.,
       *PIConGPU: Predictive Simulations of Laser-Particle Accelerators with Manycore Hardware*,
       PhD Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://doi.org/10.5281/zenodo.3266820

.. [Carstens2019]
       Carstens, F.-O.,
       *Synthetic characterization of ultrashort electron bunches using transition radiation*,
       Bachelor Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://doi.org/10.5281/zenodo.3469663

.. [Ostermayr2020]
       Ostermayr, T. M., et al.,
       *Laser-driven x-ray and proton micro-source and application to simultaneous single-shot bi-modal radiographic imaging*,
       Nature Communications volume 11, Article number: 6174 (2020),
       https://doi.org/10.1038/s41467-020-19838-y

.. [Huebl2020]
       Huebl, A. et al.,
       *Spectral control via multi-species effects in PW-class laser-ion acceleration*,
       Plasma Phys. Control. Fusion 62 124003 (2020),
       https://doi.org/0.1088/1361-6587/abbe33

.. [Perera2020]
       Perera, A. et al.,
       *Towards ultra-high gradient particle acceleration in carbon nanotubes*,
       Journal of Physics: Conference Series 1596 012028 (2020),
       https://doi.org/10.1088/1742-6596/1596/1/012028

.. [Kurz2021]
        Kurz, T. et al.,
        *Demonstration of a compact plasma accelerator powered by laser-accelerated electron beams*,
        Nature Communications volume 12, Article number: 2895 (2021),
        https://doi.org/10.1038/s41467-021-23000-7

.. [Koehler2021]
        Koehler, A., Pausch, R., Bussmann, M., et al.,
        *Restoring betatron phase coherence in a beam-loaded laser-wakefield accelerator*,
        Phys. Rev. Accel. Beams 24, 091302 – 20 September 2021,
	https://doi.org/10.1103/PhysRevAccelBeams.24.091302

.. [Bontoiu2023]
        Bontoiu, C., et al.,
	*TeV/m catapult acceleration of electrons in graphene layers*,
	Scientific Reports volume 13, Article number: 1330 (2023),
	https://doi.org/10.1038/s41598-023-28617-w

List of other references in chronological order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [ClementiRaimondi1963]
        Clementi, E. and Raimondi, D.,
        *Atomic Screening Constant from SCF Functions*,
        The Journal of Chemical Physics 38, 2686-2689 (1963),
        https://dx.doi.org/10.1063/1.1733573

.. [Keldysh]
        Keldysh, L.V.,
        *Ionization in the field of a strong electromagnetic wave*,
        Soviet Physics JETP 20, 1307-1314 (1965),
        http://jetp.ac.ru/cgi-bin/dn/e_020_05_1307.pdf

.. [Yee1966]
        Yee, K.,
	*Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media*,
	IEEE Transactions on Antennas and Propagation ( Volume: 14, Issue: 3, May 1966),
	https://doi.org/10.1109/TAP.1966.1138693

.. [ClementiRaimondi1967]
        Clementi, E. and D. Raimondi, D.,
        *Atomic Screening Constant from SCF Functions. II. Atoms with 37 to 86 Electrons*,
        The Journal of Chemical Physics 47, 1300-1307 (1967),
        https://dx.doi.org/10.1063/1.1712084

.. [Boris1970]
        Boris, J.,
	*Relativistic Plasma Simulation - Optimization of a Hybrid Code*,
	Proc. 4th Conf. on Num. Sim. of Plasmas (1970),
	http://www.dtic.mil/docs/citations/ADA023511

.. [More1985]
        R. M. More.
        *Pressure Ionization, Resonances, and the Continuity of Bound and Free States*,
        Advances in Atomic, Molecular and Optical Physics Vol. 21 C, 305-356 (1985),
        https://dx.doi.org/10.1016/S0065-2199(08)60145-1

.. [Mulser1998]
        Mulser, P. et al.,
        *Modeling field ionization in an energy conserving form and resulting nonstandard fluid dynamcis*,
        Physics of Plasmas 5, 4466 (1998),
        https://doi.org/10.1063/1.873184

.. [DeloneKrainov1998]
        Delone, N. B. and Krainov, V. P.,
        *Tunneling and barrier-suppression ionization of atoms and ions in a laser radiation field*,
        Phys. Usp. 41 469–485 (1998),
        http://dx.doi.org/10.1070/PU1998v041n05ABEH000393

.. [BauerMulser1999]
        Bauer, D. and Mulser, P.,
        *Exact field ionization rates in the barrier-suppression regime from numerical time-dependent Schrödinger-equation calculations*,
        Physical Review A 59, 569 (1999),
        https://dx.doi.org/10.1103/PhysRevA.59.569

.. [Ghrist2000]
        M. Ghrist,
        *High-Order Finite Difference Methods for Wave Equations*,
        PhD thesis (2000),
        Department of Applied Mathematics, University of Colorado

.. [Esirkepov2001]
        Esirkepov, T. Zh.,
	*Exact charge conservation scheme for Particle-in-Cell simulation with an arbitrary form-factor*,
	Computer Physics Communications, Volume 135, Issue 2, 1 April 2001, Pages 144-153,
	https://doi.org/10.1016/S0010-4655(00)00228-9

.. [FLYCHK2005]
        *FLYCHK: Generalized population kinetics and spectral model for rapid spectroscopic analysis for all elements*,
        H.-K. Chung, M.H. Chen, W.L. Morgan, Yu. Ralchenko, and R.W. Lee,
        *High Energy Density Physics* v.1, p.3 (2005)
        http://nlte.nist.gov/FLY/

.. [Vay2008]
        Vay, J.,
	*Simulation of beams or plasmas crossing at relativistic velocity*,
	Physics of Plasmas 15, 056701 (2008),
	https://doi.org/10.1063/1.2837054

.. [MulserBauer2010]
        Mulser, P. and Bauer, D.,
        *High Power Laser-Matter Interaction*,
        Springer-Verlag Berlin Heidelberg (2010),
        https://dx.doi.org/10.1007/978-3-540-46065-7

.. [Perez2012]
        Pérez, F., Gremillet, L., Decoster, A., et al.,
        *Improved modeling of relativistic collisions and collisional ionization in particle-in-cell codes*,
        Physics of Plasmas 19, 083104 (2012),
        https://doi.org/10.1063/1.4742167

.. [Lehe2013]
        Lehe, R., Lifschitz, A., Thaury, C., Malka, V. and Davoine, X.,
	*Numerical growth of emittance in simulations of laser-wakefield acceleration*,
	Phys. Rev. ST Accel. Beams 16, 021301 – Published 28 February 2013,
	https://doi.org/10.1103/PhysRevSTAB.16.021301

.. [Gonoskov2015]
       Gonoskov, A., Bastrakov, S., Efimenko, E., et al.,
       *Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and developments*,
       Phys. Rev. E 92, 023305 – Published 18 August 2015,
       https://doi.org/10.1103/PhysRevE.92.023305

.. [Vranic2016]
       Vranic, M., et al.,
       *Classical radiation reaction in particle-in-cell simulations*,
       Computer Physics Communications, Volume 204, July 2016, Pages 141-151,
       https://doi.org/10.1016/j.cpc.2016.04.002

.. [Higuera2017]
       Higuera, A. V. and Cary, J. R.,
       *Structure-preserving second-order integration of relativistic charged particle trajectories in electromagnetic fields*,
       Physics of Plasmas 24, 052104 (2017),
       https://doi.org/10.1063/1.4979989

.. [Higginson2020]
       Higginson, D. P. , Holod, I. , and Link, A.,
       *A corrected method for Coulomb scattering in arbitrarily weighted particle-in-cell plasma simulations*,
       Journal of Computational Physics 413, 109450 (2020).
       https://doi.org/10.1016/j.jcp.2020.109450
