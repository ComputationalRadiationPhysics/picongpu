.. _usage-reference:

Reference
=========

.. sectionauthor:: Axel Huebl

PIConGPU is an almost decade-long scientific project with many people contributing to it.
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

Laser plasma acceleration
"""""""""""""""""""""""""

 - Laser wakefield acceleration (LWFA) with self-truncated ionization-injection (STII)
 - Ion acceleration from droplet iradiation
 - Laser-driven proton beam structering
 - Hybrid laser-driven/beam-driven plasms acceleration 
 - Radiation from LWFA (concept paper)
   

Astrophysics
""""""""""""

 - Simulating the Kelvin Helmholtz instabilty (KHI) [Bussmann2013]_
 - Visualizing the KHI
 - Theoretical model of the radiation evolution during the KHI
 - PhD thesis covering the KHI radiation

   

Methods used in PIConGPU software
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following a list of references is given, sorted by topics, that describe PIConGPU as a software.
References to publications of implemented methods that were deveolped by other groups are also given for completness.
These references are marked by an asterix. 


General code references
"""""""""""""""""""""""

 - First publication on PIConGPU [Burau2010]_
 - Currently main reference [Bussmann2013]_

Field solvers
"""""""""""""

 - Yee field solver\ :sup:`\*`
 - Lehe fields solver\ :sup:`\*`

Particle pushers
""""""""""""""""

 - Boris pusher H\ :sup:`2`\ O
 - Vay pusher\ :sup:`\*`

Current deposition
""""""""""""""""""

 - Esirkepov method\ :sup:`\*`
 - Villasenor and Buneman method\ :sup:`\*`

Ionization-physics extensions
"""""""""""""""""""""""""""""

 - Barrier suppression ionization (BSI)\ :sup:`\*`
 - Tunneling ionization - Keldysh\ :sup:`\*`
 - Tunneling ionization - Ammosov-Delone-Krainov (ADK)\ :sup:`\*`
 - Mater thesis - model implementation
 

QED code extensions
"""""""""""""""""""

Diagnostic methods
""""""""""""""""""

 - classical radiation: [Pausch2012]_, [Pausch2019]_
 - phase space analysis: [Huebl2014]_
 - beam emittance: [Rudat2019]_
 - coherent transistion radiation (CTR): [Castens2019]_

Visualization
"""""""""""""

 - first implementation: 

Input/Output
""""""""""""

HPC kernels
"""""""""""

Thesises
^^^^^^^^

 - Diploma thesis: first in-situ rendering [Zuehl2011]_ 
 - Diploma thesis: In-situ radiation calculation [Pausch2012]_
 - Diploma thesis: Algorithms, LWFA injection, Phase Space analysis [Huebl2014]_
 - Master thesis: Ionization methods [Garten2015]_
 - Diploma thesis: QED scattering processes [Burau2016]_
 - Diploma thesis: In-situ live visualization [Mathes2016]_
 - PhD thesis: LWFA injection using STII (mainly experiment) [Couperus2018]_
 - Master thesis: In-situ live visualization [Meyer2018]_ 
 - Master thesis: Beam emittance and automated parameter scans [Rudat2019]_ 
 - PhD thesis: Radiation during LWFA and KHI, radiative corrections [Pausch2019]_
 - PhD thesis: LWFA betatron radiation (mainly experiment) [Koehler2019]_
 - PhD thesis: LWFA Coherent transistion radiation diagnostics (CTR) (mainly experiment) [Zarini2019]_
 - PhD thesis: Laser ion acceleration (mainly experiment) [Obst-Huebl2019]_
 - PhD thesis: Exascale simulations with PIConGPU, laser ion acceleration  [Huebl2019]_
 - Bachelor thesis: Synthetic coherent transistion radiation [Castens2019]_ 




List of references in chronological order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


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

.. [Pausch2012]
       Pausch, R.,
       *Electromagnetic Radiation from Relativistic Electrons as Characteristic Signature of their Dynamics*,
       Diploma Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2012),
       https://doi.org/10.5281/zenodo.843510

.. [Bussmann2013]
       Bussmann, M. et al.
       *Radiative signatures of the relativistic Kelvin-Helmholtz instability*
       SC ’13 Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 5-1-5–12)
       https://doi.org/10.1145/2503210.2504564
	
.. [Huebl2014]
        Huebl, A.,
        *Injection Control for Electrons in Laser-Driven Plasma Wakes on the Femtosecond Time Scale*,
        Diploma Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2014),
        https://doi.org/10.5281/zenodo.15924

.. [Garten2015]
        Garten, M.,
	*Modellierung und Validierung von Feldionisation in parallelen Particle-in-Cell-Codes*,
	Master Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2015)
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

.. [Couperus2018]
        Couperus, J. P.,
	*Optimal beam loading in a nanocoulomb-class laser wakefield accelerator*,
	PhD Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2018)
	https://doi.org/10.5281/zenodo.1463710

.. [Meyer2018]
        Meyer, F.,
	*Entwicklung eines Partikelvisualisierers für In-Situ-Simulationen*,
	Master Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2018)
	https://doi.org/10.5281/zenodo.1423296

.. [Hilz2018]
        Hilz, P, et al.,
	*Isolated proton bunch acceleration by a petawatt laser pulse*,
	Nature Communications, 9(1), 423 (2018),
	https://doi.org/10.1038/s41467-017-02663-1

.. [Rudat2019]
       Rudat, S.,
       *Laser Wakefield Acceleration Simulation as a Service*,
       Master Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019)
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
       *Predictive Simulations of Laser-Particle Accelerators with Manycore Hardware*,
       PhD Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://doi.org/10.5281/zenodo.3266820

.. [Carstens2019]
       Carstens, F.-O.,
       *Synthetic characterization of ultrashort electron bunches using transition radiation*,
       Bachelor Thesis at Technische Universität Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
       https://doi.org/10.5281/zenodo.3469663

.. [Kurz2020]
        Kurz, T. et al.,
	*Demonstration of a compact plasma accelerator powered by laser-accelerated electron beams*,
	in review Nature Physics,
	http://arxiv.org/abs/1909.06676B
