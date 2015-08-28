PIConGPU - A Many GPGPU PIC Code
================================================================

Open Alpha
----------

Please note that this is an Open
[Alpha](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
release for **developers** and **power users** [only](#users).

**Users** *should wait* for our 
[Open Beta](https://github.com/ComputationalRadiationPhysics/picongpu/milestones)
release!

********************************************************************************

[![PIConGPU Presentation Video](http://img.youtube.com/vi/nwZuG-XtUDE/0.jpg)](http://www.youtube.com/watch?v=nwZuG-XtUDE)
[![PIConGPU Alpha Release](doc/logo/pic_logo_vert_180x360_alpha.png)](http://www.youtube.com/watch?v=nwZuG-XtUDE)

Introduction
------------

PIConGPU is a fully relativistic, many
[GPGPU](http://en.wikipedia.org/wiki/Graphics_processing_unit),
3D3V particle-in-cell ([PIC](http://en.wikipedia.org/wiki/Particle-in-cell))
code. The Particle-in-Cell algorithm is a central tool in plasma physics.
It describes the dynamics of a plasma by computing the motion of
electrons and ions in the plasma based on
[Maxwell's equations](http://en.wikipedia.org/wiki/Maxwell%27s_equations).

PIConGPU implements various numerical schemes to solve the PIC cycle.
Its features include:
- a central or Yee-lattice for fields
- particle pushers that solve the equation of motion for charged and neutral
  particles, e.g., the *Boris-* and the *Vay-Pusher*
- Maxwell field solvers, e.g. *Yee's* and *Lehe's* scheme
- rigorously charge conserving current deposition schemes, such as
  *Villasenor-Buneman*, *Esirkepov* and *ZigZag*
- macro-particle form factors ranging from NGP (0th order), CIC (1st),
  TSC (2nd), PSQ (3rd) to P4S (4th)

Besides the electro-magnetic PIC algorithm, we developed a wide range of tools
and diagnostics, e.g.:
- online, far-field radiation diagnostics for coherent and incoherent radiation
  emitted by charged particles
- full restart and output capabilities, including
  [parallel HDF5](http://hdfgroup.org/) (via
  [libSplash](https://github.com/ComputationalRadiationPhysics/libSplash)) and
  [ADIOS](https://www.olcf.ornl.gov/center-projects/adios/), allowing for
  extreme I/O scalability and massively parallel online-analysis
- 2D and 3D live view and diagnostics tools
- a large selection of extensible
  [online-plugins](https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-Plugins)

Todays GPUs provide a computational performance of several
[TFLOP/s](http://en.wikipedia.org/wiki/FLOPS) at considerable lower invest and
maintenance costs compared to multi CPU-based compute architectures of similar
performance. The latest high-performance systems
([TOP500](http://www.top500.org/)) are enhanced by accelerator hardware that
boost their peak performance up to the multi-PFLOP/s level. With its
outstanding performance and scalability to more than 18'000 GPUs,
PIConGPU was one of the **finalists** of the 2013
[Gordon Bell Prize](http://sc13.supercomputing.org/content/acm-gordon-bell-prize).

PIConGPU is developed and maintained by the
[Computational Radiation Physics Group](http://www.hzdr.de/db/Cms?pNid=132&pOid=30354)
at the [Institute for Radiation Physics](http://www.hzdr.de/db/Cms?pNid=132)
at [HZDR](http://www.hzdr.de/) in close collaboration with the Center
for Information Services and High Performance Computing
([ZIH](http://tu-dresden.de/die_tu_dresden/zentrale_einrichtungen/zih)) of the
Technical University Dresden ([TUD](http://www.tu-dresden.de)). We are a
member of the [Dresden GPU Center of Excellence](http://ccoe-dresden.de/) that
cooperates on a broad range of scientific GPU and manycore applications,
workshops and teaching efforts.

Attribution
-----------

PIConGPU is a *scientific project*. If you **present and/or publish** scientific
results that used PIConGPU, you should set a **reference** to show your support.

Our according **up-to-date publication** at **the time of your publication**
should be inquired from:
- [REFERENCE.md](https://raw.githubusercontent.com/ComputationalRadiationPhysics/picongpu/master/REFERENCE.md)

Oral Presentations
------------------

The following slide should be part of **oral presentations**. It is intended to
acknowledge the team maintaining PIConGPU and to support our community:

(*coming soon*) presentation_picongpu.pdf
(svg version, key note version, png version: 1920x1080 and 1024x768)

Software License
----------------

*PIConGPU* is licensed under the **GPLv3+**. Furthermore, you can develop your
own particle-mesh algorithms based on our general library *libPMacc* that is
shipped alongside PIConGPU. *libPMacc* is *dual licensed* under both the
**GPLv3+ and LGPLv3+**.
For a detailed description, please refer to [LICENSE.md](LICENSE.md)

********************************************************************************

Install
-------

See our notes in [INSTALL.md](doc/INSTALL.md).

Users
-----

Dear User, please [be aware](#open-alpha) that this is a **developer and
power user only release**! We hereby emphasize that you should wait for our
[Beta](https://github.com/ComputationalRadiationPhysics/picongpu/milestones)
release.

Having said that and assuming that you are either an enthusiast or scientist
(or both/neither of them, the important point here is that you are willing to
read and understand our documentation and change logs):
*you are very welcome*!

For any questions regarding the usage of PIConGPU please **do not** contact the
developers and maintainers directly.

Instead, please sign up to our **PIConGPU-Users** mailing list so we can
distribute and archive user questions:
[Subscribe (Feed)](https://cg.hzdr.de/Lists/picongpu-users/List.html).
You can **subscribe** by simply sending an e-mail to
[picongpu-users-feed@hzdr.de](mailto:picongpu-users-feed@hzdr.de?subject=Subscribe me!)
(and unsubscribe via [picongpu-users-off@hzdr.de](mailto:picongpu-users-off@hzdr.de?subject=Unsubscribe me!)).

Before you post a question, browse the PIConGPU
[documentation](https://github.com/ComputationalRadiationPhysics/picongpu/search?l=markdown),
[wiki](https://github.com/ComputationalRadiationPhysics/picongpu/wiki),
[issue tracker](https://github.com/ComputationalRadiationPhysics/picongpu/issues) and the
[mailing list history](https://cg.hzdr.de/Lists/picongpu-users/List.html)
to see if your question has been answered, already.

PIConGPU is a collaborative project. We thus encourage users to engage in
answering questions of other users and post solutions to problems to the
list. A problem you have encountered might be the future problem of another
user.

In addition, please consider using the collaborative features of GitHub if you
have questions or comments on code or documentation. This will allow other
users to see the piece of code or documentation you are referring to.

Feel free to visit [picongpu.hzdr.de](http://picongpu.hzdr.de) to learn more
about the PIC algorithm. Further ressources are the
[user section](https://github.com/ComputationalRadiationPhysics/picongpu/wiki)
of our wiki, documentation files in `.md` format in this repository and a
[getting started video](http://www.youtube.com/watch?v=7ybsD8G4Rsk).

Software Upgrades
-----------------

PIConGPU follows a
[master - dev](http://nvie.com/posts/a-successful-git-branching-model/)
development model. That means our latest stable release is shipped in a branch
called `master` while new and frequent changes to the code are incooporated
in the development branch `dev`.

Every time we update the *master* branch, we publish a new release
of PIConGPU. Before you pull the changes in, please read our
[ChangeLog](CHANGELOG.md)!
You may have to update some of your simulation `.param` and `.cfg` files by
hand since PIConGPU is an active project and new features often require changes
in input files. Additionally, a full description of new features and fixed bugs
in comparison to the previous release is provided in that file.

In case you decide to use *new, potentially buggy and experimental* features
from our `dev` branch, be aware that support is very limited and you must
participate or at least follow the development yourself. Syntax changes
and in-development bugs will *not* be announced outside of their according pull
requests and issues.

Before drafting a new release, we open a new `release-*` branch from `dev` with
the `*` being the version number of the upcoming release. This branch only
receives bug fixes (feature freeze) and users are welcome to try it out
(however, the change log and a detailed announcement might still be missing in
it).

Developers
----------

### How to participate

See [CONTRIBUTING.md](CONTRIBUTING.md)

Active Team
-----------

### Scientific Supervision

- Dr. Michael Bussmann
- Dr.-Ing. Guido Juckeland

### Maintainers* and core developers

- Heiko Burau*
- Dr. Alexander Debus
- Carlchristian Eckert
- Marco Garten
- Alexander Grund
- Axel Huebl*
- Maximilian Knespel
- Richard Pausch*
- Stefan Tietze
- Rene Widera*
- Benjamin Worpitz*

### Former Members, Contributions and Thanks

The PIConGPU Team expresses its thanks to:

- Florian Berninger
- Robert Dietrich
- Wen Fu, PhD
- Anton Helm
- Wolfgang Hoehnig
- Dr. Remi Lehe
- Felix Schmitt(*)
- Benjamin Schneider
- Joseph Schuchart
- Conrad Schumann
- Klaus Steiniger

Kudos to everyone who helped!

********************************************************************************

![image of an lwfa](doc/images/lwfa_grey.png "LWFA")
![image of our strong scaling](doc/images/StrongScalingPIConGPU_log.png "Strong Scaling")
