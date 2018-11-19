.. _usage-workflows-memoryPerAccelerator:

Calculating the memory requirement per accelerator
--------------------------------------------------

.. sectionauthor:: Marco Garten

The planning of simulations for realistically sized problems requires a careful estimation of memory usage and is often a trade-off between resolution of the plasma, overall box size and the available resources.
The file :ref:`memory_calculator.py <usage-python-utils>` contains a class for this purpose.

The following paragraph shows the use of the ``MemoryCalculator`` for the ``4.cfg`` setup of the :ref:`FoilLCT example <usage-examples-foilLCT>` example.

.. code:: python

    from picongpu.utils import MemoryCalculator

    cell_size = 0.8e-6 / 384. # 2.083e-9 m

    y0 = 0.5e-6 # position of foil surface (m)
    y1 = 1.0e-6 # target thickness (m)
    L = 10.e-9 # pre-plasma scale length (m)
    L_cutoff = 4.0 * L # pre-plasma length (m)

    # number of cells per GPU
    Nx = 128
    Ny = 640
    Nz = 1

    vacuum_cells = (y0 - L_cutoff) / cell_size # with pre-plasma: 221 cells
    target_cells = (y1 - y0 + 2 * L_cutoff) / cell_size # 398 cells

    pmc = MemoryCalculator(Nx, Ny, Nz)

    target_x = Nx # full transversal dimension of the GPU
    target_y = target_cells # only the first row of GPUs holds the target
    target_z = Nz

    # typical number of particles per cell which is multiplied later for
    # each species and its relative number of particles
    N_PPC = 6

    print("Memory requirement per GPU:")
    # field memory per GPU
    field_gpu = pmc.mem_req_by_fields(Nx, Ny, Nz, field_tmp_slots=2,
                                      particle_shape_order=2)
    print("+ fields: {:.2f} MB".format(
        field_gpu / (1024 * 1024)))

    # electron macroparticles per supercell
    e_PPC = N_PPC * (
        # H,C,N pre-ionization - higher weighting electrons
        3 \
        # electrons created from C ionization
        + (6 - 2) \
        # electrons created from N ionization
        + (7 - 2) 
    )
    # particle memory per GPU - only the target area contributes here
    e_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=0,
        particles_per_cell=e_PPC
        )
    H_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        # no bound electrons since H is preionized
        num_additional_attributes=0,
        particles_per_cell=N_PPC
        )
    C_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=1,
        particles_per_cell=N_PPC
        )
    N_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=1,
        particles_per_cell=N_PPC
        )
    # memory for calorimeters
    cal_gpu = pmc.mem_req_by_calorimeter(
        n_energy=1024, n_yaw=360, n_pitch=1
    ) * 2  # electrons and protons
    # memory for random number generator states
    rng_gpu = pmc.mem_req_by_rng(Nx, Ny, Nz)

    print("+ species:")
    print("- e: {:.2f} MB".format(e_gpu / (1024 * 1024)))
    print("- H: {:.2f} MB".format(H_gpu / (1024 * 1024)))
    print("- C: {:.2f} MB".format(C_gpu / (1024 * 1024)))
    print("- N: {:.2f} MB".format(N_gpu / (1024 * 1024)))
    print("+ RNG states: {:.2f} MB".format(
        rng_gpu / (1024 * 1024)))
    print(
        "+ particle calorimeters: {:.2f} MB".format(
            cal_gpu / (1024 * 1024)))

    mem_sum = field_gpu + e_gpu + H_gpu + C_gpu + N_gpu + rng_gpu + cal_gpu
    print("Sum of required GPU memory: {:.2f} MB".format(
        mem_sum / (1024 * 1024)))

The output of this would be:

.. code:: bash

    Memory requirement per GPU:
    + fields: 42.74 MB
    + species:
    - e: 75.85 MB
    - H: 6.32 MB
    - C: 7.14 MB
    - N: 7.14 MB
    + RNG states: 3.75 MB
    + particle calorimeters: 5.62 MB
    Sum of required GPU memory: 148.57 MB
