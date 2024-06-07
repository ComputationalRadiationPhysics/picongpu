"""
This file is a modified version of the pipe script from the openPMD-api.

Authors: Richard Pausch, Franz Poeschel, Nico Wrobel
License: LGPLv3+
"""
import sys

import openpmd_api as io
import numpy as np


class vec3D:
    """
    a helper class to easily handle 3D vectors in python without the need
    to reference another dimension of a numpy array
    """

    x = 0
    y = 0
    z = 0

    def __init__(self, x, y, z):
        """
        initialize with 3 values (x,y,z)

        Arguments:
        x: first value
        y: second value
        z: third value
        """
        self.x = x
        self.y = y
        self.z = z

    def prod(self):
        """
        product of all 3 components

        returns x * y * z
        """
        return self.x * self.y * self.z

    def print(self):
        """
        helper function to print values to screen
        """
        print("x: {}".format(self.x))
        print("y: {}".format(self.y))
        print("z: {}".format(self.z))

    def __truediv__(self, other):
        """
        component wise division using the '/' operator

        Arguments:
        other: float value
               the number by which all 3 components should be divided

        Return:
        vec3D( x/other, y/other, z/other )
        """
        return vec3D(self.x / other, self.y / other, self.z / other)


class addParticles2Checkpoint:
    def print(self, string):
        """
        helper function that prints information depending on the set
        verbose level

        Arguments:
        string: string
                message to post
        """
        if self.verbose:
            print("\t" * self.tabs + string)

    def __init__(self, filename_in, filename_out, speciesName="e", copyRNG=True, verbose=False):
        """
        initialization of manipulation routine

        This class writes particles named speciesName into a copy of the
        bp checkpoint given in filename_in into filename_out.

        Arguments:
        filename_in: string
                path to bp file to copy from (only time step 0 accepted)
        filename_out: string
                path to bp file to create
        speciesName: string
                short name in PIConGPU for the species to manipulate
        copyRNG: bool
                True: copy RNG values of the RNGProvider3XorMin field to
                new checkpoint, False: Do not copy RNG values
                Can be used to reduce memory consumption during copy process
                at the cost of reproducibility
        verbose: bool
                True: print output, False: Do not print output to screen
        """
        self.verbose = verbose  # verbose level
        self.tabs = 0  # tab counter for output
        self.copyRNG = copyRNG

        self.timestep = 0  # time step (fixed to 0)
        self.speciesName = speciesName
        self.filename_in = filename_in
        self.filename_out = filename_out
        self.f = io.Series(self.filename_in, io.Access.read_only)
        if int(list(self.f.iterations)[0]) != self.timestep:
            # throw error if not time step zero
            raise NameError("Not time step zero")
        # TODO: maybe raise error, when filename_out already exists
        # to not overwrite it

        tmp_handle = self.f.iterations[self.timestep].meshes["E"]

        # get cell size per dimension
        # the * unpacks the arguments for x,y,z
        self.cellSize = vec3D(*(tmp_handle.grid_unit_SI * np.array(tmp_handle.grid_spacing)))

        # extract number of cells in each dimension
        tmp_mesh = self.f.iterations[self.timestep].meshes["E"]["x"]

        tmp_handle = self.f.iterations[self.timestep].particles[self.speciesName].particle_patches["offset"]
        off_x = tmp_handle["x"].load()
        off_y = tmp_handle["y"].load()
        off_z = tmp_handle["z"].load()

        # get extent of each GPU in cells (per dimension)
        tmp_handle = self.f.iterations[self.timestep].particles[self.speciesName].particle_patches["extent"]
        ext_x = tmp_handle["x"].load()
        ext_y = tmp_handle["y"].load()
        ext_z = tmp_handle["z"].load()

        # get number of particles before each patch
        tmp_numOff = (
            self.f.iterations[self.timestep]
            .particles[self.speciesName]
            .particle_patches["numParticlesOffset"][io.Mesh_Record_Component.SCALAR]
            .load()
        )
        # get number of particles in each patch
        tmp_num = (
            self.f.iterations[self.timestep]
            .particles[self.speciesName]
            .particle_patches["numParticles"][io.Mesh_Record_Component.SCALAR]
            .load()
        )

        # flush all loads before calculations
        self.f.flush()

        tmp = np.shape(tmp_mesh)
        self.N_cells = vec3D(tmp[0], tmp[1], tmp[2])  # cells in each dimension

        # extract number of GPUs used in each dimension
        # use np.unique() to reduce patches offset and len() to get number
        # of GPUs per dimension
        self.N_gpus = vec3D(len(np.unique(off_x)), len(np.unique(off_y)), len(np.unique(off_z)))

        # get patch offset
        self.offset = vec3D(off_x, off_y, off_z)

        # get simulation box size in meter
        self.simBoxSize = vec3D(
            self.cellSize.x * self.N_cells.x,
            self.cellSize.y * self.N_cells.y,
            self.cellSize.z * self.N_cells.z,
        )

        self.extent = vec3D(ext_x, ext_y, ext_z)

        self.numParticlesOffset = tmp_numOff

        self.numParticles = tmp_num

        # raise error if there are particles in the checkpoint
        if np.sum(self.numParticles) > 0:
            raise NameError("There are particles in the checkpoint")

        # extract momentum unit (attributes don't need flushing)
        tmp_handle = self.f.iterations[self.timestep].particles[self.speciesName]["momentum"]["x"]
        self.unitMomentum = tmp_handle.unit_SI

        self.has_probeE = False
        self.has_probeB = False
        self.has_id = False

        # extract data type for position
        self.dtype_position = self.f.iterations[self.timestep].particles[self.speciesName]["position"]["x"].dtype

        # extract data type for positionOffset
        self.dtype_positionOffset = (
            self.f.iterations[self.timestep].particles[self.speciesName]["positionOffset"]["x"].dtype
        )

        # extract data type for momentum
        self.dtype_momentum = self.f.iterations[self.timestep].particles[self.speciesName]["momentum"]["x"].dtype

        if "probeE" in self.f.iterations[self.timestep].particles[self.speciesName]:
            self.has_probeE = True
            # type of E-Field
            self.dtype_probeE = self.f.iterations[self.timestep].particles[self.speciesName]["probeE"]["x"].dtype
        self.print("contains probeE = {}".format(self.has_probeE))

        if "probeB" in self.f.iterations[self.timestep].particles[self.speciesName]:
            self.has_probeB = True
            # type of B-Field
            self.dtype_probeB = self.f.iterations[self.timestep].particles[self.speciesName]["probeB"]["x"].dtype
        self.print("contains probeB {}".format(self.has_probeB))

        # extract data type for weighting
        self.dtype_weighting = (
            self.f.iterations[self.timestep]
            .particles[self.speciesName]["weighting"][io.Mesh_Record_Component.SCALAR]
            .dtype
        )

        if "id" in self.f.iterations[self.timestep].particles[self.speciesName]:
            self.has_id = True
            # type of particleID
            self.dtype_id = (
                self.f.iterations[self.timestep]
                .particles[self.speciesName]["id"][io.Mesh_Record_Component.SCALAR]
                .dtype
            )
        self.print("contains id =  {}".format(self.has_id))

        del self.f  # close checkpoint file

    def addParticles(self, pos, mom, w):
        """
        add particles to the restart file

        Arguments:
        pos - vec3 array
              position in SI units
        mom - vec3 array
              momentum in SI units
        w - float array
            macro particle weighting
        """
        self.N_particles_input = len(w)  # number of particles to add

        # calculate positionOffset (cell location) from given position
        self.positionOffset = vec3D(
            (pos.x / self.cellSize.x).astype(self.dtype_positionOffset),
            (pos.y / self.cellSize.y).astype(self.dtype_positionOffset),
            (pos.z / self.cellSize.z).astype(self.dtype_positionOffset),
        )

        # calculate (in cell) position from given position
        self.position = vec3D(
            (np.mod(pos.x, self.cellSize.x) / self.cellSize.x).astype(self.dtype_position),
            (np.mod(pos.y, self.cellSize.y) / self.cellSize.y).astype(self.dtype_position),
            (np.mod(pos.z, self.cellSize.z) / self.cellSize.z).astype(self.dtype_position),
        )

        # calculate momentum in PIC units from given momentum
        self.momentum = vec3D(
            (mom.x * w / self.unitMomentum).astype(self.dtype_momentum),
            (mom.y * w / self.unitMomentum).astype(self.dtype_momentum),
            (mom.z * w / self.unitMomentum).astype(self.dtype_momentum),
        )

        if self.has_probeE:
            # data for witnessed E-Field
            temp_zeros = np.zeros(len(w), dtype=self.dtype_probeE)
            self.probeE = vec3D(temp_zeros, temp_zeros, temp_zeros)

        if self.has_probeB:
            # data for witnessed B-Field
            temp_zeros = np.zeros(len(w), dtype=self.dtype_probeB)
            self.probeB = vec3D(temp_zeros, temp_zeros, temp_zeros)

        # copy weighting
        self.weighting = w.copy().astype(self.dtype_weighting)

        if self.has_id:
            # give every particle an ID
            self.id = np.arange(len(w), dtype=self.dtype_id)

    def makePatchMask(self):
        """
        calculate particle patches for given particles
        """
        # create empty patch mask (N_GPUs x N_particles)
        self.patch_mask = np.empty((self.N_gpus.prod(), self.N_particles_input), dtype=bool)

        # calculate patch  for each GPU
        for i in np.arange(self.N_gpus.prod()):
            # x direction
            a = np.greater_equal(self.positionOffset.x, self.offset.x[i])
            b = np.less(self.positionOffset.x, self.offset.x[i] + self.extent.x[i])

            # y direction
            c = np.greater_equal(self.positionOffset.y, self.offset.y[i])
            d = np.less(self.positionOffset.y, self.offset.y[i] + self.extent.y[i])

            # z direction:
            e = np.greater_equal(self.positionOffset.z, self.offset.z[i])
            f = np.less(self.positionOffset.z, self.offset.z[i] + self.extent.z[i])

            # combine all 3*2 bools to just give true or false for GPU(i)
            tmp1 = np.logical_and(np.logical_and(a, b), np.logical_and(c, d))
            tmp2 = np.logical_and(tmp1, np.logical_and(e, f))
            self.patch_mask[i, :] = tmp2

        # determine number of particles in all patches
        self.numParticles = np.sum(self.patch_mask, axis=1, dtype=np.uint)
        # calculate number of particles before the patch
        self.numParticlesOffset = np.cumsum(self.numParticles, dtype=np.uint) - self.numParticles
        # fix possible negative value for first patch (if number of particles
        # in first patch != 0)
        self.numParticlesOffset[0] = 0

    def writeParticles(self):
        """
        write all particle data to checkpoint with the help of the pipe class
        """
        self.print("make patch mask")
        self.makePatchMask()  # calculate particle patch
        self.N_particles = np.sum(self.numParticles)

        run_pipe = pipe(self.filename_in, self.filename_out, self, verbose=self.verbose)
        run_pipe.run()


class Chunk:
    """
    A Chunk is an n-dimensional hypercube, defined by an offset and an extent.
    Offset and extent must be of the same dimensionality (Chunk.__len__).
    """

    def __init__(self, offset, extent):
        """
        Inititalization of the slicing class.

        Arguments:
        offset: array
                offsets of each slice
        extent: array
                extent of each slice

        """
        assert len(offset) == len(extent)
        self.offset = offset
        self.extent = extent

    def __len__(self):
        return len(self.offset)

    def slice1D(self):
        """
        Slice this chunk into a hypercube along the dimension with
        the largest extent on this hypercube.

        Return:
        the 0'th of the sliced chunks.
        """
        # pick that dimension which has the highest count of items
        dimension = 0
        maximum = self.extent[0]
        for k, v in enumerate(self.extent):
            if v > maximum:
                dimension = k
        assert dimension < len(self)
        # no offset
        assert self.offset == [0 for _ in range(len(self))]
        offset = [0 for _ in range(len(self))]
        extent = self.extent.copy()

        return Chunk(offset, extent)


class particle_patch_load:
    """
    A deferred load/store operation for a particle patch.
    The openPMD particle-patch API requires that users pass a concrete value
    for storing, even if the actual write operation occurs much later at
    series.flush().
    So, unlike other record components, we cannot call .store_chunk() with
    a buffer that has not yet been filled, but must wait until the point where
    we actual have the data at hand already.
    In short: calling .store() must be deferred, until the data has been fully
    read from the sink.
    This class stores the needed parameters to .store().
    """

    def __init__(self, data, dest):
        self.data = data
        self.dest = dest

    def run(self):
        for index, item in enumerate(self.data):
            self.dest.store(index, item)


class pipe:
    """
    Represents the configuration of one "pipe" pass.
    """

    def print(self, string):
        """
        helper function that prints information depending on the set
        verbose level

        Arguments:
        string: string
                message to post
        """
        if self.verbose:
            print(string)

    def __init__(
        self,
        infile,
        outfile,
        particles=[],
        inconfig="{}",
        # can increase write performance with no known downsides, see:
        # https://adios2.readthedocs.io/en/latest/engines/engines.html#bp5
        outconfig="adios2.engine.parameters.BufferChunkSize = 2147381248",
        verbose=False,
    ):
        """
        routine to copy and overwrite data from one checkpoint to another.

        Arguments:
        infile: string
                path to the checkpoint to copy from
        outfile: string
                path to the checkpoint to copy to
        particles: class
                class containing the new particle data
        inconfig: string
                configuration file when opening checkpoint from infile
        outconfig: string
                configuration file when opening checkpoint from outfile
        verbose: bool
                prints verbose messages to the screen if True
        """
        self.infile = infile
        self.outfile = outfile
        self.particles = particles  # particle data to write
        self.inconfig = inconfig
        self.outconfig = outconfig
        self.verbose = verbose

    def run(self):
        """
        starts the copy routine.
        """
        print("Opening data source")
        sys.stdout.flush()
        inseries = io.Series(self.infile, io.Access.read_only, self.inconfig)
        print("Opening data sink")
        sys.stdout.flush()
        outseries = io.Series(self.outfile, io.Access.create, self.outconfig)
        print("Opened input and output")
        sys.stdout.flush()

        self.__copy(inseries, outseries)
        print("\nFinished!")

    def __copy(self, src, dest, current_path="/data/"):
        """
        Copies data from src to dest. May represent any point in the openPMD
        hierarchy, but src and dest must both represent the same layer.
        Writes own data for given particle.

        Arguments:
        src: openPMD layer
                layer of a openPMD series to copy from
        dest: openPMD layer
                layer of a openPMD series to copy to
        current_path: string
                path of the current layer, only for verbose printing.
        """
        self.print(current_path)
        sys.stdout.flush()

        if (
            type(src) is not type(dest)
            and not isinstance(src, io.IndexedIteration)
            and not isinstance(dest, io.Iteration)
        ):
            raise RuntimeError("Internal error: Trying to copy mismatching types")

        # copy attributes of current layer
        self.copy_attributes(src, dest)

        container_types = [
            io.Mesh_Container,
            io.Particle_Container,
            io.ParticleSpecies,
            io.Record,
            io.Mesh,
            io.Particle_Patches,
            io.Patch_Record,
        ]
        is_container = any([isinstance(src, container_type) for container_type in container_types])

        if isinstance(src, io.Series):
            # main loop: read iterations of src, write to dest
            write_iterations = dest.write_iterations()
            for in_iteration in src.read_iterations():
                print(
                    "Iteration {0} contains {1} meshes:".format(in_iteration.iteration_index, len(in_iteration.meshes))
                )
                for m in in_iteration.meshes:
                    print("\t {0}".format(m))
                print("")
                print(
                    "Iteration {0} contains {1} particle species:".format(
                        in_iteration.iteration_index, len(in_iteration.particles)
                    )
                )
                for ps in in_iteration.particles:
                    print("\t {0}".format(ps))
                    print("With records:")
                    for r in in_iteration.particles[ps]:
                        print("\t {0}".format(r))
                out_iteration = write_iterations[in_iteration.iteration_index]
                sys.stdout.flush()
                self.__particle_patches = []
                self.__copy(
                    in_iteration,
                    out_iteration,
                    current_path + str(in_iteration.iteration_index) + "/",
                )
                in_iteration.close()

                for patch_load in self.__particle_patches:
                    patch_load.run()

                # overwrite copied shape attribute of mass and charge to
                # match particle count
                out_iteration.particles[self.particles.speciesName]["mass"].set_attribute(
                    "shape", np.uint64(self.particles.N_particles)
                )
                out_iteration.particles[self.particles.speciesName]["charge"].set_attribute(
                    "shape", np.uint64(self.particles.N_particles)
                )

                out_iteration.close()
                self.__particle_patches.clear()
                sys.stdout.flush()

        elif isinstance(src, io.Record_Component) and (not is_container or src.scalar):
            # copies record components
            shape = src.shape
            dtype = src.dtype
            offset = [0 for _ in shape]
            dest.reset_dataset(io.Dataset(dtype, shape))
            if src.empty:
                # empty record component automatically created by
                # dest.reset_dataset()
                pass
            elif src.constant:
                dest.make_constant(src.get_attribute("value"))
            else:
                chunk = Chunk(offset, shape)
                local_chunk = chunk.slice1D()

                # write content of src record to dest record and
                # flush afterwards
                loaded_buffer = src.load_chunk(local_chunk.offset, local_chunk.extent)
                src.series_flush()
                dest.store_chunk(loaded_buffer, local_chunk.offset, local_chunk.extent)
                dest.series_flush()

        elif isinstance(src, io.Patch_Record_Component) and (not is_container or src.scalar):
            # copies patch record components
            dest.reset_dataset(io.Dataset(src.dtype, src.shape))
            self.__particle_patches.append(particle_patch_load(src.load(), dest))

        elif isinstance(src, io.Iteration):
            # copies iterations (meshes and particles)
            self.print("\ncopy meshes")
            self.__copy(src.meshes, dest.meshes, current_path + "meshes/")
            self.print("\ncopy particles")
            self.__copy(src.particles, dest.particles, current_path + "particles/")

        elif is_container:
            for key in src:
                # writes given particle data instead of copying
                if isinstance(src[key], io.ParticleSpecies) and key == self.particles.speciesName:
                    self.print("writing new particles data")
                    self.write_particles(src[key], dest[key], current_path + key)
                    self.print("resume to copying")
                # skips copying of RNGProvider3XorMin field to reduce memory if needed
                elif not self.particles.copyRNG and isinstance(src[key], io.Mesh) and key == "RNGProvider3XorMin":
                    self.print("skipped RNGProvider3XorMin")
                else:
                    self.__copy(src[key], dest[key], current_path + key + "/")

            if isinstance(src, io.ParticleSpecies):
                # copies particle patches of species
                self.__copy(src.particle_patches, dest.particle_patches, current_path + "particlePatches/")
        else:
            raise RuntimeError("Unknown openPMD class: " + str(src))

    def write_particles(self, src, dest, current_path="/data/"):
        """
        Write own data into the particle given in dest. Particle Patches,
        attributes and data types are copied from src.

        Arguments:
        src: openPMD layer
                layer of a openPMD series to copy attributes, particles patches
                and data types from
        dest: openPMD layer
                layer of a openPMD series to write new data to
        current_path: string
                path of the current layer, only for verbose printing.
        """
        # iterate and copy all attributes
        self.copy_attributes(src, dest, iterate=True)

        # write own particle data
        self.print("\twriting positions")
        self.write(
            src["position"]["x"],
            dest["position"]["x"],
            self.particles.position.x,
        )
        self.write(
            src["position"]["y"],
            dest["position"]["y"],
            self.particles.position.y,
        )
        self.write(
            src["position"]["z"],
            dest["position"]["z"],
            self.particles.position.z,
        )

        self.print("\twriting position offsets")
        self.write(
            src["positionOffset"]["x"],
            dest["positionOffset"]["x"],
            self.particles.positionOffset.x,
        )
        self.write(
            src["positionOffset"]["y"],
            dest["positionOffset"]["y"],
            self.particles.positionOffset.y,
        )
        self.write(
            src["positionOffset"]["z"],
            dest["positionOffset"]["z"],
            self.particles.positionOffset.z,
        )

        self.print("\twriting momenta")
        self.write(
            src["momentum"]["x"],
            dest["momentum"]["x"],
            self.particles.momentum.x,
        )
        self.write(
            src["momentum"]["y"],
            dest["momentum"]["y"],
            self.particles.momentum.y,
        )
        self.write(
            src["momentum"]["z"],
            dest["momentum"]["z"],
            self.particles.momentum.z,
        )

        if self.particles.has_probeE:
            self.print("\twriting probeE")
            self.write(
                src["probeE"]["x"],
                dest["probeE"]["x"],
                self.particles.probeE.x,
            )
            self.write(
                src["probeE"]["y"],
                dest["probeE"]["y"],
                self.particles.probeE.y,
            )
            self.write(
                src["probeE"]["z"],
                dest["probeE"]["z"],
                self.particles.probeE.z,
            )

        if self.particles.has_probeB:
            self.print("\twriting probeB")
            self.write(
                src["probeB"]["x"],
                dest["probeB"]["x"],
                self.particles.probeB.x,
            )
            self.write(
                src["probeB"]["y"],
                dest["probeB"]["y"],
                self.particles.probeB.y,
            )
            self.write(
                src["probeB"]["z"],
                dest["probeB"]["z"],
                self.particles.probeB.z,
            )

        self.print("\twriting weighting")
        self.write(
            src["weighting"][io.Mesh_Record_Component.SCALAR],
            dest["weighting"][io.Mesh_Record_Component.SCALAR],
            self.particles.weighting,
        )

        if self.particles.has_id:
            self.print("\twriting id")
            self.write(
                src["id"][io.Mesh_Record_Component.SCALAR],
                dest["id"][io.Mesh_Record_Component.SCALAR],
                self.particles.id,
            )

        # write own particle patches
        self.print("\tcopying patches")
        temp_src = src.particle_patches["numParticles"][io.Mesh_Record_Component.SCALAR]
        temp_dest = dest.particle_patches["numParticles"][io.Mesh_Record_Component.SCALAR]

        temp_dest.reset_dataset(io.Dataset(temp_src.dtype, temp_src.shape))
        self.__particle_patches.append(particle_patch_load(self.particles.numParticles, temp_dest))

        temp_src = src.particle_patches["numParticlesOffset"][io.Mesh_Record_Component.SCALAR]
        temp_dest = dest.particle_patches["numParticlesOffset"][io.Mesh_Record_Component.SCALAR]

        temp_dest.reset_dataset(io.Dataset(temp_src.dtype, temp_src.shape))
        self.__particle_patches.append(particle_patch_load(self.particles.numParticles, temp_dest))

        # copy offset and extent from old checkpoint
        temp_src = src.particle_patches["offset"]
        temp_dest = dest.particle_patches["offset"]

        for keyP in temp_src:
            self.__copy(temp_src[keyP], temp_dest[keyP], current_path + "/particlePatches/offset/" + keyP + "/")

        temp_src = src.particle_patches["extent"]
        temp_dest = dest.particle_patches["extent"]

        for keyP in temp_src:
            self.__copy(temp_src[keyP], temp_dest[keyP], current_path + "/particlePatches/extent/" + keyP + "/")

    def copy_attributes(self, src, dest, iterate=False):
        """
        Copies attributes from src to dest. Optionally iterates over them.

        Arguments:
        src: openPMD layer
                layer of a openPMD series to copy attributes from
        dest: openPMD layer
                layer of a openPMD series to copy attributes to
        iterate: Bool
                if True: iterates over all consecutive layers of the src/dest
                layer and copies attributes.
        """
        attribute_dtypes = src.attribute_dtypes
        # The following attributes are written automatically by openPMD-api
        # and should not be manually overwritten here
        ignored_attributes = {
            io.Series: ["basePath", "iterationEncoding", "iterationFormat", "openPMD"],
            io.Iteration: ["snapshot"],
            io.Record_Component: ["value", "shape"] if isinstance(src, io.Record_Component) and src.constant else [],
        }
        for key in src.attributes:
            ignore_this_attribute = False
            for openpmd_group, to_ignore_list in ignored_attributes.items():
                if isinstance(src, openpmd_group):
                    for to_ignore in to_ignore_list:
                        if key == to_ignore:
                            ignore_this_attribute = True
            if not ignore_this_attribute:
                attr = src.get_attribute(key)
                attr_type = attribute_dtypes[key]
                dest.set_attribute(key, attr, attr_type)
        if iterate:
            for key in src:
                self.copy_attributes(src[key], dest[key], iterate=True)

    def write(self, src, dest, data):
        """
        Writes new data to given record component in dest with data types
        from src.

        Arguments:
        src: openPMD layer
                layer of a openPMD series to copy datatypes from
        dest: openPMD layer
                layer of a openPMD series to copy new data to
        data: array of data which is written to dest.
        """
        shape = (self.particles.N_particles,)
        dtype = src.dtype

        dest.reset_dataset(io.Dataset(dtype, shape))
        # write particle data for each patch
        for i in range(self.particles.N_gpus.prod()):
            dest.store_chunk(
                array=data[self.particles.patch_mask[i, :]],
                offset=[self.particles.numParticlesOffset[i]],
                extent=(self.particles.numParticles[i],),
            )
