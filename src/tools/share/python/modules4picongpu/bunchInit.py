import numpy as np
import h5py


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
        """
        if(self.verbose):
            print("\t"*self.tabs + string)

    def __init__(self, filename, speciesName='e', verbose=True):
        """
        initialization of manipulation routine

        This class manupulates the particles named speciesName of the
        hdf5 checkpoint given in filename.

        Arguments:
        filename: string
                  path to hdf5 file to manipulate (only time step 0 accepted)
        speciesName: string
                     short name in PIConGPU for the species to manipulate
        verbose: bool
                 (True: print output, False: Do not print output to screen)
        """
        self.verbose = verbose  # verbose level
        self.tabs = 0  # tab counter for output

        self.timestep = 0 # time step (fixed to 0)
        self.speciesName = speciesName
        self.filename = filename
        self.f = h5py.File(self.filename, "r")
        if int(list(self.f['/data'].items())[0][0]) != self.timestep:
            raise NameError('Not time step zero') # throw wrror if not time step zero

        # extract number of cells in each dimension
        tmp_handle = self.f['/data/{}/particles/{}/weighting'.format(self.timestep,
                                                                     self.speciesName)]
        tmp = tmp_handle.attrs['_size']
        self.N_cells = vec3D(tmp[0], tmp[1], tmp[2]) # cells in each dimension

        # extract number of GPUs used in each dimension
        tmp_handle = self.f['/data/{}/particles/{}/particlePatches/offset'.format(self.timestep,
                                                                                  self.speciesName)]
        # use np.unique() to reduce patches offset and len() to get number of GPUs per dimension
        self.N_gpus = vec3D(len(np.unique(tmp_handle['x'])),
                            len(np.unique(tmp_handle['y'])),
                            len(np.unique(tmp_handle['z'])))

        # get patch offset
        self.offset = vec3D(tmp_handle['x'].value,
                            tmp_handle['y'].value,
                            tmp_handle['z'].value)

        # get cell size per dimension
        self.cellSize = vec3D(tmp_handle['x'].attrs['unitSI'],
                              tmp_handle['y'].attrs['unitSI'],
                              tmp_handle['z'].attrs['unitSI'])

        # get simulation box size in meter
        self.simBoxSize = vec3D(self.cellSize.x * self.N_cells.x,
                                self.cellSize.y * self.N_cells.y,
                                self.cellSize.z * self.N_cells.z)

        # get extent of each GPU in cells (per dimension)
        tmp_handle = self.f['/data/{}/particles/{}/particlePatches/extent'.format(self.timestep,
                                                                                  self.speciesName)]
        self.extent = vec3D(tmp_handle['x'].value,
                            tmp_handle['y'].value,
                            tmp_handle['z'].value)

        # get number of particles before each patch
        tmp_handle = self.f['/data/{}/particles/{}/particlePatches/numParticlesOffset'.format(self.timestep,
                                                                                              self.speciesName)]
        self.numParticlesOffset = tmp_handle.value

        # get number of particles in each patch
        tmp_handle = self.f['/data/{}/particles/{}/particlePatches/numParticles'.format(self.timestep,
                                                                                        self.speciesName)]
        self.numParticles = tmp_handle.value

        # raise error if there are particles in the checkpoint
        if np.sum(self.numParticles) > 0:
            raise NameError('There are particles in the checkpoint')

        # extract momentum unit
        tmp_handle = self.f['/data/{}/particles/{}/momentum/x'.format(self.timestep, self.speciesName)]
        self.unitMomentum = tmp_handle.attrs['unitSI']

        # extract data type for position
        self.dtype_position = self.f['/data/{}/particles/{}/position/x'.format(self.timestep, self.speciesName)].dtype

        # extract data type for positionOffset
        self.dtype_positionOffset = self.f['/data/{}/particles/{}/positionOffset/x'.format(self.timestep, self.speciesName)].dtype

        # extract data type for momentum
        self.dtype_momentum = self.f['/data/{}/particles/{}/momentum/x'.format(self.timestep, self.speciesName)].dtype

# extract data type for momentumPrev1
        self.dtype_momentumPrev1 = self.f['/data/{}/particles/{}/momentumPrev1/x'.format(self.timestep, self.speciesName)].dtype

        # extract data type for weighting
        self.dtype_weighting = self.f['/data/{}/particles/{}/weighting'.format(self.timestep, self.speciesName)].dtype

        self.f.close() # close checkpoint file



    def addParticles(self, pos, mom, w):
        """
        add particles to the hdf5 restart file

        Arguments:
        pos - vec3 array
              position in SI units
        mom - vec3 array
              momentum in SI units
        w - float array
            macro particle weighting
        """
        self.N_particles_input = len(w) # number of particles to add

        # calculate positionOffset (cell location) from given position
        self.positionOffset = vec3D((pos.x / self.cellSize.x).astype(int),
                                    (pos.y / self.cellSize.y).astype(int),
                                    (pos.z / self.cellSize.z).astype(int))

        # calculate (in cell) position from given position
        self.position = vec3D(np.mod(pos.x, self.cellSize.x)/self.cellSize.x,
                              np.mod(pos.y, self.cellSize.y)/self.cellSize.y,
                              np.mod(pos.z, self.cellSize.z)/self.cellSize.z)

        # calculate momentum in PIC units from given momentum
        self.momentum = vec3D(mom.x * w / self.unitMomentum,
                              mom.y * w / self.unitMomentum,
                              mom.z * w / self.unitMomentum)
        # copy weighting
        self.weighting = w.copy()


    def loadParticles(self, cpFileName):
        """
        load particle data from hdf5 checkpoint

        Arguments:
        cpFileName - string
                     path to PIConGPU checkpoint
        """

        f = h5py.File(cpFileName, "r") # open hdf5 file
        cpTimestep = (list(f['/data/'].items())[0])[0] # extract iteration

        # extract global offset due to moving window
        tmp = f['/data/{}/fields/E/x'.format(cpTimestep)].attrs['_global_start']
        globalStart = vec3D(tmp[0], tmp[1], tmp[2])

        # extract number of particles
        tmp = f['/data/{}/particles/{}/positionOffset'.format(cpTimestep, self.speciesName)]
        self.N_particles_input = len(tmp['x'])

        # convert positionOffset to a value without moving window and save
        self.positionOffset = vec3D(tmp['x'].value - globalStart.x,
                                    tmp['y'].value - globalStart.y,
                                    tmp['z'].value - globalStart.z)

        # save in cell position
        tmp = f['/data/{}/particles/{}/position'.format(cpTimestep, self.speciesName)]
        self.position = vec3D(tmp['x'].value,
                              tmp['y'].value,
                              tmp['z'].value)

        # save momentum
        tmp = f['/data/{}/particles/{}/momentum'.format(cpTimestep, self.speciesName)]
        self.momentum = vec3D(tmp['x'].value,
                              tmp['y'].value,
                              tmp['z'].value)

        # save weighting
        self.weighting = f['/data/{}/particles/{}/weighting'.format(cpTimestep,
                                                                    self.speciesName)].value
        f.close() # close hdf5 file


    def copyFields(self, cpFileName):
        """
        copy magnetic and electric field from hdf5 checkpoint to other checkpoint

        Arguments:
        cpFileName - string
                     path to PIConGPU checkpoint to extract data from
        """
        f_in = h5py.File(cpFileName, "r") # open hdf5 file to read data from
        cpTimestep = (list(f_in['/data/'].items())[0])[0] # extract iteration

        f_out = h5py.File(self.filename, "r+") # open hdf5 file to write data to

        # both E field handlers
        h_E_in = f_in['/data/{}/fields/E/'.format(cpTimestep)]
        h_E_out = f_out['/data/{}/fields/E/'.format(self.timestep)]

        # copy all components of E field
        for index in ['x', 'y', 'z']:
            self.print("E_"+index)
            h_E_out[index][:,:,:] = h_E_in[index][:,:,:]

        # both B field handlers
        h_B_in = f_in['/data/{}/fields/B/'.format(cpTimestep)]
        h_B_out = f_out['/data/{}/fields/B/'.format(self.timestep)]

        # copy all components of B field
        for index in ['x', 'y', 'z']:
            self.print("B_"+index)
            h_B_out[index][:,:,:] = h_B_in[index].value

        # close file handlers
        f_out.close()
        f_in.close()


    def makePatchMask(self):
        """
        calculate particle patches for given particles (previously added
        via addParticles or loadParticles method)
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
            tmp1 = np.logical_and( np.logical_and(a,b), np.logical_and(c,d) )
            tmp2 = np.logical_and(tmp1 , np.logical_and(e,f) )
            self.patch_mask[i, :] = tmp2

        # determine number of particles in all patches
        self.numParticles = np.sum(self.patch_mask, axis=1)
        # calculate number of particles before the patch
        self.numParticlesOffset = np.cumsum(self.numParticles) - self.numParticles
        # fix possible negative value for first patch (if number of particles in first patch != 0)
        self.numParticlesOffset[0] = 0


    def writePatch(self):
        """
        method that analysis particle distribution and writes the patch to the
        associated hdf5 checkpoint
        """
        f = h5py.File(self.filename, "r+")
        f['/data/{}/particles/{}/particlePatches/numParticles'.format(self.timestep, self.speciesName)][:] = self.numParticles
        f['/data/{}/particles/{}/particlePatches/numParticlesOffset'.format(self.timestep, self.speciesName)][:] = self.numParticlesOffset
        f.close()


    def writeParticleData(self, dsName, data, dtype):
        """
        write particle data into (empty) hdf5 check point

        Parameter:
        dsName - string
                 name of data set in hdf5 file
        dtype - data type
                data type of this data set
        """
        # calculate total number of particles to store from particle patch
        self.N_particles = np.sum(self.numParticles)
        f = h5py.File(self.filename, "r+") # open hdf5 checkpoint

        # handler to data set
        dsPath = '/data/{}/particles/{}/{}'.format(self.timestep, self.speciesName, dsName)
        # get all attributes (entire dict) of this data set
        attrs = f[dsPath].attrs.items()
        # remove entire data set
        del(f[dsPath])
        # create new data set for particles
        f.create_dataset(dsPath, (self.N_particles,), dtype=dtype)

        # copy attributes
        for i in attrs:
            f[dsPath].attrs.create(i[0], i[1])

        # copy particle data for each patch
        for i in np.arange(self.N_gpus.prod()):
            f[dsPath][self.numParticlesOffset[i]:
                      (self.numParticlesOffset[i]+self.numParticles[i])] = data[self.patch_mask[i, :]]

        f.close() # close checkpoint


    def writeParticles(self):
        """
        write all particle data to checkpoint
        """
        self.print("make patch mask")
        self.makePatchMask() # calculate particle patch

        # write all required attributes
        self.print("position")
        self.writeParticleData('position/x', self.position.x, self.dtype_position)
        self.writeParticleData('position/y', self.position.y, self.dtype_position)
        self.writeParticleData('position/z', self.position.z, self.dtype_position)

        self.print("positionOffset")
        self.writeParticleData('positionOffset/x', self.positionOffset.x, self.dtype_positionOffset)
        self.writeParticleData('positionOffset/y', self.positionOffset.y, self.dtype_positionOffset)
        self.writeParticleData('positionOffset/z', self.positionOffset.z, self.dtype_positionOffset)

        self.print("momentum")
        self.writeParticleData('momentum/x', self.momentum.x, self.dtype_momentum)
        self.writeParticleData('momentum/y', self.momentum.y, self.dtype_momentum)
        self.writeParticleData('momentum/z', self.momentum.z, self.dtype_momentum)

        self.print("momentumPrev1")
        self.writeParticleData('momentumPrev1/x', self.momentum.x, self.dtype_momentum)
        self.writeParticleData('momentumPrev1/y', self.momentum.y, self.dtype_momentum)
        self.writeParticleData('momentumPrev1/z', self.momentum.z, self.dtype_momentum)

        self.print("weighting")
        self.writeParticleData('weighting', self.weighting, self.dtype_weighting)

        # write particle patch
        self.print("patch")
        self.writePatch()


    def copyCheckpoint(self, cpFileName):
        """
        Copy data (fields and particles) from a checkpoint into the
        checkpoint associated with this class instance.

        Parameters:
        cpFileName: string
                    path to the checkpoint file to copy from
        """
        # load particle data
        self.print("load  particles")
        self.tabs +=1
        self.loadParticles(cpFileName)
        self.tabs -=1

        # write these particles to own checkpoint
        self.print("write particles")
        self.tabs +=1
        self.writeParticles()
        self.tabs -=1

        # copy field data
        self.print("copy fields")
        self.tabs +=1
        self.copyFields(cpFileName)
        self.tabs -=1
