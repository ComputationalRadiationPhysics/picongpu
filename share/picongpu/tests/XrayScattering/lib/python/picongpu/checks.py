from os.path import join
import numpy as np
import openpmd_api as api
from picongpu.plugins.data import XrayScatteringData
from is_close import is_close


def compare_with_fft(species, bound_electrons, rotation=None):
    simulation_path = '../../../../'

    # Load pluginOutput
    xray_scattering_data = XrayScatteringData(simulation_path, species, 'h5')
    amplitude = xray_scattering_data.get(iteration=0)
    del xray_scattering_data

    # Load density
    internal_path = 'simOutput/h5'
    file_name = 'simData_%T.h5'
    path_output = join(simulation_path, internal_path, file_name)
    series_output = api.Series(path_output, api.Access_Type.read_only)
    i = series_output.iterations[0]
    e_mesh = i.meshes['e_density']
    ed = e_mesh[api.Mesh_Record_Component.SCALAR]
    electron_density = ed.load_chunk()
    # ions have the same density in this setup
    electron_density *= bound_electrons
    series_output.flush()

    # Transform data
    # (SideZ)
    if electron_density.ndim == 3:
        # zyx(openPMD) -> xyz(PIC) -> yxz(beam Side z)
        electron_density = np.moveaxis(electron_density, (0, 1, 2), (2, 0, 1))
    # for dim == 2 nothing changes xy are swiped twice.
    if rotation is not None:
        electron_density = rotation(electron_density)
    fft = np.fft.fftn(electron_density)
    if electron_density.ndim == 3:
        fft = fft[:, :, 0]  # Take the z=0 slice.
    fft = np.fft.fftshift(fft)
    # Now some magic. Since x_beam = -1 * y_PIC (side z) we need to do the
    # equivalent transformation q_x -> -q_x. The [1:,:] is necessary since the
    # fft output has one extra, mismatching after reflection, frequency. It is
    # left out of the comparision.
    fft, amplitude = fft[1:, 1:], amplitude[1:, 1:]
    fft = fft[::-1, :]

    fft = fft.astype(amplitude.dtype.type)
    if amplitude.real.dtype.type is np.float32:
        params = {"abs_tolerance": 1e-1,
                  "threshold": 1e-1, "rel_tolerance": 1e-1}
    elif amplitude.real.dtype.type is np.float64:
        params = {"abs_tolerance": 1e-8,
                  "threshold": 1e-8, "rel_tolerance": 1e-8}
    else:
        raise TypeError

    check_real = is_close(amplitude.real, fft.real, **params)
    check_imag = is_close(amplitude.imag, fft.imag, **params)
    return check_real and check_imag


def check_summation():
    simulation_path = '../../../../'
    # Load pluginOutput
    xray_scattering_data = XrayScatteringData(simulation_path, 'e', 'h5')
    amplitude0 = xray_scattering_data.get(iteration=0)
    amplitude1 = xray_scattering_data.get(iteration=1)
    del xray_scattering_data
    difference = amplitude1 - amplitude0
    if amplitude0.real.dtype.type is np.float32:
        params = {"abs_tolerance": 1e-4,
                  "threshold": 1e-2, "rel_tolerance": 1e-3}
    elif amplitude0.real.dtype.type is np.float64:
        params = {"abs_tolerance": 1e-12,
                  "threshold": 1e-11, "rel_tolerance": 1e-11}
    else:
        raise TypeError
    real_check = is_close(difference.real, amplitude0.real, **params)
    imag_check = is_close(difference.imag, amplitude0.imag, **params)
    return real_check and imag_check
