"""
This file is part of PIConGPU.

Copyright 2024 Fabia Dietrich
"""

import openpmd_api as io
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

# please correct if other units than mm and fs are used!
c = 2.99792e-4  # speed of light in mm/fs


def gauss2D(xy, A, x0, y0, w):
    """
    2D gaussian distribution

    Arguments:
    xy: coordinates
    A: amplitude
    x0, y0: center coordinates
    w: waist size
    """
    x, y = xy
    g = A * np.exp(-((x - x0) ** 2 / (w**2) + (y - y0) ** 2 / (w**2)))
    return g.ravel()


def supergauss2D(xy, A, x0, y0, w, n=4):
    """
    2D supergaussian distribuition

    Arguments:
    xy: coordinates
    A: amplitude
    x0, y0: center coordinates
    w: waist size
    n: superpower (default is 4)
    """
    assert n > 2, "n > 2 required"
    x, y = xy
    g = A * np.exp(-(((x - x0) ** 2 / (w**2) + (y - y0) ** 2 / (w**2)) ** (n - 2)))
    return g.ravel()


class preproutines:
    """
    class to manipulate and prepare Insight data for PIConGPU usage:
        - phase corrections
        - artefact corrections in the near field
        - propagation
        - transformation into the time domain

    Members:
    --------
    Ew: measured far field in dependence of x, y, w
    x, y: transversal far field scales in mm
    xc, yc: far field center coordinates
    dx, dy: far field coordinate spacing
    waist: waist size in focus (w0)
    zR: Rayleigh length estimated from w0

    w: frequency scale in rad/fs
    wc_idx: central frequency index
    dw: frequency spacing

    Ew_NF: reconstructed near field
    x_NF, y_NF: transversal near field scales
    xc_NF, yc_NF: near field center coordinates
    dx_NF, dy_NF: coordinate spacing
    waist_NF: waist size in near field

    Et: field in the time domain
    t, dt: time axis and spacing
    """

    def __init__(self, path, name, foc):
        """
        Load Insight data (= far field), propagate to near field
        and extract (gaussian) fit parameters.
        The far field has to be measured in dependence of "x", "y" and "w".
        If the scales are named differently, they have to be changed manually in this function.

        Arguments:
        path: path to Insight data file
        name: name of Insight data file (.h5)
        foc:  focal length in mm
        """

        self.path = path
        self.name = name
        self.foc = foc

        # read Insight far field data
        f = h5py.File(self.path + self.name, "r")
        groups = list(f.keys())
        self.Ew = np.array(f["data/{}".format(*f["/{}".format(groups[0])].keys())])  # 3D array (x, y, w)
        # change scale name here if necessary
        self.x = np.array(f["scales/x"])  # mm
        self.y = np.array(f["scales/y"])  # mm
        self.w = np.array(f["scales/w"])  # rad/fs
        f.close()

        self.dx = np.diff(self.x)[0]
        self.dy = np.diff(self.y)[0]
        self.dw = np.diff(self.w)[0]
        self.wc_idx = int(len(self.w) / 2)  # central freq. index

        # fit far field data with gaussian
        # use intensity instead of amplitude to minimize possible halo influence
        popt, pcov = curve_fit(
            gauss2D,
            (np.meshgrid(self.y, self.x)),
            (np.abs(self.Ew[:, :, self.wc_idx]) ** 2).ravel(),
            p0=(1, 0, 0, 0.05),
        )
        self.xc = popt[1]
        self.yc = popt[2]
        self.waist = popt[3] * np.sqrt(2)
        self.zR = self.waist**2 / (2 * c) * self.w[self.wc_idx]

        # propagate to near field (right before the lens, i.e. d=0)
        preproutines.FFtoNF(self)
        self.dx_NF = np.diff(self.x_NF)[0]
        self.dy_NF = np.diff(self.y_NF)[0]

        # fit near field data with supergaussian
        popt_NF, pcov_NF = curve_fit(
            supergauss2D,
            (np.meshgrid(self.y_NF, self.x_NF)),
            np.abs(self.Ew_NF[:, :, self.wc_idx]).ravel(),
            p0=(1, 0, 0, 10),
        )
        self.xc_NF = popt_NF[1]
        self.yc_NF = popt_NF[2]
        self.waist_NF = popt_NF[3]

        self.isPhaseCorrected = False  # phase has not been corrected yet

        # print fit parameters etc.
        print("Central wavelength: %.f nm" % (c * 2 * np.pi / self.w[self.wc_idx] * 10**6))
        print("Rayleigh length: zR = %.2f mm" % (self.zR))
        print("Far field center coordinates: xc = %.2f um, yc = %.2f um" % (self.xc * 1000, self.yc * 1000))
        print("Far field waist size: w = %.2f um" % (self.waist * 1000))
        print("Near field center coordinates: xc = %.2f mm, yc = %.2f mm" % (self.xc_NF, self.yc_NF))
        print("Near field waist size: w = %.2f mm" % (self.waist_NF))

    def FFtoNF(self, d=0, method="linear"):
        """
        Propagate far field (in frequency domain) to near field at distance d in front of lens

        Arguments:
        d: distance of near field to lens, default is 0
        method: interpolation method of RegularGridInterpolator, default is 'linear'
        """
        Y, X, W = np.meshgrid(self.y, self.x, self.w)
        fac = (
            1j
            / (2 * np.pi * c)
            * W
            / self.foc
            * np.exp(-1j * W / c / 2 / self.foc * (X**2 + Y**2) * (1 - d / self.foc))
        )
        NF = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Ew * fac, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        a = np.fft.fftshift(np.fft.fftfreq(self.x.size, self.dx))
        b = np.fft.fftshift(np.fft.fftfreq(self.y.size, self.dy))
        # rescaling from spatial frequency to space for every lambda (x = a*lambda*f)
        NF_resc = np.empty_like(NF, dtype=complex)
        scale = 2 * np.pi * self.foc * c / self.w
        self.x_NF = np.linspace(a[0] * scale[-1], a[-1] * scale[-1], a.size)
        self.y_NF = np.linspace(b[0] * scale[-1], b[-1] * scale[-1], b.size)
        Y_NF, X_NF = np.meshgrid(self.y_NF, self.x_NF, indexing="ij")
        for i in range(self.w.size):
            interp_NF = RegularGridInterpolator((b * scale[i], a * scale[i]), NF[:, :, i], method=method)
            NF_resc[:, :, i] = interp_NF((Y_NF, X_NF))

        self.Ew_NF = NF_resc / np.abs(NF_resc).max()  # rescale amplitude to 1

    def NFtoFF(self, d=0, method="linear"):
        """
        Propagate near field (in frequency domain) to far field from distance d in front of lens

        Arguments:
        d: distance to lens, default is 0
        method: interpolation method of RegularGridInterpolator, default is 'linear'
        """
        a = np.fft.fftshift(np.fft.fftfreq(self.x_NF.size, self.dx_NF))
        b = np.fft.fftshift(np.fft.fftfreq(self.y_NF.size, self.dy_NF))
        B, A, W = np.meshgrid(b, a, self.w)
        fac = (
            W
            / (1j * c * 2 * np.pi * self.foc)
            * np.exp(1j * 2 * np.pi**2 * self.foc * c / W * (A**2 + B**2) * (1 - d / self.foc))
        )
        FF = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.Ew_NF, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)) * fac
        # rescaling from spatial frequency to space for every lambda
        scale = 2 * np.pi * self.foc * c / self.w
        self.x = np.linspace(a[0] * scale[-1], a[-1] * scale[-1], a.size)
        self.y = np.linspace(b[0] * scale[-1], b[-1] * scale[-1], b.size)
        Y, X = np.meshgrid(self.y, self.x, indexing="ij")
        for i in range(self.w.size):
            interp_FF = RegularGridInterpolator((a * scale[i], b * scale[i]), FF[:, :, i], method=method)
            self.Ew[:, :, i] = interp_FF((Y, X))

        self.Ew = self.Ew / np.abs(self.Ew).max()  # rescale amplitude to 1

    def correctPhase(self, GVD=0, TOD=0):
        """
        Since there is no information about the global phase of every wavelength measured with Insight,
        there has to be a phase correction.
        For that, we assume perfect compression of the beam center in the near field, thus we substract
        a global phase for every wavelength.
        Furthermore, there is the possibility to add dispersion parameters (GVD, TOD) to the data.

        Arguments:
        GVD : Group velocity disperion in fs**2/rad,  optional. The default is 0.
        TOD : Third order dispersion in fs**3/rad**2, optional. The default is 0.
        """
        # phase in near field beam center
        phase_centr = np.unwrap(
            np.angle(self.Ew_NF[np.abs(self.y_NF - self.yc_NF).argmin(), np.abs(self.x_NF - self.xc_NF).argmin(), :])
        )
        dispersion = -GVD / 2 * (self.w - self.w[self.wc_idx]) ** 2 - TOD / 6 * (self.w - self.w[self.wc_idx]) ** 3

        self.Ew_NF = np.abs(self.Ew_NF) * np.exp(
            1j * (np.angle(self.Ew_NF) - phase_centr + dispersion)
        )  # correcting near field
        self.Ew = np.abs(self.Ew) * np.exp(1j * (np.angle(self.Ew) - phase_centr + dispersion))  # correcting far field

        self.isPhaseCorrected = True
        print(
            "Phase has been corrected. \nApplied dispersion parameters: GVD = %.f fs**2/rad, TOD = %.f fs**3/rad**2"
            % (GVD, TOD)
        )

    def correctUglySpotInNF(self, x_ugly, y_ugly, uglybins=3, method="linear"):
        """
        In case there is an ugly spot in the near field (such as an unnatural peak), it can be smoothened out with this function.
        It automatically also corrects the far field by propagating the corrected near field back into the focus.

        Arguments:
        x_ugly: (near field) x coordinate of the ugly spot
        y_ugly: (near field) y coordinate of the ugly spot
        uglybins: number of bins surrounding the ugly spot which will be cut away / smoothened out (default is 3)
        method: RegularGridInterpolator interpolation method,
                default is 'linear' (since 'cubic' can cause some bumps in amplitude or phase)
        """
        # index borders of hole
        x_ugly_idx = np.abs(self.x_NF - x_ugly).argmin()
        y_ugly_idx = np.abs(self.y_NF - y_ugly).argmin()
        xidx_low = x_ugly_idx - uglybins
        xidx_upp = x_ugly_idx + uglybins
        yidx_low = y_ugly_idx - uglybins
        yidx_upp = y_ugly_idx + uglybins

        x_surr = np.concatenate((self.x_NF[xidx_low - 3 : xidx_low], self.x_NF[x_ugly_idx + uglybins : xidx_upp + 3]))
        Ew_surr = np.concatenate(
            (
                self.Ew_NF[yidx_low:yidx_upp, xidx_low - 3 : xidx_low, :],
                self.Ew_NF[yidx_low:yidx_upp, xidx_upp : xidx_upp + 3, :],
            ),
            axis=1,
        )
        interp = RegularGridInterpolator((self.y_NF[yidx_low:yidx_upp], x_surr, self.w), Ew_surr, method=method)
        Y_hole, X_hole, W_hole = np.meshgrid(
            self.y_NF[yidx_low:yidx_upp], self.x_NF[xidx_low:xidx_upp], self.w, indexing="ij"
        )
        self.Ew_NF[yidx_low:yidx_upp, xidx_low:xidx_upp, :] = interp((Y_hole, X_hole, W_hole))

        # correct the far field
        preproutines.NFtoFF(self)

        print("corrected ugly spot in near field at x = %.2f mm, y = %.2f mm" % (x_ugly, y_ugly))

    def shiftNFtoCenter(self):
        """
        If the near field is not centered around (0, 0), it can be done with this function.
        It automatically also corrects the far field by propagating the centered near field back into the focus.
        """
        nx = int(self.xc_NF / self.dx_NF + 0.5)  # number of pixels to shift in x-direction
        ny = int(self.yc_NF / self.dy_NF + 0.5)  # number of pixels to shift in y-direction

        if nx == 0 and ny == 0:
            print("already centered, no corrections necessary")
            return 0

        if nx != 0:
            if nx > 0:
                self.Ew_NF[:, :-nx, :] = self.Ew_NF[:, nx:, :]
            else:
                self.Ew_NF[:, -nx:, :] = self.Ew_NF[:, :nx, :]
        if ny != 0:
            if ny > 0:
                self.Ew_NF[:-ny, :, :] = self.Ew_NF[ny:, :, :]
            else:
                self.Ew_NF[-ny:, :, :] = self.Ew_NF[:ny, :, :]

        # new fit of near field data with supergaussian
        popt_NF, pcov_NF = curve_fit(
            supergauss2D,
            (np.meshgrid(self.y_NF, self.x_NF)),
            np.abs(self.Ew_NF[:, :, self.wc_idx]).ravel(),
            p0=(1, 0, 0, 10),
        )
        self.xc_NF = popt_NF[1]
        self.yc_NF = popt_NF[2]
        self.waist_NF = popt_NF[3]
        print("new near field center coordinates: xc = %.2f mm, yc = %.2f mm" % (self.xc_NF, self.yc_NF))

        # correct the far field
        preproutines.NFtoFF(self)

    def propagate(self, z):
        """
        Propagate the far field (in frequency domain) using the angular spectrum method.
        Should be called after correcting the phase.
        Propagation leads to a linear phase term resulting in a time shift (by t = z/c). This linear phase term
        will be removed since it
            1. probably cannot be resolved by the w spacing anway and
            2. thus shifts the intensity maximum in the time domain to unexpected places,
               maybe even to the border of the sampling interval.

        Arguments:
        z: propagation distance in mm

        Returns:
        propagated (complex) field data
        """
        assert self.isPhaseCorrected, "Oops, you forgot to correct the phase!"

        # check that propagated beam will still fit into the window
        w_exp = self.waist * np.sqrt(1 + (z / self.zR) ** 2)
        assert w_exp < 0.2 * min(
            self.dx * self.Ew.shape[0], self.dy * self.Ew.shape[1]
        ), "Oops, you wanted to propagate too far!"

        a = np.fft.fftfreq(self.x.size, d=self.dx)
        b = np.fft.fftfreq(self.y.size, d=self.dy)
        B, A, W = np.meshgrid(b, a, self.w)
        cond = (2 * np.pi * c / W) ** 2 * (A**2 + B**2)
        idx_in = np.where(cond < 1)
        idx_out = np.where(cond > 1)
        F_Ew = np.fft.fft2(self.Ew, axes=(0, 1))
        F_Ew[idx_in] *= np.exp(1j * W[idx_in] / c * np.sqrt(1 - cond[idx_in]) * z)
        F_Ew[idx_out] *= np.exp(-W[idx_out] / c * np.sqrt(cond[idx_out] - 1) * z)
        lin_phase = np.exp(-1j * W * z / c)  # cancel out the time shift

        print("far field propagated to z = %.2f mm" % (z))
        return np.fft.ifft2(F_Ew, axes=(0, 1)) * lin_phase

    def toTimeDomain(self, Ew, lamb_supp=10):
        """
        Transform far field from frequency to time domain

        Arguments:
        Ew: far field data in frequency domain (at the focus or propagated)
        lamb_supp: number of sampling points on one wavelength;
                   from this, the number of zeros to be padded will be derived. Default is 10.
        """
        assert self.isPhaseCorrected, "Oops, you forgot to correct the phase!"

        # calculate the number of zeros to be padded at the right side of the spectrum
        # longitudinal axis length N_tot = 2*pi/(dw*dt)
        # necessary time spacing dt = lambda/(c*lamb_supp)
        zeros = int(self.w[self.wc_idx] * lamb_supp / self.dw / 2 - self.w[-1] / self.dw)

        # the data has to be sorted according to fft requirements:  w = dw* (0, 1, ... n/2, -n/2, ... -1)
        # so we have to extent (and interpolate) first the positive side of the spectrum
        w_fft = np.arange(0, self.w[-1] + zeros * self.dw, self.dw)
        Y_fft, X_fft, W_fft = np.meshgrid(self.y, self.x, w_fft, indexing="ij")
        fft_interp = RegularGridInterpolator(
            (self.y, self.x, self.w), Ew, bounds_error=False, fill_value=0, method="linear"
        )
        Ew_fft = fft_interp((Y_fft, X_fft, W_fft))
        # set the negative frequency part to 0 (instead of the complex conjugate) and neglegt the imaginary part afterwards
        w_fft = np.concatenate((w_fft, -np.flip(w_fft[1:])))
        Ew_fft = np.concatenate((Ew_fft, np.zeros_like(Ew_fft[:, :, :-1])), axis=-1)

        # transform to time domain
        self.Et = np.fft.fftshift(np.fft.ifft(Ew_fft), axes=-1)
        self.Et = self.Et / np.abs(self.Et).max()  # rescale amplitude to one
        self.t = np.fft.fftshift(np.fft.fftfreq(w_fft.size, self.dw / (2 * np.pi)))  # fs
        self.dt = np.diff(self.t)[0]

        print("field data size: %.1f GB" % (np.prod(self.Et.shape) * 8 * 10**-9))  # assume datatype double

    def saveToOpenPMD(self, outputpath, outputname, energy, pol="x", crop_x=0, crop_y=0, crop_t=0):
        """
        Save the field data in time domain to an OpenPMD checkpoint. This output will be used for the InsightPulse profile.
        The propagation axis ("z") will be transformed via z = c*t to a spatial one.

        Arguments:
        outputpath: path to the OpenPMD checkpoint
        outputname: name of the output file, e.g. "insightData%T.h5"
        pol: polarisation direction, either "x" or "y". Default is "x"
        energy: beam energy in Joule. Is used to determine the correct amplitude in the time domain.
                For that, the approximation z = c*t is used, which holds only for tau << zR/c!
        crop_x/y/t: if the field data chunk is too big, one can crop the borders from both sides by the length
                    given with these values (same unit as the corresponding scales). Default is 0.
        """
        assert pol == "x" or pol == "y", "Oops, invalid polarisation direction!"
        idx_x = int(crop_x / self.dx + 0.5)
        idx_y = int(crop_y / self.dy + 0.5)
        idx_t = int(crop_t / self.dt + 0.5)
        Nx, Ny, Nt = np.shape(self.Et)
        # we only need the real part of the field
        E_save = np.array(np.real(self.Et[idx_y : Ny - idx_y, idx_x : Nx - idx_x, idx_t : Nt - idx_t]))

        series = io.Series(outputpath + outputname, io.Access.create)
        ite = series.iterations[0]  # use the 0th iteration
        ite.time = 0.0
        ite.dt = self.dt
        ite.time_unit_SI = 1.0e-15

        # record E-field in 2+1 spatial dimensions with 3 components
        E = ite.meshes["E"]
        E.geometry = io.Geometry.cartesian
        E.grid_spacing = [self.dy, self.dx, c * self.dt]  # mm
        E.grid_global_offset = [0, 0, 0]  # mm
        E.grid_unit_SI = 1e-3  # mm to m
        E.axis_labels = ["y", "x", "z"]
        E.data_order = "C"
        E_pol = E[pol]
        if pol == "x":
            E_trans = E["y"]
        else:
            E_trans = E["x"]
        E_z = E["z"]
        data_E = io.Dataset(E_save.dtype, E_save.shape)
        E_pol.reset_dataset(data_E)
        E_trans.reset_dataset(data_E)
        E_z.reset_dataset(data_E)

        # unit system agnostic dimension
        E.unit_dimension = {
            io.Unit_Dimension.M: 1,
            io.Unit_Dimension.L: 1,
            io.Unit_Dimension.I: -1,
            io.Unit_Dimension.T: -3,
        }

        # conversion of field data to SI
        # total field energy: W = dV * eps0/2 * sum(E**2 + c**2 * B**2) in Joule
        # B_trans = +- E_pol / c
        # B_pol = 0
        # B_z = -+ i/w * d/d_trans E_pol; but neglectable since B_z << B_trans
        # 1st sign for pol = "x", 2nd for "y"
        threshold = 0.0  # just consider (relative) field values above threshold
        idx_thres = np.where(np.abs(E_save) > threshold)
        dV = self.dx * self.dy * self.dt * c * 10**-9  # m**3
        W = dV * 8.854e-12 * np.sum(E_save[idx_thres] ** 2)
        amp_fac = np.sqrt(energy / W)  # scaling to actual beam energy
        print("Maximum amplitude: %.2e V/m" % (amp_fac))

        E_pol.unit_SI = amp_fac
        E_trans.unit_SI = 0.0
        E_z.unit_SI = 0.0

        # register and flush chunk
        E_pol.store_chunk(E_save)
        E_trans.make_constant(0.0)
        E_z.make_constant(0.0)
        series.flush()

        del series

        print(
            "data successfully saved, field data size: %.f MB" % (np.prod(E_save.shape) * 8 * 10**-6)
        )  # assume datatype double
