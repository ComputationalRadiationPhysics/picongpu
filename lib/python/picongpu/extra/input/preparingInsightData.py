"""
This file is part of PIConGPU.

Copyright 2024 Fabia Dietrich
"""

import openpmd_api as openpmd
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

"""
ATTENTION!
----------
When saving the pulse data to openPMD, the pulse's time evolution at a specific
z position will be transformed to a spatial evolution along the propagation
direction via z=c*t. This approximation is only valid when the pulse length is
much smaller than a Rayleign length, because otherwise the true spatial
evolution is affected by defocusing.
BUT since the FromOpenPMDPulse profile transforms this axis back to a spatial
one by division by c, this will not lead to errors also if the pulse length is
of the order of a Rayleign length.
So for now, this transformation should be considered as "meaningless"; it is
only done to fulfil the openPMD requirements for field storage.
In principle, this should be refactored by using several iterations in the
openPMD file instead of just one, but this will complicate reading the file in
the PIConGPU initialization procedure (coding- and time wise).
"""

# please correct if other units than mm and fs are used!
c = 2.99792e-4  # speed of light in mm/fs


def gauss2D(xy, amp, x0, y0, w):
    """
    2D gaussian distribution

    Arguments:
    xy: coordinates (numpy meshgrid)
    amp: amplitude
    x0, y0: center coordinates
    w: waist size
    """
    x, y = xy
    g = amp * np.exp(-((x - x0) ** 2 / (w**2) + (y - y0) ** 2 / (w**2)))
    return g.ravel()


def supergauss2D(xy, amp, x0, y0, w, n=4):
    """
    2D supergaussian distribution

    Arguments:
    xy: coordinates (numpy meshgrid)
    amp: amplitude
    x0, y0: center coordinates
    w: waist size
    n: superpower (default is 4)
    """
    assert n > 2, "n > 2 required"
    x, y = xy
    g = amp * np.exp(-(((x - x0) ** 2 / (w**2) + (y - y0) ** 2 / (w**2)) ** (n - 2)))
    return g.ravel()


def gauss(t, amp, t0, tau):
    """
    1D gaussian distribution to fit the temporal envelope of the pulse in the time domain
    Definition of the pulse duration according to the GaussianPulse profile

    Arguments:
    t: time
    amp: amplitude
    t0: central time
    tau: pulse duration
    """
    return amp * np.exp(-((t - t0) ** 2) / (2 * tau) ** 2)


def lin(x, a, b):
    """
    linear function f(x) = a * x + b

    Arguments:
    x: function arguments
    a: slope
    b: offset
    """
    return a * x + b


class PrepRoutines:
    """
    class to manipulate and prepare Insight data for usage as FromOpenPMDPulse profile in PIConGPU:
        - phase corrections
        - artifact corrections in the near field
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
        Load Insight data (in focus = far field), propagate to near field
        and extract (gaussian) fit parameters.
        The far field has to be measured in dependence of the transverse coordinates "x", "y" (in mm) and
        frequency "w" (in rad/fs). If the scales are named differently or measured in another unit, they
        have to be adjusted manually here.

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

        self.lamb = 2 * np.pi * c / self.w  # wavelength in mm
        self.dx = np.diff(self.x)[0]  # transverse spacing
        self.dy = np.diff(self.y)[0]  # transverse spacing
        self.dw = np.diff(self.w)[0]  # frequency spacing
        self.wc_idx = int(len(self.w) / 2)  # central freq. index

        # fit far field data with gaussian
        # use intensity instead of amplitude to minimize possible halo influence
        popt, pcov = curve_fit(
            gauss2D,
            (np.meshgrid(self.x, self.y)),
            (np.abs(self.Ew[:, :, self.wc_idx]) ** 2).ravel(),
            p0=(1, 0, 0, 0.05),
        )
        self.xc = popt[1]  # x center coordinate
        self.yc = popt[2]  # y center coordinate
        self.waist = popt[3] * np.sqrt(2)  # waist size
        self.zR = self.waist**2 / (2 * c) * self.w[self.wc_idx]  # rayleigh length

        # (transverse) indices of main beam spot area (xmin, xmax, ymin, ymax)
        # this will result in a square area covering the beam spot
        waist_part = 0.7  # size of the square w.r.t waist size; should be < 1
        self.mainSpotIdx = np.array(
            [
                np.abs(self.x - self.xc + waist_part * self.waist).argmin(),
                np.abs(self.x - self.xc - waist_part * self.waist).argmin(),
                np.abs(self.y - self.yc + waist_part * self.waist).argmin(),
                np.abs(self.y - self.yc - waist_part * self.waist).argmin(),
            ]
        )

        # spectral intensity FWHM indices
        spec_int = np.sum(np.sum(np.abs(self.Ew) ** 2, axis=0), axis=0)
        spec_int = spec_int / spec_int.max()
        idx_in_FWHM = np.where(spec_int > 0.5)[0]
        self.spectFWHMIdx = np.array([idx_in_FWHM[0], idx_in_FWHM[-1]])

        # propagate to near field (right before the lens, i.e. d=0)
        self.ff_to_nf()
        self.dx_NF = np.diff(self.x_NF)[0]
        self.dy_NF = np.diff(self.y_NF)[0]

        # fit near field data with supergaussian
        popt_NF, pcov_NF = curve_fit(
            supergauss2D,
            (np.meshgrid(self.x_NF, self.y_NF)),
            np.abs(self.Ew_NF[:, :, self.wc_idx]).ravel(),
            p0=(1, 0, 0, 10),
        )
        self.xc_NF = popt_NF[1]  # x center coordinate
        self.yc_NF = popt_NF[2]  # y center coordinate
        self.waist_NF = popt_NF[3]  # waist size

        # (transverse) indices of main beam spot area (xmin, xmax, ymin, ymax)
        # this will result in a square area covering the beam spot
        self.mainSpotIdx_NF = np.array(
            [
                np.abs(self.x_NF - self.xc_NF + waist_part * self.waist_NF).argmin(),
                np.abs(self.x_NF - self.xc_NF - waist_part * self.waist_NF).argmin(),
                np.abs(self.y_NF - self.yc_NF + waist_part * self.waist_NF).argmin(),
                np.abs(self.y_NF - self.yc_NF - waist_part * self.waist_NF).argmin(),
            ]
        )

        self.isPhaseCorrected = False  # phase has not been corrected yet

        # print fit parameters etc.
        print("Central wavelength: %.f nm" % (self.lamb[self.wc_idx] * 10**6))
        print("Rayleigh length: zR = %.2f mm" % (self.zR))
        print("Far field center coordinates: xc = %.2f um, yc = %.2f um" % (self.xc * 1000, self.yc * 1000))
        print("Far field waist size: w = %.2f um" % (self.waist * 1000))
        print("Near field center coordinates: xc = %.2f mm, yc = %.2f mm" % (self.xc_NF, self.yc_NF))
        print("Near field waist size: w = %.2f mm" % (self.waist_NF))

    def ff_to_nf(self, d=0, method="linear"):
        """
        Propagate far field (in frequency domain) to near field at distance d in front of lens

        Arguments:
        d: distance of near field to lens, default is 0
        method: interpolation method of RegularGridInterpolator, default is 'linear'
        """
        X, Y, W = np.meshgrid(self.x, self.y, self.w)
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
        # rescaling from spatial frequency to space for every lambda (x = a * lamb * f)
        NF_resc = np.empty_like(NF, dtype=complex)
        scale = self.lamb * self.foc
        self.x_NF = np.linspace(a[0] * scale[-1], a[-1] * scale[-1], a.size)
        self.y_NF = np.linspace(b[0] * scale[-1], b[-1] * scale[-1], b.size)
        self.dx_NF = np.diff(self.x_NF)[0]
        self.dy_NF = np.diff(self.y_NF)[0]
        Y_NF, X_NF = np.meshgrid(self.y_NF, self.x_NF, indexing="ij")
        for i in range(self.w.size):
            interp_NF = RegularGridInterpolator((b * scale[i], a * scale[i]), NF[:, :, i], method=method)
            NF_resc[:, :, i] = interp_NF((Y_NF, X_NF))

        self.Ew_NF = NF_resc / np.abs(NF_resc).max()  # rescale amplitude to 1

    def nf_to_ff(self, d=0, method="linear"):
        """
        Propagate near field (in frequency domain) to far field from distance d in front of lens

        Arguments:
        d: distance to lens, default is 0
        method: interpolation method of RegularGridInterpolator, default is 'linear'
        """
        a = np.fft.fftshift(np.fft.fftfreq(self.x_NF.size, self.dx_NF))
        b = np.fft.fftshift(np.fft.fftfreq(self.y_NF.size, self.dy_NF))
        A, B, W = np.meshgrid(a, b, self.w)
        fac = (
            W
            / (1j * c * 2 * np.pi * self.foc)
            * np.exp(1j * 2 * np.pi**2 * self.foc * c / W * (A**2 + B**2) * (1 - d / self.foc))
        )
        FF = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.Ew_NF, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)) * fac
        # rescaling from spatial frequency to space for every lambda
        scale = self.lamb * self.foc
        self.x = np.linspace(a[0] * scale[-1], a[-1] * scale[-1], a.size)
        self.y = np.linspace(b[0] * scale[-1], b[-1] * scale[-1], b.size)
        self.dx = np.diff(self.x)[0]
        self.dy = np.diff(self.y)[0]
        Y, X = np.meshgrid(self.y, self.x, indexing="ij")
        for i in range(self.w.size):
            interp_FF = RegularGridInterpolator((b * scale[i], a * scale[i]), FF[:, :, i], method=method)
            self.Ew[:, :, i] = interp_FF((Y, X))

        self.Ew = self.Ew / np.abs(self.Ew).max()  # rescale amplitude to 1

    def correct_phase(self, GVD=0, TOD=0):
        """
        Since there is no information about the global phase of every wavelength measured with Insight,
        there has to be a phase correction.
        For that, we assume perfect compression of the beam center in the near field, thus we subtract a
        frequency's phase value at the beam center from every phase value in the frequency's transverse
        distribution in the near field.
        Furthermore, there is the possibility to add dispersion parameters (GVD, TOD) to the data.

        Arguments:
        GVD : Group velocity disperison in fs**2/rad,  optional. The default is 0.
        TOD : Third order dispersion in fs**3/rad**2, optional. The default is 0.
        """
        # phase in near field beam center
        phase_centr = np.unwrap(
            np.angle(self.Ew_NF[np.abs(self.y_NF - self.yc_NF).argmin(), np.abs(self.x_NF - self.xc_NF).argmin(), :])
        )
        dispersion = -GVD / 2 * (self.w - self.w[self.wc_idx]) ** 2 - TOD / 6 * (self.w - self.w[self.wc_idx]) ** 3

        # correcting near field
        self.Ew_NF = np.abs(self.Ew_NF) * np.exp(1j * (np.angle(self.Ew_NF) - phase_centr + dispersion))
        # correcting far field
        self.Ew = np.abs(self.Ew) * np.exp(1j * (np.angle(self.Ew) - phase_centr + dispersion))
        self.isPhaseCorrected = True
        print(
            "Phase has been corrected. \nApplied dispersion parameters: GVD = %.f fs**2/rad, TOD = %.f fs**3/rad**2"
            % (GVD, TOD)
        )

    def correct_ugly_spot_in_nf(self, x_ugly, y_ugly, uglybins=3, method="linear"):
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

        # safety check: are we still in the volume?
        assert xidx_low >= 0, "Sorry, too close to the left border!"
        assert yidx_low >= 0, "Sorry, too close to the lower border!"
        assert xidx_upp < len(self.x_NF), "Sorry, too close to the right border!"
        assert yidx_upp < len(self.y_NF), "Sorry, too close to the upper border!"

        # thickness of border surrounding the hole (number of bins)
        # for linear interpolation, 1 would be sufficient; but 3 is chosen so that cubic interpolation is still possible
        border_thickn = 3
        # safety check: are we still in the volume?
        assert xidx_low - border_thickn >= 0, "Sorry, too close to the left border!"
        assert xidx_upp + border_thickn < len(self.x_NF), "Sorry, too close to the right border!"

        # since it is pretty messy to work with meshgrids when there is a hole in the middle, we do the
        # interpolation only along the x direction (but any other direction would have been just as valid)
        x_surr = np.concatenate(
            (
                self.x_NF[xidx_low - border_thickn : xidx_low],
                self.x_NF[x_ugly_idx + uglybins : xidx_upp + border_thickn],
            )
        )
        Ew_surr = np.concatenate(
            (
                self.Ew_NF[yidx_low:yidx_upp, xidx_low - border_thickn : xidx_low, :],
                self.Ew_NF[yidx_low:yidx_upp, xidx_upp : xidx_upp + border_thickn, :],
            ),
            axis=1,
        )
        interp = RegularGridInterpolator((self.y_NF[yidx_low:yidx_upp], x_surr, self.w), Ew_surr, method=method)
        Y_hole, X_hole, W_hole = np.meshgrid(
            self.y_NF[yidx_low:yidx_upp], self.x_NF[xidx_low:xidx_upp], self.w, indexing="ij"
        )
        self.Ew_NF[yidx_low:yidx_upp, xidx_low:xidx_upp, :] = interp((Y_hole, X_hole, W_hole))

        # correct the far field
        self.nf_to_ff()

        print("corrected ugly spot in near field at x = %.2f mm, y = %.2f mm" % (x_ugly, y_ugly))

    def shift_nf_to_center(self):
        """
        If the near field is not centered around (0, 0), it can be done with this function.
        It automatically also corrects the far field by propagating the centered near field back into the focus.
        """
        # number of pixels to shift in x-direction
        nx = int(self.xc_NF / self.dx_NF + 0.5)
        # number of pixels to shift in y-direction
        ny = int(self.yc_NF / self.dy_NF + 0.5)

        # check shift for validity: should be less than half the size of the data extent
        assert (
            nx < len(self.x_NF) / 2 and ny < len(self.y_NF) / 2
        ), "Something's off, can't shift more than half the size of the data extent!"

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
            (np.meshgrid(self.x_NF, self.y_NF)),
            np.abs(self.Ew_NF[:, :, self.wc_idx]).ravel(),
            p0=(1, 0, 0, 10),
        )
        self.xc_NF = popt_NF[1]
        self.yc_NF = popt_NF[2]
        self.waist_NF = popt_NF[3]
        print("new near field center coordinates: xc = %.2f mm, yc = %.2f mm" % (self.xc_NF, self.yc_NF))

        # correct the far field
        self.nf_to_ff()

    def aperture_in_mf(self, d, R, xc=0, yc=0):
        """
        Put an aperture in the mid field.
        This function propagates far field to mid field, applies an aperture there and propagates back.

        Arguments:
        d: distance of aperture to focal plane (same unit as the transverse scales)
        R: radius of aperture (same unit as the transverse scales)
        xc, yc: center coordinates of aperture in mid field (w.r.t. the transverse scales)
        """

        # propagation FF to MF
        X, Y, W = np.meshgrid(self.x, self.y, self.w)
        fac = 1j / (2 * np.pi * c) * W / self.foc * np.exp(-1j * W / c / 2 / d * (X**2 + Y**2))
        MF = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Ew * fac, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        # transversal scales in mid field, but still as spatial frequencies
        a = np.fft.fftshift(np.fft.fftfreq(self.x.size, self.dx))  # = x / (lamb * d)
        b = np.fft.fftshift(np.fft.fftfreq(self.y.size, self.dy))  # = y / (lamb * d)

        # aperture = cropping MF content
        MF_cropped = np.zeros_like(MF, dtype=complex)
        for i in range(a.size):
            for j in range(b.size):
                for k in range(self.w.size):
                    if np.abs(a[i] * self.lamb[k] * d - xc) < R:
                        if np.abs(b[j] * self.lamb[k] * d - yc) < np.sqrt(R**2 - (a[i] * self.lamb[k] * d - xc) ** 2):
                            MF_cropped[j, i, k] = MF[j, i, k]

        # propagation MF to FF
        self.Ew = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(MF_cropped, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        # rescale amplitude to 1
        self.Ew = self.Ew / np.abs(self.Ew).max()

    def measure_ad_in_nf(self):
        """
        measure angular dispersion in the near field
        """
        # unwrapped phase in near field main beam spot area
        angle_AD = np.unwrap(
            np.unwrap(
                np.angle(
                    self.Ew_NF[
                        self.mainSpotIdx_NF[2] : self.mainSpotIdx_NF[3],
                        self.mainSpotIdx_NF[0] : self.mainSpotIdx_NF[1],
                        :,
                    ]
                ),
                axis=0,
            ),
            axis=1,
        )

        # calculating phase slope
        mx = []
        my = []
        for j in range(self.w.size):
            mx_j = []
            my_j = []
            for i in range(min(angle_AD.shape[0], angle_AD.shape[1])):
                # linear fit of every row and column of main beam spot area
                popt_x, pcov_x = curve_fit(
                    lin, self.x_NF[self.mainSpotIdx_NF[0] : self.mainSpotIdx_NF[1]], angle_AD[i, :, j], p0=(1, 0)
                )
                mx_j.append(popt_x[0])
                popt_y, pcov_y = curve_fit(
                    lin, self.y_NF[self.mainSpotIdx_NF[2] : self.mainSpotIdx_NF[3]], angle_AD[:, i, j], p0=(1, 0)
                )
                my_j.append(popt_y[0])
            # average phase slope for every lambda
            mx.append(np.average(mx_j))
            my.append(np.average(my_j))

        # calculating wavefront tilt angle
        alpha_x = np.arcsin(self.lamb * mx / (2 * np.pi))
        alpha_y = np.arcsin(self.lamb * my / (2 * np.pi))

        # AD = change of wavefront tilt angle with lambda
        # fitting just values inside the spectral intensity FHWM
        popt_x, pcov_x = curve_fit(
            lin,
            self.lamb[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]] * 10**6,
            alpha_x[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]],
        )
        perr_x = np.diag(pcov_x)
        AD_x = popt_x[0]
        e_AD_x = perr_x[0]

        popt_y, pcov_y = curve_fit(
            lin,
            self.lamb[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]] * 10**6,
            alpha_y[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]],
        )
        perr_y = np.diag(pcov_y)
        AD_y = popt_y[0]
        e_AD_y = perr_y[0]

        print("measured AD in NF in x direction: %.2e +- %.2e rad/nm" % (AD_x, e_AD_x))
        print("measured AD in NF in y direction: %.2e +- %.2e rad/nm" % (AD_y, e_AD_y))

    def measure_ad_in_ff(self):
        """
        measure angular dispersion in the far field
        """
        # unwrapped phase in far field main beam spot area
        angle_AD = np.unwrap(
            np.unwrap(
                np.angle(
                    self.Ew[self.mainSpotIdx[2] : self.mainSpotIdx[3], self.mainSpotIdx[0] : self.mainSpotIdx[1], :]
                ),
                axis=0,
            ),
            axis=1,
        )

        # calculating phase slope
        mx = []
        my = []
        for j in range(self.w.size):
            mx_j = []
            my_j = []
            for i in range(min(angle_AD.shape[0], angle_AD.shape[1])):
                # linear fit of every row and column of main beam spot area
                popt_x, pcov_x = curve_fit(
                    lin, self.x[self.mainSpotIdx[0] : self.mainSpotIdx[1]], angle_AD[i, :, j], p0=(-200, 0)
                )
                mx_j.append(popt_x[0])
                popt_y, pcov_y = curve_fit(
                    lin, self.y[self.mainSpotIdx[2] : self.mainSpotIdx[3]], angle_AD[:, i, j], p0=(-200, 0)
                )
                my_j.append(popt_y[0])
            # average phase slope for every lambda
            mx.append(np.average(mx_j))
            my.append(np.average(my_j))

        # calculating wavefront tilt angle
        alpha_x = np.arcsin(self.lamb * mx / (2 * np.pi))
        alpha_y = np.arcsin(self.lamb * my / (2 * np.pi))

        # AD = change of wavefront tilt angle with lambda
        # fitting just values inside the spectral intensity FHWM
        popt_x, pcov_x = curve_fit(
            lin,
            self.lamb[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]] * 10**6,
            alpha_x[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]],
        )
        perr_x = np.diag(pcov_x)
        AD_x = popt_x[0]
        e_AD_x = perr_x[0]

        popt_y, pcov_y = curve_fit(
            lin,
            self.lamb[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]] * 10**6,
            alpha_y[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]],
        )
        perr_y = np.diag(pcov_y)
        AD_y = popt_y[0]
        e_AD_y = perr_y[0]

        print("measured AD in FF in x direction: %.2e +- %.2e rad/nm" % (AD_x, e_AD_x))
        print("measured AD in FF in y direction: %.2e +- %.2e rad/nm" % (AD_y, e_AD_y))

    def measure_sd_in_nf(self):
        """
        measure spatial dispersion in the near field
        """
        # lists to store center coordinates
        xc_SD = []
        yc_SD = []

        X_NF, Y_NF = np.meshgrid(self.x_NF, self.y_NF)

        # supergaussian fit to near field amplitude to extract center coordinates
        # only inside the spectral intensity FWHM
        for i in range(self.spectFWHMIdx[1] - self.spectFWHMIdx[0]):
            popt, pcov = curve_fit(
                supergauss2D,
                (X_NF, Y_NF),
                (np.abs(self.Ew_NF[:, :, self.spectFWHMIdx[0] + i])).ravel(),
                p0=(1, self.xc_NF, self.yc_NF, self.waist_NF),
            )
            xc_SD.append(popt[1])
            yc_SD.append(popt[2])

        # linear fit to central x coordinate
        popt, pcov = curve_fit(lin, self.lamb[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]] * 10**6, xc_SD)
        perr = np.diag(pcov)
        SDx = popt[0]
        eSDx = perr[0]
        print("measured SD in NF in x direction: %.2e +- %.2e mm / nm" % (SDx, eSDx))

        popt, pcov = curve_fit(lin, self.lamb[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]] * 10**6, yc_SD)
        perr = np.diag(pcov)
        SDy = popt[0]
        eSDy = perr[0]
        print("measured SD in NF in y direction: %.2e +- %.2e mm / nm" % (SDy, eSDy))

    def measure_sd_in_ff(self):
        """
        measure spatial dispersion in the far field
        """
        # lists to store center coordinates
        xc_SD = []
        yc_SD = []

        X, Y = np.meshgrid(self.x, self.y)

        # supergaussian fit to near field amplitude to extract center coordinates
        # only inside the spectral intensity FWHM
        for i in range(self.spectFWHMIdx[1] - self.spectFWHMIdx[0]):
            popt, pcov = curve_fit(
                gauss2D,
                (X, Y),
                (np.abs(self.Ew[:, :, self.spectFWHMIdx[0] + i]) ** 2).ravel(),
                p0=(1, self.xc, self.yc, self.waist),
            )
            xc_SD.append(popt[1])
            yc_SD.append(popt[2])

        # linear fit to central x coordinate
        popt, pcov = curve_fit(lin, self.lamb[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]] * 10**6, xc_SD)
        perr = np.diag(pcov)
        SDx = popt[0]
        eSDx = perr[0]
        print("measured SD in FF in x direction:  %.2e +- %.2e mm / nm" % (SDx, eSDx))

        popt, pcov = curve_fit(lin, self.lamb[self.spectFWHMIdx[0] : self.spectFWHMIdx[1]] * 10**6, yc_SD)
        perr = np.diag(pcov)
        SDy = popt[0]
        eSDy = perr[0]
        print("measured SD in FF in y direction:  %.2e +- %.2e mm / nm" % (SDy, eSDy))

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
        ), "Oops, you wanted to propagate too far! The pulse will not fit in the transverse window."

        a = 2 * np.pi * c * np.fft.fftfreq(self.x.size, d=np.diff(self.x)[0])
        b = 2 * np.pi * c * np.fft.fftfreq(self.y.size, d=np.diff(self.y)[0])
        A, B, W = np.meshgrid(a, b, self.w)

        # depending on the following mask, the wavenumber k_z (= spatial frequency in propagation direction)
        # is either complex or real, resulting in plane or evanescent waves, respectively. For a detailed
        # derivation, please refer to Godmans "Introduction to Fourier Optics".
        cond = (A**2 + B**2) / W**2
        idx_in = np.where(cond < 1)
        idx_out = np.where(cond > 1)
        F_Ew = np.fft.fft2(self.Ew, axes=(0, 1))
        F_Ew[idx_in] *= np.exp(1j * W[idx_in] / c * np.sqrt(1 - cond[idx_in]) * z)
        F_Ew[idx_out] *= np.exp(-W[idx_out] / c * np.sqrt(cond[idx_out] - 1) * z)
        lin_phase = np.exp(-1j * W * z / c)  # cancel out the time shift

        print("far field propagated to z = %.2f mm" % (z))
        return np.fft.ifft2(F_Ew, axes=(0, 1)) * lin_phase

    def to_time_domain(self, Ew, lamb_supp=10):
        """
        Transform far field from frequency to time domain

        Arguments:
        Ew: far field data in frequency domain (at the focus or propagated)
        lamb_supp: number of sampling points on one wavelength;
                   from this, the number of zeros to be padded will be derived. Default is 10.
        """
        assert self.isPhaseCorrected, "Oops, you forgot to correct the phase!"

        # calculate the number of zeros to be padded at the right side of the spectrum
        # longitudinal axis length N_tot = 2 * pi / (dw * dt)
        # necessary time spacing dt = lamb / (c * lamb_supp)
        zeros = int(self.w[self.wc_idx] * lamb_supp / self.dw / 2 - self.w[-1] / self.dw)

        # the data has to be sorted according to fft requirements:  w = dw * (0, 1, ... n/2, -n/2, ... -1)
        # so we have to extent (and interpolate) first the positive side of the spectrum
        w_fft = np.arange(0, self.w[-1] + zeros * self.dw, self.dw)
        Y_fft, X_fft, W_fft = np.meshgrid(self.y, self.x, w_fft, indexing="ij")
        fft_interp = RegularGridInterpolator(
            (self.y, self.x, self.w), Ew, bounds_error=False, fill_value=0, method="linear"
        )
        Ew_fft = fft_interp((Y_fft, X_fft, W_fft))
        # set the negative frequency part to 0 (instead of the complex conjugate) and neglect the imaginary part afterwards
        w_fft = np.concatenate((w_fft, -np.flip(w_fft[1:])))
        Ew_fft = np.concatenate((Ew_fft, np.zeros_like(Ew_fft[:, :, :-1])), axis=-1)

        # transform to time domain
        self.Et = np.fft.fftshift(np.fft.ifft(Ew_fft), axes=-1)
        # rescale the amplitude to one
        self.Et = self.Et / np.abs(self.Et).max()
        self.t = np.fft.fftshift(np.fft.fftfreq(w_fft.size, self.dw / (2 * np.pi)))  # fs
        self.dt = np.diff(self.t)[0]

        # estimate the pulse length
        popt, pcov = curve_fit(
            gauss,
            self.t,
            np.abs(self.Et[np.abs(self.y - self.yc).argmin(), np.abs(self.x - self.xc).argmin(), :]),
            p0=(1, 0, 15),
        )
        print("Pulse duration: %.f fs / FHWM intensity: %.f fs" % (popt[2], 2 * np.sqrt(2 * np.log(2)) * popt[2]))

        print("field data size: %.1f GB" % (np.prod(self.Et.shape) * 8 * 10**-9))  # assumes datatype double

    def save_to_openPMD(self, outputpath, outputname, energy, pol="x", crop_x=0, crop_y=0, crop_t=0):
        """
        Save the field data in time domain to an openPMD file. This output will be used for the
        FromOpenPMDPulse profile.
        The pulse's time evolution at a specific z position will be transformed to a spatial evolution along
        the propagation direction via z=c*t. ATTENTION: This approximation is only valid when the pulse length
        is much smaller than a Rayleign length, because otherwise the true spatial evolution is affected by
        defocusing. BUT since the FromOpenPMDPulse profile transforms this axis back to a spatial one by division
        by c, this will not lead to errors also if the pulse length is of the order of a Rayleign length.
        So this transformation should be considered as "meaningless"; it is only done to fulfil the openPMD
        requirements for field storage.
        In principle, this can be refactored by using several iterations in the openPMD file instead of just
        the 0th, but this will complicate reading the file in the PIConGPU initialization procedure.

        Arguments:
        outputpath: path to the openPMD file
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

        series = openpmd.Series(outputpath + outputname, openpmd.Access.create)
        ite = series.iterations[0]  # use the 0th iteration
        ite.time = 0.0
        ite.dt = self.dt
        ite.time_unit_SI = 1.0e-15

        # record E-field in 2+1 spatial dimensions with 3 components
        E = ite.meshes["E"]
        E.geometry = openpmd.Geometry.cartesian
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
        data_E = openpmd.Dataset(E_save.dtype, E_save.shape)
        E_pol.reset_dataset(data_E)
        E_trans.reset_dataset(data_E)
        E_z.reset_dataset(data_E)

        # unit system agnostic dimension
        E.unit_dimension = {
            openpmd.Unit_Dimension.M: 1,
            openpmd.Unit_Dimension.L: 1,
            openpmd.Unit_Dimension.I: -1,
            openpmd.Unit_Dimension.T: -3,
        }

        # conversion of field data to SI
        # total field energy: W = dV * eps0/2 * sum(E**2 + c**2 * B**2) in Joule
        # B_trans = +- E_pol / c
        # B_pol = 0
        # B_z = -+ i/w * d/d_trans E_pol; but neglectable since B_z << B_trans
        # 1st sign for pol = "x", 2nd for "y"
        dV = self.dx * self.dy * self.dt * c * 10**-9  # m**3
        W = dV * 8.854e-12 * np.sum(E_save**2)
        # scaling to actual beam energy
        amp_fac = np.sqrt(energy / W)
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
        )  # assumes datatype double
