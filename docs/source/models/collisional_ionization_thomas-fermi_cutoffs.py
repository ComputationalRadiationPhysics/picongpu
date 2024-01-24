import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


params = {
    "font.size": 20,
    "lines.linewidth": 3,
    "legend.fontsize": 20,
    "legend.frameon": False,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    # RTD default textwidth: 800px
    "figure.figsize": [12, 8],
}
mpl.rcParams.update(params)


class ThomasFermiIonization:
    """
    Thomas-Fermi fitting model (@see More 1985) for statistical collisional
    ionization.
    """

    def __init__(self, proton_number, mass_number):
        """Initialize model and read input parameters

        Args:
            proton_number : Proton number of the ion
            mass_number   : Atomic mass number of the ion
        """
        # input parameters
        self.Z = proton_number
        self.massA = mass_number

        # TF fitting parameters
        self.a_1 = 0.003323
        self.a_2 = 0.9718
        self.a_3 = 9.26148e-5
        self.a_4 = 3.10165

        self.b_0 = -1.7630
        self.b_1 = 1.43175
        self.b_2 = 0.31546

        self.c_1 = -0.366667
        self.c_2 = 0.983333

        self.alpha = 14.3139
        self.beta = 0.6624

    def CalcT_0(self, temperature, protonNumber):
        T = temperature
        return T / np.power(protonNumber, (4.0 / 3.0))

    def CalcR(self, density, massNumber, protonNumber):
        rho = density
        return rho / (protonNumber * massNumber)

    def CalcT_F(self, T_0):
        return T_0 / (1 + T_0)

    def CalcA(self, T_0):
        return self.a_1 * np.power(T_0, self.a_2) + self.a_3 * np.power(T_0, self.a_4)

    def CalcB(self, T_F):
        return -np.exp(self.b_0 + self.b_1 * T_F + self.b_2 * np.power(T_F, 7.0))

    def CalcC(self, T_F):
        return self.c_1 * T_F + self.c_2

    def CalcQ_1(self, A_TF, R, B):
        """
        :param: A_TF the parameter A from ThomasFermi and NOT the mass number
        """
        return A_TF * np.power(R, B)

    def CalcQ(self, R, Q_1, C):
        R_1 = np.power(R, C) + np.power(Q_1, C)
        return np.power(R_1, (1.0 / C))

    def Calcx(self, Q):
        return self.alpha * np.power(Q, self.beta)

    def CalcZStar(self, x, proton_number):
        Z = proton_number
        return Z * x / (1.0 + x + np.sqrt(1.0 + 2.0 * x))

    def TFIonState(self, temperature, density):
        """
        Args:
            temperature   : Electron "temperature" [unit: eV]
            density       : Ion mass density [unit: g/cm^3]
        """

        self.T = temperature
        self.rho = density

        # calculation
        T_0 = self.CalcT_0(self.T, self.Z)
        T_F = self.CalcT_F(T_0)
        A = self.CalcA(T_0)
        R = self.CalcR(self.rho, self.massA, self.Z)
        B = self.CalcB(T_F)
        C = self.CalcC(T_F)
        Q_1 = self.CalcQ_1(A, R, B)
        Q = self.CalcQ(R, Q_1, C)
        x = self.Calcx(Q)

        ionState = self.CalcZStar(x, self.Z)
        return ionState


if __name__ == "__main__":
    """
    On execution this script produces the figure showing the Thomas-Fermi model
    issues and cutoffs for the PIConGPU documentation.
    """

mass_density = 10 ** np.linspace(-2, 2, 1000)  # g/cm^3
temp_array = np.array([0, 10, 100])  # eV

Z_H = 1
A_H = 1
Z_C = 6
A_C = 12

TF_H = ThomasFermiIonization(Z_H, A_H)
TF_C = ThomasFermiIonization(Z_C, A_C)

# initial alpha value for the plots
alpha = 1.0
# linestyles to distinguish the plots further
linestyles = ["solid", "dashed", "dotted"]
# index for linestyles
i = 0

for temp in temp_array:
    CS_H = TF_H.TFIonState(temp, mass_density)
    CS_C = TF_C.TFIonState(temp, mass_density)

    plt.plot(
        mass_density,
        CS_H,
        label="H @ {} eV".format(temp),
        color="blue",
        alpha=alpha,
        ls=linestyles[i],
    )
    plt.plot(
        mass_density,
        CS_C,
        label="C @ {} eV".format(temp),
        color="orange",
        alpha=alpha,
        ls=linestyles[i],
    )

    # reduce alpha value to differentiate between electron temperatures
    alpha -= 0.1
    # increment i
    i += 1

plt.xscale("log")
plt.ylabel(r"Charge State Prediction $\langle Z \rangle$")
plt.xlabel(r"Mass Density [g/cm$^3$]")

filter_dens = np.less_equal(mass_density, 2.2) * np.greater_equal(mass_density, 0.8)
plastic_density = mass_density[filter_dens]
plt.fill_between(plastic_density, y1=6, y2=0, alpha=0.5, color="green")
plt.text(x=1.0, y=5, s="typ. plastic density", color="white", rotation=90)
plt.ylim(0, 6)
plt.legend(loc="upper left")

plt.draw()
plt.show()
