import numpy as np
import scipy.constants as const
import sys
import os


# =============================================================================
# Pade Approximation Parameters for D-T Fusion Cross Section S-Factor
# =============================================================================
class PadeParams:
    """
    Holds the coefficients for the Pade approximation of the D-T fusion
    astrophysical S-factor. These parameters are typically valid for
    center-of-mass energies in the ~1-1000 keV range.
    """

    BG = 34.3827  # Gamow factor coefficient
    A1 = 6.927e4
    A2 = 7.454e8
    A3 = 2.050e6
    A4 = 5.2002e4
    A5 = 0.0
    B1 = 6.38e1
    B2 = -9.95e-1
    B3 = 6.981e-5
    B4 = 1.728e-4


# =============================================================================
# Core Fusion Calculation Functions
# =============================================================================


def sigma(E_rel_keV):
    """
    Calculates the D-T fusion cross-section in square meters (m^2).

    The cross-section is derived from the astrophysical S-factor, which is
    approximated using a Pade polynomial.

    Args:
        E_rel_keV (float): The relative kinetic energy in the center-of-mass
                           frame, in units of kilo-electron-volts (keV).

    Returns:
        float: The fusion cross-section in square meters (m^2).
    """
    p = PadeParams

    # The Pade approximation for the S-factor S(E)
    s_numerator = p.A1 + E_rel_keV * (p.A2 + E_rel_keV * (p.A3 + E_rel_keV * (p.A4 + E_rel_keV * p.A5)))
    s_denominator = 1.0 + E_rel_keV * (p.B1 + E_rel_keV * (p.B2 + E_rel_keV * (p.B3 + E_rel_keV * p.B4)))

    s_factor = s_numerator / s_denominator  # Result in keV-barns

    # The Gamow factor for tunneling probability
    gamow_factor = np.exp(-p.BG / np.sqrt(E_rel_keV))

    # Cross-section formula: sigma(E) = S(E)/E * GamowFactor
    # The result is in milibarns. 1 milibarn = 1e-31 m^2
    sigma_barns = (s_factor / E_rel_keV) * gamow_factor

    return sigma_barns * 1e-31  # Convert barns to m^2


def reactionRate(n1, n2, sigma_m2, v_rel):
    """Calculates reaction rate density (reactions per m^3 per second)."""
    return n1 * n2 * sigma_m2 * v_rel


def totalRate(n1, n2, sigma_m2, v_rel, vol):
    """Calculates total reactions per second in a given volume."""
    return n1 * n2 * sigma_m2 * v_rel * vol


def read_particle_count(run_folder: str, particle_name: str, final: bool = True):
    """
    Reads the final particle count from the energy histogram file.

    Args:
        run_folder: Path to the simulation run folder
        particle_name: Name of the particle (e.g., 'd', 't', 'He4', 'n')

    Returns:
        tuple: (final_count, file_path)
    """
    import pandas as pd

    filename = f"{particle_name}_energyHistogram_all.dat"
    path = os.path.join(run_folder, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{filename} not found in {path}")
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    if not final:
        final_total = float(df.iloc[0, -1])
    else:
        final_total = float(df.iloc[-1, -1])
    return final_total, path


def read_he4_final_count(run_folder: str):
    """Legacy function for backward compatibility."""
    return read_particle_count(run_folder, "He4")


# =============================================================================
# Main Simulation Function
# =============================================================================


def main():
    """
    Main function to set up parameters and calculate fusion reactions.
    """
    print("--- D-T Fusion Rate Simulation ---")

    # --- 1. Define Simulation and Particle Parameters ---
    n_D = 1e28  # Deuterium density (particles/m^3)
    n_T = 1e28  # Tritium density (particles/m^3)
    gamma = 1.001  # Lorentz factor for both clouds

    # Particle masses
    m_D = const.value("deuteron mass in u")  # Unified atomic mass units
    m_T = const.value("triton mass in u")

    # Simulation geometry
    CELL_WIDTH = 6e-7  # meters
    CELL_HEIGHT = 6e-7  # meters
    CELL_DEPTH = 6e-7  # meters
    GRID_X, GRID_Y, GRID_Z = 24, 24, 24

    # Simulation time
    DELTA_T = 1e-15  # Timestep in seconds
    TIMESTEPS = 500  # Number of timesteps to simulate

    # --- 2. Relativistic Kinematics Calculation ---
    # Calculate velocity of particles from gamma
    beta = np.sqrt(1 - 1 / gamma**2)

    # Calculate relativistic relative velocity for head-on collision
    v_rel = (2 * beta * const.c) / (1 + beta**2)

    # Calculate the relative Lorentz factor
    gamma_rel = gamma**2 * (1 + beta**2)  # Equivalent to 2*gamma**2 - 1

    # Calculate reduced mass
    mu_u = (m_D * m_T) / (m_D + m_T)
    mu_kg = mu_u * const.u

    # Calculate relative kinetic energy in the center-of-mass frame
    E_rel_joules = (gamma_rel - 1) * mu_kg * const.c**2

    # Convert energy to keV for the sigma function
    E_rel_keV = E_rel_joules / const.e / 1000.0

    print(f"Relativistic relative velocity: {v_rel / const.c:.3e} c ({v_rel:.3e} m/s)")
    print(f"Relative kinetic energy (CM): {E_rel_keV:.3e} keV")
    print("-" * 35)

    # --- 3. Cross-Section and Volume Calculation ---

    # CRITICAL WARNING for the user
    print("WARNING: The calculated relative energy is extremely high.")
    print("   The Pade approximation for the cross-section is being used")
    print("   far outside its valid physical range (~1-1000 keV).")
    print("   The following results are a numerical calculation based on the")
    print("   provided formula but are NOT physically realistic.")
    print("-" * 35)

    sigma_m2 = sigma(E_rel_keV)

    # Calculate total simulation volume
    cell_volume = CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH
    total_volume = cell_volume * GRID_X * GRID_Y * GRID_Z

    print(f"Calculated cross-section: {sigma_m2 / 1e-31:.3e} milibarns ({sigma_m2:.3e} m^2)")
    print(f"Total simulation volume: {total_volume:.3e} m^3")

    # --- 4. Final Reaction Calculation ---
    # Calculate the total reaction rate (reactions/sec) in the volume
    total_reactions_per_sec = totalRate(n_D, n_T, sigma_m2, v_rel, total_volume)

    # Calculate the number of reactions in a single timestep
    reactions_in_timestep = total_reactions_per_sec * DELTA_T * TIMESTEPS

    print("-" * 35)
    print(f"Total reactions per second: {total_reactions_per_sec:.3e}")
    print(f"Predicted number of reactions in {TIMESTEPS} timesteps ({DELTA_T} s): {reactions_in_timestep:.3e}")
    print("-" * 35)

    # --- 5. Read Actual Simulation Results and Compare ---
    # Check if data path was provided as command line argument
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

        print(f"Reading simulation data from: {data_path}")

        try:
            # Read all particle counts from energy histogram files
            counts_initial = {}
            counts_final = {}
            particle_names = ["d", "t", "He4", "n"]

            for name in particle_names:
                try:
                    final_count, file_path = read_particle_count(data_path, name)
                    initial_count, _ = read_particle_count(data_path, name, final=False)
                    counts_initial[name] = initial_count
                    counts_final[name] = final_count
                    print(f"Read {name} count from: {os.path.basename(file_path)}")
                except FileNotFoundError as e:
                    print(f"ERROR: {e}")
                    return 1

            # Display particle counts
            print("\nParticle Counts:")
            print(f"{'Particle':<10} {'Initial':<20} {'Final':<20} {'Change':<20}")
            print("-" * 70)
            for name in ["d", "t", "He4", "n"]:
                initial = counts_initial[name]
                final = counts_final[name]
                change = final - initial
                print(f"{name:<10} {initial:<20.3e} {final:<20.3e} {change:+20.3e}")
            print("-" * 70)

            # --- Validation Checks ---
            validation_passed = True

            # Check 1: He4 count matches prediction (within 20%)
            actual_he4_count = counts_final["He4"]
            percent_diff_he4 = abs(actual_he4_count - reactions_in_timestep) / reactions_in_timestep * 100
            print("\nCheck 1: He4 count vs prediction")
            print(f"   Predicted reactions: {reactions_in_timestep:.3e}")
            print(f"   Actual He4 particles: {actual_he4_count:.3e}")
            print(f"   Percentage difference: {percent_diff_he4:.3e}%")
            if percent_diff_he4 <= 20.0:
                print("   PASSED: Within 20% tolerance")
            else:
                print("   FAILED: Outside 20% tolerance")
                validation_passed = False

            # Check 2: Deuteron and Triton decrease should be equal (within 1%)
            d_decrease = abs(counts_initial["d"] - counts_final["d"])
            t_decrease = abs(counts_initial["t"] - counts_final["t"])
            print("\nCheck 2: Deuteron decrease vs Triton decrease")
            print(f"   Deuteron decrease: {d_decrease:.3e}")
            print(f"   Triton decrease: {t_decrease:.3e}")
            if d_decrease > 0 and t_decrease > 0:
                percent_diff_dt = abs(d_decrease - t_decrease) / max(d_decrease, t_decrease) * 100
                print(f"   Percentage difference: {percent_diff_dt:.3e}%")
                if percent_diff_dt <= 1.0:
                    print("   PASSED: Within 1% tolerance")
                else:
                    print("   FAILED: Outside 1% tolerance")
                    validation_passed = False
            else:
                print("   WARNING: No significant decrease detected")

            # Check 3: He4 and neutron production should be equal (within 1%)
            he4_produced = counts_final["He4"] - counts_initial["He4"]
            n_produced = counts_final["n"] - counts_initial["n"]
            print("\nCheck 3: He4 production vs Neutron production")
            print(f"   He4 produced: {he4_produced:.3e}")
            print(f"   Neutrons produced: {n_produced:.3e}")
            if he4_produced > 0 and n_produced > 0:
                percent_diff_he4n = abs(he4_produced - n_produced) / max(he4_produced, n_produced) * 100
                print(f"   Percentage difference: {percent_diff_he4n:.3e}%")
                if percent_diff_he4n <= 1.0:
                    print("   PASSED: Within 1% tolerance")
                else:
                    print("   FAILED: Outside 1% tolerance")
                    validation_passed = False
            else:
                print("   WARNING: No significant production detected")

            # Check 4: Conservation - decrease in reactants equals increase in products
            reactants_consumed = d_decrease + t_decrease
            products_created = he4_produced + n_produced
            print("\nCheck 4: Particle conservation (D+T consumed vs He4+n produced)")
            print(f"   Deuterons + Tritons consumed: {reactants_consumed:.3e}")
            print(f"   He4 + Neutrons produced: {products_created:.3e}")
            if reactants_consumed > 0 and products_created > 0:
                percent_diff_conservation = abs(reactants_consumed - products_created) / reactants_consumed * 100
                print(f"   Percentage difference: {percent_diff_conservation:.3e}%")
                if percent_diff_conservation <= 1.0:
                    print("   PASSED: Within 1% tolerance")
                else:
                    print("   FAILED: Outside 1% tolerance")
                    validation_passed = False
            else:
                print("   WARNING: No significant reaction activity detected")

            print("-" * 60)

            if validation_passed:
                print("\nALL VALIDATION CHECKS PASSED!")
                return 0
            else:
                print("\nSOME VALIDATION CHECKS FAILED!")
                return 1

        except Exception as e:
            print(f"ERROR reading or processing data files: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        print("INFO: No data path provided. Showing prediction only.")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
