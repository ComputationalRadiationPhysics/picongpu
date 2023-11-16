import argparse
import sys
import numpy as np
import scipy.constants as spc
from openpmd_viewer import OpenPMDTimeSeries
import openpmd_viewer

print("openPMD-viewer version:", openpmd_viewer.__version__)
parser = argparse.ArgumentParser(description="1")


parser.add_argument(
    "-r",
    help="Path where the results of the test-suite should be saved",
    dest="data",
    type=str,
)

args = parser.parse_args()

# Initialize the OpenPMDTimeSeries object with the path to the data
ts = OpenPMDTimeSeries(args.data)

# Define parameters for electromagnetic fields and charge/energy densities
base_density = 1.0e25
gamma = 1.021
field_param = {
    "Ex": -30.0e2,
    "Ey": -20.0e4,
    "Ez": -10.0e6,
    "Bx": 25,
    "By": -50,
    "Bz": 50,
    "e_all_chargeDensity": -spc.elementary_charge * base_density,
    "e_all_energyDensity": spc.m_e * spc.c**2 * (gamma - 1) * base_density,
    "i_all_chargeDensity": spc.elementary_charge * base_density,
    "i_all_energyDensity": 0,
    "field_epsilon_relative": 1.0e-6,
    "field_epsilon": 1.0e-2,
    "density_epsilon_relative": 1.0e-5,
    "density_epsilon": 1.0e-2,
}

# Define parameters for particle properties
# Velocity of the particles calculated from gamma
v = gamma * (spc.c * (gamma**2 - 1) ** (1 / 2)) / spc.c

particle_param = {
    "ux": 0.267261 * v,
    "uy": 0.534522 * v,
    "uz": 0.801784 * v,
    "location_epsilon": 1.0e-1,
    "momentum_epsilon": 1.0e-1,
}

# Define grid parameters
grid_param = {
    "x_shape": 32,
    "y_shape": 32,
    "z_shape": 12,
}


# Function to check parameters against simulation data
def check_params(num_iterations):
    for i in range(num_iterations):
        # Check if the fields are correct
        fields = ["E", "B"]
        coords = ["x", "y", "z"]
        field_data = {}
        for field in fields:
            field_data[field] = {}
            for coord in coords:
                (
                    field_data[field][coord],
                    info,
                ) = ts.get_field(iteration=i, field=field, coord=coord)
        for field in fields:
            for coord in coords:
                field_param["field_epsilon_relative"] *= np.abs(field_param[f"{field}{coord}"])
                value = field_data[field][coord]
                # Check if the field values are within a tolerance range
                if not np.logical_and(
                    np.less_equal(
                        field_param[f"{field}{coord}"] - field_param["field_epsilon_relative"],
                        value,
                    ),
                    np.greater_equal(
                        field_param[f"{field}{coord}"] + field_param["field_epsilon_relative"],
                        value,
                    ),
                ).all():
                    return False
                if np.not_equal(field_param[f"{field}{coord}"], 0):
                    value -= field_param[f"{field}{coord}"]
                    if not np.less_equal(
                        np.abs(np.std(value) / field_param[f"{field}{coord}"]),
                        field_param["field_epsilon"],
                    ):
                        return False
                else:
                    if not np.std(value) == 0:
                        return False
        # Check if the densities are correct
        densities = [
            "e_all_chargeDensity",
            "e_all_energyDensity",
            "i_all_chargeDensity",
            "i_all_energyDensity",
        ]
        density_data = {}
        for density in densities:
            density_data[density] = {}
            density_data[density], info = ts.get_field(iteration=i, field=density)
            relative_error = field_param["density_epsilon_relative"] * np.abs(field_param[f"{density}"])
            value = density_data[density]
            # Check if the density values are within a tolerance range
            if not np.logical_and(
                np.less_equal(
                    field_param[f"{density}"] - relative_error,
                    density_data[density],
                ),
                np.greater_equal(
                    field_param[f"{density}"] + relative_error,
                    density_data[density],
                ),
            ).all():
                return False
            if np.not_equal(field_param[f"{density}"], 0):
                value -= field_param[f"{density}"]
                if not np.less_equal(
                    np.abs(np.std(value) / field_param[f"{density}"]),
                    field_param["density_epsilon"],
                ):
                    return False
            else:
                if not np.std(value) == 0:
                    return False

        x, y, z = ts.get_particle(
            var_list=["x", "y", "z"],
            iteration=i,
            species="e",
        )
        Ex, info_Ex = ts.get_field(iteration=0, field="E", coord="x")
        cell_dx = info_Ex.dx
        cell_dy = info_Ex.dy
        cell_dz = info_Ex.dz
        particle_map = np.zeros(shape=(np.shape(Ex)))
        # Calculate particle density map
        for k in range(len(x)):
            particle_map[
                int(z[k] / cell_dz),
                int(y[k] / cell_dy),
                int(x[k] / cell_dx),
            ] += 1
        # Check if the maximum and minimum particle counts are equal
        if not np.equal(np.amax(particle_map), np.amin(particle_map)):
            return False
        # Calculate the center map and check if values are within a tolerance
        center_map = [
            item
            for array in [
                (x % cell_dx) / cell_dx,
                (y % cell_dy) / cell_dy,
                (z % cell_dz) / cell_dz,
            ]
            for item in array
        ]
        for value in center_map:
            if not np.logical_and(
                np.less_equal(0.5 - particle_param["location_epsilon"], value),
                np.greater_equal(0.5 + particle_param["location_epsilon"], value),
            ).all():
                return False
        momenta = ["ux", "uy", "uz"]
        momentum_data = {}
        for momentum in momenta:
            momentum_data[momentum] = {}
            momentum_data[momentum] = [
                item
                for array in ts.get_particle(
                    var_list=[momentum],
                    iteration=i,
                    species="e",
                )
                for item in array
            ]
            # Check if the momentum values are within a tolerance range
        if not np.logical_and(
            np.less_equal(
                particle_param[f"{momentum}"] - particle_param["momentum_epsilon"],
                momentum_data[momentum],
            ),
            np.greater_equal(
                particle_param[f"{momentum}"] + particle_param["momentum_epsilon"],
                momentum_data[momentum],
            ),
        ).all():
            return False
    return True


bool_pass = check_params(2)  # false if failure else true if success
print("Test result: ", bool_pass)
sys.exit(not bool_pass)  # 0 if success else 1 if failure
