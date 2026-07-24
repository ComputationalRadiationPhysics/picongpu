# Copyright 2024-2024 Brian Marre, Tapish Narwal
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.


import openpmd_api as opmd
import numpy as np
import typeguard


@typeguard.typechecked
def validate(relativeAbundanceErrorThreshold: float = 0.01) -> bool:
    # read in Atomic Physics Data
    series = opmd.Series("binningOpenPMD/atomicStateBinning_000025.bp", opmd.Access.read_only)
    i = series.iterations[25]
    binning = i.meshes["Binning"]
    binning_component = binning["\vScalar"]
    binning_data = binning_component.load_chunk()
    series.flush()

    # loadReferenceData
    mean_reference = np.loadtxt("./validation/referenceData/mean_reference.data")

    # compare to reference
    #! @details may not use standard deviation from sample or reference, since we seem to consistently underestimate the
    # actual variation

    # check state has more than expected deviation if abundance is above 10^-5
    mask = mean_reference > 1.0e-5
    result = np.all(
        np.abs((binning_data / np.expand_dims(np.sum(binning_data, axis=1), axis=-1)) - mean_reference)[mask]
        < relativeAbundanceErrorThreshold
    )

    print(f"result of the test:{result}")

    return bool(result)


if __name__ == "__main__":
    validate()
