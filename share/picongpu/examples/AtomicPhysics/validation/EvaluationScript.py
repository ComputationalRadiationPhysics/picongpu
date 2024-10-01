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

    # check state has more than expected deviation
    result = np.all(
        np.abs((binning_data / np.expand_dims(np.sum(binning_data, axis=1), axis=-1)) - mean_reference)
        < relativeAbundanceErrorThreshold
    )

    print(f"result of the test:{result}")

    return bool(result)


if __name__ == "__main__":
    validate()
