import numpy as np
import openpmd_api as opmd


def generateReference():
    outputs = [
        "output_1",
        "output_2",
        "output_3",
        "output_4",
        "output_5",
        "output_6",
        "output_7",
        "output_8",
        "output_9",
        "output_10",
    ]

    results = np.empty((10, 26, 869))
    for j, output in enumerate(outputs):
        series = opmd.Series(output + "/binningOpenPMD/atomicStateBinning_000025.bp", opmd.Access.read_only)

        i = series.iterations[25]
        binning = i.meshes["Binning"]
        binning_component = binning["\vScalar"]
        binning_data = binning_component.load_chunk()
        series.flush()

        results[j] = binning_data / np.expand_dims(np.sum(binning_data, axis=1), axis=-1)

    mean = np.mean(results, axis=0)
    stdDev = np.std(results, axis=0, ddof=1)

    np.savetxt("./validation/referenceData/mean_reference.data", mean)
    np.savetxt("./validation/referenceData/stdDev_reference.data", stdDev)


if __name__ == "__main__":
    generateReference()
