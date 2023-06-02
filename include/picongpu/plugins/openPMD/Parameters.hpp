#pragma once

#include <string>
#include <tuple>

namespace picongpu::openPMD
{
    struct PluginParameters
    {
        std::string fileName = "simData"; /* Name of the openPMDSeries, excluding the extension */
        std::string fileInfix = "_%06T";
        std::string fileExtension = "bp"; /* Extension of the file name */
        std::string dataPreparationStrategyString = "doubleBuffer";
        std::string jsonConfigString = "{}";
        std::string rangeString = ":,:,:";
        std::string jsonRestartParams = "{}";
    };
} // namespace picongpu::openPMD
