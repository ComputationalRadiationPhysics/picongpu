#pragma once

#include <string>
#include <tuple>

namespace picongpu::openPMD
{
    /*
     * No default values here since those are registered in the
     * plugins::multi::Option data members of openPMDWriter::Help.
     * The openPMD plugin will automatically use those default values here.
     * Ref.: openPMDWriter::Help::pluginParameters() (when using cmd line parameters)
     *       openPMDWriter::openPMDWriter()          (when using TOML configuration)
     */
    struct PluginParameters
    {
        std::string fileName; /* Name of the openPMDSeries, excluding the extension */
        std::string fileInfix;
        std::string fileExtension; /* Extension of the file name */
        std::string particleIOChunkSizeString;
        std::string dataPreparationStrategyString;
        std::string backendConfigString;
        std::string backendConfigStringForbidden;
        std::string rangeString;
        std::string backendConfigRestartString;
        std::string backendConfigRestartStringForbidden;
        std::string writeAccessString;
    };
} // namespace picongpu::openPMD
