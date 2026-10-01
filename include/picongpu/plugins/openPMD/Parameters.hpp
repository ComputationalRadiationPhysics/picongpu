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
        std::string rangeString;
        std::string backendConfigRestartString;
        std::string writeAccessString;
        /** Delay plugin initialization until the first run.
         *
         * By default the plugin parses and validates its configuration and
         * opens the output Series directly at simulation startup (time step
         * zero). This gives early failure for invalid configuration but may
         * lead to hangups in certain contexts, e.g. when the underlying
         * file system is not ready yet. Set this flag to opt out and defer
         * initialization to the first actual run.
         */
        bool lateInit;
    };
} // namespace picongpu::openPMD
