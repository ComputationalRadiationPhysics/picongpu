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
        /** Open the output Series lazily on the first run.
         *
         * The plugin always parses and validates its configuration at
         * simulation startup (time step zero). If this flag is set, opening
         * the output Series, and with it the validation of the backend
         * configuration against the openPMD API, is deferred to the first
         * actual run. This can avoid hangups in certain contexts, e.g. when
         * the underlying file system is not ready yet. Set this flag to false
         * to open the output Series already at simulation startup for early
         * backend validation.
         */
        bool lateInit{true};
    };
} // namespace picongpu::openPMD
