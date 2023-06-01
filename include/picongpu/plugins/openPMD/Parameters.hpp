#pragma once

#include <string>
#include <tuple>

namespace picongpu::openPMD
{
    struct PluginOptions
    {
        std::string fileName = "simData";
        std::string fileInfix = "_%06T";
        std::string fileExtension = "bp";
        std::string dataPreparationStrategy = "doubleBuffer";
        std::string jsonConfig = "{}";
        std::string range = ":,:,:";
        std::string jsonRestartParams = "{}";

        // for using this with std::tie()
        operator std::
            tuple<std::string&, std::string&, std::string&, std::string&, std::string&, std::string&, std::string&>()
        {
            return std::tuple<
                std::string&,
                std::string&,
                std::string&,
                std::string&,
                std::string&,
                std::string&,
                std::string&>{
                fileName,
                fileInfix,
                fileExtension,
                dataPreparationStrategy,
                jsonConfig,
                range,
                jsonRestartParams};
        }
    };
} // namespace picongpu::openPMD