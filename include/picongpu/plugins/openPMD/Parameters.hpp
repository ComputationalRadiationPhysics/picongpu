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

        // for using this with std::tie()
        operator std::tuple<std::string&, std::string&, std::string&, std::string&, std::string&, std::string&>()
        {
            return std::tuple<std::string&, std::string&, std::string&, std::string&, std::string&, std::string&>{
                fileName,
                fileInfix,
                fileExtension,
                dataPreparationStrategy,
                jsonConfig,
                range};
        }
    };
} // namespace picongpu::openPMD