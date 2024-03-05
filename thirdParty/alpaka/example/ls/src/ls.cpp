// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: ISC

#include "alpaka/test/acc/TestAccs.hpp"

#include <alpaka/alpaka.hpp>

#include <string_view>

struct PerAcc
{
    bool verbose;

    template<typename TAcc>
    void operator()() const
    {
        auto const platform = alpaka::Platform<TAcc>{};
        std::cout << alpaka::getAccName<TAcc>() << '\n';
        for(auto const& dev : alpaka::getDevs(platform))
        {
            std::cout << '\t' << alpaka::getName(dev) << '\n';
            if(verbose)
            {
                auto const props = alpaka::getAccDevProps<TAcc>(dev);
                auto const globalMem = alpaka::getMemBytes(dev);

                std::cout << "\t\tMultiProcessorCount  " << props.m_multiProcessorCount << '\n';
                std::cout << "\t\tGlobalMemSizeBytes   " << globalMem << '\n';
                std::cout << "\t\tSharedMemSizeBytes   " << props.m_sharedMemSizeBytes << '\n';

                std::cout << "\t\tGridBlockExtentMax   " << props.m_gridBlockExtentMax << '\n';
                std::cout << "\t\tBlockThreadExtentMax " << props.m_blockThreadExtentMax << '\n';
                std::cout << "\t\tThreadElemExtentMax  " << props.m_threadElemExtentMax << '\n';

                std::cout << "\t\tGridBlockCountMax    " << props.m_gridBlockCountMax << '\n';
                std::cout << "\t\tBlockThreadCountMax  " << props.m_blockThreadCountMax << '\n';
                std::cout << "\t\tThreadElemCountMax   " << props.m_threadElemCountMax << '\n';
            }
        }
    }
};

int main(int argc, char const* argv[])
{
    auto const verbose = argc >= 2 && std::string_view{argv[1]} == "-v";

    using Idx = int;
    using Dim = alpaka::DimInt<1>;
    alpaka::meta::forEachType<alpaka::test::EnabledAccs<Dim, Idx>>(PerAcc{verbose});
}
