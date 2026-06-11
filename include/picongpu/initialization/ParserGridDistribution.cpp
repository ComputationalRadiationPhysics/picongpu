/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "picongpu/initialization/ParserGridDistribution.hpp"

#include <pmacc/verify.hpp>

#include <cstdint>
#include <iterator> // std::distance
#include <regex>
#include <string> // std::string
#include <vector> // std::vector

namespace picongpu
{
    ParserGridDistribution::ParserGridDistribution(std::string const s)
    {
        parsedInput = parse(s);
    }

    uint32_t ParserGridDistribution::getOffset(uint32_t const devicePos, uint32_t const maxCells) const
    {
        auto iter = parsedInput.begin();
        // go to last device of these n subdomains extent{n}
        uint32_t i = iter->count - 1u;
        uint32_t sum = 0u;

        while(i < devicePos)
        {
            // add last subdomain
            sum += iter->extent * iter->count;

            ++iter;
            // go to last device of these n subdomains extent{n}
            i += iter->count;
        }

        // add part of this subdomain that is before me
        sum += iter->extent * (devicePos + iter->count - i - 1u);

        // check total number of cells
        uint32_t sumTotal = 0u;
        for(iter = parsedInput.begin(); iter != parsedInput.end(); ++iter)
            sumTotal += iter->extent * iter->count;

        PMACC_VERIFY(sumTotal == maxCells);

        return sum;
    }

    uint32_t ParserGridDistribution::getLocalSize(uint32_t const devicePos) const
    {
        auto iter = parsedInput.begin();
        // go to last device of these n subdomains extent{n}
        uint32_t i = iter->count - 1u;

        while(i < devicePos)
        {
            ++iter;
            // go to last device of these n subdomains extent{n}
            i += iter->count;
        }

        return iter->extent;
    }

    void ParserGridDistribution::verifyDevices(uint32_t const numDevices) const
    {
        uint32_t numSubdomains = 0u;
        for(SubdomainPair const& p : parsedInput)
            numSubdomains += p.count;

        PMACC_VERIFY(numSubdomains == numDevices);
    }

    ParserGridDistribution::value_type ParserGridDistribution::parse(std::string const s) const
    {
        std::regex regFind(R"([0-9]+(\{[0-9]+})*)", std::regex::egrep);

        std::sregex_token_iterator iter(s.begin(), s.end(), regFind, 0);
        std::sregex_token_iterator end;

        value_type newInput;
        newInput.reserve(std::distance(iter, end));

        for(; iter != end; ++iter)
        {
            std::string pM = *iter;

            // find count n and extent b of b{n}
            std::regex regCount(R"((.*\{)|(}))", std::regex::egrep);
            std::string count = std::regex_replace(pM, regCount, "");

            std::regex regExtent(R"(\{.*})", std::regex::egrep);
            std::string extent = std::regex_replace(pM, regExtent, "");

            // no count {n} given (implies one)
            if(count == *iter)
                count = "1";

            SubdomainPair const g
                = {static_cast<uint32_t>(std::stoul(extent)), static_cast<uint32_t>(std::stoul(count))};
            newInput.emplace_back(g);
        }

        return newInput;
    }

} // namespace picongpu
