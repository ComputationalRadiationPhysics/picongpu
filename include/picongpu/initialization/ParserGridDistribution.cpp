/* Copyright 2013-2021 Axel Huebl, Rene Widera, Benjamin Worpitz
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "picongpu/initialization/ParserGridDistribution.hpp"

#include <pmacc/verify.hpp>
#include <cstdint>
#include <vector> // std::vector
#include <string> // std::string
#include <iterator> // std::distance

#include <regex>


namespace picongpu
{
    ParserGridDistribution::ParserGridDistribution(std::string const s)
    {
        parsedInput = parse(s);
    }

    uint32_t ParserGridDistribution::getOffset(uint32_t const devicePos, uint32_t const maxCells) const
    {
        value_type::const_iterator iter = parsedInput.begin();
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
        value_type::const_iterator iter = parsedInput.begin();
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

            const SubdomainPair g
                = {static_cast<uint32_t>(std::stoul(extent)), static_cast<uint32_t>(std::stoul(count))};
            newInput.emplace_back(g);
        }

        return newInput;
    }

} // namespace picongpu
