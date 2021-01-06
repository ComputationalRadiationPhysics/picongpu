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

#pragma once

#include <vector> // std::vector
#include <string> // std::string
#include <cstdint>


namespace picongpu
{
    class ParserGridDistribution
    {
    private:
        /** 1D sudomain extents
         *
         * Pair of extent and count entry in our grid distribution.
         *
         * For example, a single entry of the grid distribution a,b,c{n},d{m},e,f
         * is stored as entry (a,1) in SubdomainPair. Another as (b,1), another
         * n equally spaced subdomains as (c,n), another m subdomains of extent d
         * as (d,m), and so on.
         */
        struct SubdomainPair
        {
            // extent of the current subdomain
            uint32_t extent;
            // count of how often the subdomain shall be repeated
            uint32_t count;
        };
        using value_type = std::vector<SubdomainPair>;

    public:
        ParserGridDistribution(std::string const s);

        uint32_t getOffset(uint32_t const devicePos, uint32_t const maxCells) const;

        /** Get local Size of this dimension
         *
         *  \param[in] devicePos as unsigned integer in the range [0, n-1] for this dimension
         *  \return uint32_t with local number of cells
         */
        uint32_t getLocalSize(uint32_t const devicePos) const;

        /** Verify the number of subdomains matches the devices
         *
         * Check that the number of subdomains in a dimension, after we
         * expanded all regexes, matches the number of devices for it.
         *
         * \param[in] numDevices number of devices for this dimension
         */
        void verifyDevices(uint32_t const numDevices) const;

    private:
        value_type parsedInput;

        /** Parses the input string to a vector of SubdomainPair(s)
         *
         * Parses the input string in the form a,b,c{n},d{m},e,f
         * to a vector of SubdomainPair with extent number (a,b,c,d,e,f) and
         * counts (1,1,n,m,e,f)
         *
         * \param[in] s as string in the form a,b{n}
         * \return std::vector<SubdomainPair> with 2x uint32_t (extent, count)
         */
        value_type parse(std::string const s) const;
    };

} // namespace picongpu
