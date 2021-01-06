/* Copyright 2015-2021 Axel Huebl
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

#include "picongpu/plugins/common/stringHelpers.hpp"


namespace picongpu
{
    namespace helper
    {
        /** Return the current date as string
         *
         * \param format, \see http://www.cplusplus.com/reference/ctime/strftime/
         * \return std::string with formatted date
         */
        std::string getDateString(std::string format)
        {
            time_t rawtime;
            struct tm* timeinfo;
            const size_t maxLen = 30;
            char buffer[maxLen];

            time(&rawtime);
            timeinfo = localtime(&rawtime);

            strftime(buffer, maxLen, format.c_str(), timeinfo);

            std::stringstream dateString;
            dateString << buffer;

            return dateString.str();
        }

        GetSplashArrayOfString::Result GetSplashArrayOfString::operator()(
            std::list<std::string> listOfStrings,
            char padding)
        {
            Result result;

            // find length of longest string in list
            CompStrBySize compStrBySize;
            std::string longestString = *std::max_element(listOfStrings.begin(), listOfStrings.end(), compStrBySize);
            result.maxLen = longestString.size();

            // allocate & prepare buffer with padding
            //   size per buffer must include terminator \0 !
            const size_t bytesPerEntry = result.maxLen + 1;
            const size_t lenAllBuffers = listOfStrings.size() * bytesPerEntry;
            result.buffers.assign(lenAllBuffers, padding);

            // copy buffers
            std::list<std::string>::iterator listIt = listOfStrings.begin();
            for(size_t i = 0; i < listOfStrings.size(); ++i, ++listIt)
            {
                // index points to each part of the buffer individually
                const size_t startIdx = i * bytesPerEntry;
                std::vector<char>::iterator startIt = result.buffers.begin() + startIdx;

                // copy byte-wise onto padding
                std::copy(listIt->begin(), listIt->end(), startIt);
                if(padding != '\0')
                    result.buffers.at(startIdx + result.maxLen) = '\0';
            }

            // return
            return result;
        }

        GetADIOSArrayOfString::Result GetADIOSArrayOfString::operator()(std::list<std::string> listOfStrings)
        {
            Result result;

            // sum of all strings + their null terminators
            StrSize strSize;
            const size_t sumLen = std::accumulate(listOfStrings.begin(), listOfStrings.end(), 0u, strSize);

            // allocate & prepare buffer, starts
            result.buffers.assign(sumLen, '\0');
            result.starts.assign(listOfStrings.size(), nullptr);

            // concat all strings, \0 terminated
            size_t startIdx = 0;
            std::list<std::string>::iterator listIt = listOfStrings.begin();
            for(size_t i = 0; i < listOfStrings.size(); ++i, ++listIt)
            {
                std::vector<char>::iterator startIt = result.buffers.begin() + startIdx;

                // copy byte-wise onto padding
                std::copy(listIt->begin(), listIt->end(), startIt);

                // start pointer
                result.starts.at(i) = &(*startIt);

                startIdx += listIt->size() + 1;
            }

            // return
            return result;
        }
    } // namespace helper
} // namespace picongpu
