#
# Copyright 2013-2021 Rene Widera
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

#example: awk -v minValue=150 -v maxValue=300 -f SumEnergyRange.awk DATAFILE

BEGIN {
    columnBegin=3;
    columnEnd=0;
}
NR==1 {
      for(i=columnBegin;i <= NF;++i)
      {
        if($i>=minValue)
        {
            columnBegin=i;
            break;
        }
      }
      for(i=columnBegin;i <= NF;++i)
        if($i>maxValue)
        {
            columnEnd=i-1;
            break;
        }
        else if($i==maxValue)
        {
            columnEnd=i;
            break;
        }
}
NR>1 && NF>0 {
    value=0;
    for(i =columnBegin;i<=columnEnd;++i)
    {
        value+=$i
    }
    print($1" "value);
}
