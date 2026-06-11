# SPDX-FileCopyrightText: Rene Widera
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
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
