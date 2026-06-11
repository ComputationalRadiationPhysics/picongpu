# SPDX-FileCopyrightText: Rene Widera
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
#

{
    for (i=1; i<=NF; i++)  {
        a[NR,i] = $i
    }
}
NF>p { p = NF }
END {
    for(j=1; j<=p; j++) {
        str=a[1,j]
        for(i=2; i<=NR; i++){
            str=str" "a[i,j];
        }
        print str
    }
}
