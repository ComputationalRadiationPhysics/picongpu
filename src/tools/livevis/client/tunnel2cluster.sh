#!/usr/bin/env bash
#

# set in /etc/hosts for the n_head
# 127.0.0.1       hypnos
#
# set DEFAULT_SERVER_IP to "127.0.0.1" in main.cpp

# config
n_login=huebl@uts.fz-rossendorf.de
n_head=149.220.4.37

# functions
function getNewSSH {
  tmpSSH2=$( pidof ssh | sed 's/ /\n/g' | sort )
  echo "$tmpSSH1" > tmpFile1
  echo "$tmpSSH2" > tmpFile2
  grep -Fxv -f tmpFile1 tmpFile2 | tee -a .tmpSSHtunnels
  rm -rf tmpFile1 tmpFile2
}
function finish {
  tmpPIDs=$(cat .tmpSSHtunnels)
  kill $tmpPIDs
  rm -rf .tmpSSHtunnels
  echo "done"
}
trap finish EXIT

# start udp tunnel
tmpSSH1=$( pidof ssh | sed 's/ /\n/g' | sort )
ssh -f -L 8203:$n_head:8203 $n_login -N
getNewSSH

ssh $n_login "ssh $n_head 'rm -rf testfifo && mkfifo testfifo'"

tmpSSH1=$( pidof ssh | sed 's/ /\n/g' | sort )
ssh $n_login "ssh $n_head 'netcat -v -l -p 8203 < testfifo | nc -u 127.0.0.1 8200 > testfifo'" &
getNewSSH

rm -rf testfifo && mkfifo testfifo
netcat -v -l -u -p 8200 < testfifo | nc localhost 8203 > testfifo &

# start tcp (rivlib data) tunnel
ssh -L 52000:$n_head:52000 $n_login -N
