PIConGPU SIGNALS
================

This document is a white paper and should be reviewed.
Furthermore, one has to implement it ;)



Overview
--------

### SIGNALS handled by PIConGPU on POSIX systems

 1) HUP:  Triggers TERM. Controlling process/terminal hangup.
 2) INT:  Triggers TERM. This SIGNAL gets triggert while hitting ^C.
 3) QUIT: Triggers TERM. This is the terminal quit SIGNAL.
 6) ABRT: Triggers TERM. Can only be called from within the code.
10) USR1: Mark next time step as temporary restart time step and create a
          permanent check point (hdf5) for it.
12) USR2: Reset temporary restart time step to initial value (0).
14) ALRM: Restart simulation from temporary check point time step or n = 0.
          Remove all saved check points (hdf5) behind this point.
          Note: Should NOT be caused to pause (sleep) the simulation!
                (since sleep can be implemented with the ALRM SIGNAL)
15) TERM: Finish current loop and exit normally by setting time step n_max = n.
20) TSTP: After the next timestep, sleep for 15 seconds.
          This is the terminal/keyboard stop SIGNAL.
23) URG:  Create (hdf5) output but do not mark as checkpoint. Triggers TERM afterwards.
29) IO:   Create (hdf5) output but do not mark as checkpoint.


### Default SIGNALS

These can not be handled:

9)  KILL
18) CONT
19) STOP



Batch Systems
-------------

### Standard Values

qdel: KILL (-9)
  http://www.vub.ac.be/BFUCC/LSF/man/qdel.1.html
qsig: TERM
  http://pubs.opengroup.org/onlinepubs/9699919799/utilities/qsig.html


### Examples

You are running into the wall time, let's save the run!
  qsig -s URG <jobid>

Create a 30 seconds loop, for example for presentations:
  qsig -s USR1 <jobid>
  while true; do qsig -s ALRM <jobid> && sleep 30; done

Freeze the simulation for 15 seconds (to point out that cool picture):
  qsig -s TSTP <jobid>

Abort the simulation in a "clean way":
  qsig -s TERM <jobid>

Abort the simulation - hard!
  qdel <jobid>

Remove job from queue (if qdel is ignored for hours):
  qsig -s 0 <jobid>


### To Do

Is it possible to configure various batch systems to send a SIGNAL m minutes
before a job runs into the wall time?
The SIGNAL URG would be perfect, since it is ignored in the default handler.



Reference to SIGNALS
--------------------

Introduction to SIGNALS:
  http://beej.us/guide/bgipc/output/html/multipage/signals.html


Developer Notes:
  Take care with async-safe behaviour, blocking of incompatible SIGNALS and
  side-effects!

  "You also cannot safely alter any shared (e.g. global) data, with one notable
   exception: variables that are declared to be of storage class and type
   volatile sig_atomic_t."


List of POSIX SIGNALS:
  http://en.wikipedia.org/wiki/Unix_signal#POSIX_signals
  http://unixhelp.ed.ac.uk/CGI/man-cgi?signal+7

Boost Asio (Boost.Asio):
  http://www.boost.org/doc/libs/1_54_0/doc/html/boost_asio/overview/signals.html
  http://www.boost.org/doc/libs/1_54_0/doc/html/boost_asio/reference/signal_set.html
  http://stackoverflow.com/questions/4639909/standard-way-to-perform-a-clean-shutdown-with-boost-asio
