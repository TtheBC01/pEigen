# Hyperdyne
Consensus algorithms, API, P2P, etc
the p2p.d directory contains classes that encapsulate code
necessary for establishing client side and server side connections. 

the topology.d directory contains classes that wrap the p2p tools into 
relavent data structures for simplifying local connection toplogies

the protocols.d directory contains classes that use predefined topology classes
to implement consensus algorithms

pythonwrapper.d directory wraps code for interfacing with python

Building requimes cmake version 2.6 or higher. Additionally, the following boost
libraries are required: 
system
thread
timer
serialization
python

Also, python development libraries are required to build

to build run "cmake ." in this directory then type "make" in the command line
