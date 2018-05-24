CUDA HLT
========

Welcome to the CUDA High Level Trigger project, an attempt to provide
a full HLT1 realization on GPU.

How to create the input
-----------------------

In the current development stage, the input is created by running Brunel. 
On one hand, the raw bank / hit information is written to binary files; 
on the other hand, the MC truth information is written to binary files to be 
able to run the PrChecker. Use the branch 
dovombru_extract_output_for_CUDA_HLT1Project (branched from Brunel v53r1)
of the Rec repository to create the input by following these steps on lxplus:

Compilation:

    fresh shell
    source /cvmfs/lhcb.cern.ch/group_login.sh
    lb-dev Brunel/v53r1
    cd BrunelDev_v53r1
    git lb-use Rec
    git lb-checkout Rec/dovombru_extract_output_for_CUDA_HLT1Project Pr/PrPixel
    git lb-checkout Rec/dovombru_extract_output_for_CUDA_HLT1Project Pr/PrEventDumper
    make
    
Running:
    
    mkdir velopix_raw
    mkdir velopix_MC
    ./run gaudirun.py options.py upgrade-bsphipi-magdown.py
    
The options and data input files can be found in the Brunel_config directory
of this repository (CUDA HLT). The output is stored in the velopix_raw and velopix_MC 
directories and needs to be copied to these folders in the CUDA_HLT1 project
to be used as input there.
    

How to run it
-------------

The project requires a graphics card with CUDA support.
The build process doesn't differ from standard cmake projects:

    mkdir build
    cd build
    cmake ..
    make

Some binary input files are included with the project for testing.
A run of the program with no arguments will let you know the basic options:

    Usage: ./cu_hlt
     -f {folder containing .bin files}
     [-n {number of files to process}=0 (all)]
     [-t {number of threads / streams}=3]
     [-r {number of repetitions per thread / stream}=10]
     [-a {transmit host to device}=1]
     [-b {transmit device to host}=1]
     [-c {consolidate tracks}=0]
     [-k {simplified kalman filter}=0]
     [-v {verbosity}=3 (info)]
     [-p (print rates)]


Here are some example run options:

    # Run all input files once
    ./cu_hlt -f ../input

    # Run a total of 1000 events, round robin over the existing ones
    ./cu_hlt -f ../input -n 1000

    # Run four streams, each with 4000 events, 20 repetitions
    ./cu_hlt -f ../input -t 4 -n 4000 -r 20

    # Run twelve streams, each with 3500 events, 40 repetitions
    ./cu_hlt -f ../input -n 3500 -t 12 -r 40
