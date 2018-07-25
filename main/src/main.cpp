/**
 *      CUDA HLT1
 *      
 *      author  -  GPU working group
 *      e-mail  -  lhcb-parallelization@cern.ch
 *
 *      Original development June, 2014
 *      Restarted development on February, 2018
 *      CERN
 */

#include <iostream>
#include <string>
#include <cstring>
#include <exception>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <unistd.h>
#include "tbb/tbb.h"
#include "cuda_runtime.h"
#include "CudaCommon.h"
#include "Logger.h"
#include "Tools.h"
#include "InputTools.h"
#include "Timer.h"
#include "../../stream/sequence/include/StreamWrapper.cuh"
#include "../../stream/sequence/include/InitializeConstants.cuh"

void printUsage(char* argv[]){
  std::cerr << "Usage: "
    << argv[0]
    << std::endl << " -f {folder containing .bin files with raw bank information}"
    << std::endl << (mc_check_enabled ? " " : " [") << "-g {folder containing .root files with MC truth information}"
    << (mc_check_enabled ? "" : " ]")
    << std::endl << " -e {folder containing, bin files with UT hit information}"
    << std::endl << " -s {folder containing, bin files with FT hit information}"
    << std::endl << " [-n {number of files to process}=0 (all)]"
    << std::endl << " [-t {number of threads / streams}=1]"
    << std::endl << " [-r {number of repetitions per thread / stream}=1]"
    << std::endl << " [-a {transmit host to device}=1]"
    << std::endl << " [-b {transmit device to host}=1]"
    << std::endl << " [-c {run checkers}=0]"
    << std::endl << " [-k {simplified kalman filter}=0]"
    << std::endl << " [-m {reserve Megabytes}=1024]"
    << std::endl << " [-v {verbosity}=3 (info)]"
    << std::endl << " [-p (print memory usage)]"
    << std::endl;
}

int main(int argc, char *argv[])
{
  std::string folder_name_velopix_raw;
  std::string folder_name_MC = "";
  std::string folder_name_ut_hits = "";
  std::string folder_name_ft_hits = "";
  uint number_of_files = 0;
  uint tbb_threads = 1;
  uint number_of_repetitions = 1;
  uint verbosity = 3;
  bool print_memory_usage = false;
  bool transmit_host_to_device = true;
  bool transmit_device_to_host = true;
  // By default, do_check will be true when mc_check is enabled 
  bool do_check = mc_check_enabled;
  bool do_simplified_kalman_filter = false;
  size_t reserve_mb = 1024;
   
  signed char c;
  while ((c = getopt(argc, argv, "f:g:e:n:t:r:pha:b:d:v:c:k:m:")) != -1) {
    switch (c) {
    case 'f':
      folder_name_velopix_raw = std::string(optarg);
      break;
    case 'g':
      folder_name_MC = std::string(optarg);
      break;
    case 'e':
      folder_name_ut_hits = std::string(optarg);
      break;
    case 's':
      folder_name_ft_hits = std::string(optarg);
      break;
    case 'm':
      reserve_mb = atoi(optarg);
      break;
    case 'n':
      number_of_files = atoi(optarg);
      break;
    case 't':
      tbb_threads = atoi(optarg);
      break;
    case 'r':
      number_of_repetitions = atoi(optarg);
      break;
    case 'a':
      transmit_host_to_device = atoi(optarg);
      break;
    case 'b':
      transmit_device_to_host = atoi(optarg);
      break;
    case 'c':
      do_check = atoi(optarg);
      break;
    case 'k':
      do_simplified_kalman_filter = atoi(optarg);
      break;
    case 'v':
      verbosity = atoi(optarg);
      break;
    case 'p':
      print_memory_usage = true;
      break;
    case '?':
    case 'h':
    default:
      printUsage(argv);
      return -1;
    }
  }

  // Check how many files were specified and
  // call the entrypoint with the suggested format
  if(folder_name_velopix_raw.empty()){
    std::cerr << "No folder for velopix raw events specified" << std::endl;
    printUsage(argv);
    return -1;
  }
  if(folder_name_ut_hits.empty()){
    std::cerr << "No folder for ut hits specified" << std::endl;
    printUsage(argv);
    return -1;
  }

  if(folder_name_ft_hits.empty()){
    std::cerr << "No folder for ft hits specified" << std::endl;
    printUsage(argv);
    return -1;
  } 
 
  if(folder_name_MC.empty() && do_check){
    std::cerr << "No MC folder specified, but MC CHECK turned on" << std::endl;
    printUsage(argv);
    return -1;
  }

  // Set verbosity level
  std::cout << std::fixed << std::setprecision(2);
  logger::ll.verbosityLevel = verbosity;

  // Get device properties
  cudaDeviceProp device_properties;
  cudaCheck(cudaGetDeviceProperties(&device_properties, 0));
  
  // Show call options
  std::cout << "Requested options:" << std::endl
    << " folder with velopix raw bank input (-f): " << folder_name_velopix_raw << std::endl
    << " folder with MC truth input (-g): " << folder_name_MC << std::endl
    << " folder with ut hits input (-e): " << folder_name_ut_hits << std::endl
    << " folder with ft hits input (-e): " << folder_name_ft_hits << std::endl
    << " number of files (-n): " << number_of_files << std::endl
    << " tbb threads (-t): " << tbb_threads << std::endl
    << " number of repetitions (-r): " << number_of_repetitions << std::endl
    << " transmit host to device (-a): " << transmit_host_to_device << std::endl
    << " transmit device to host (-b): " << transmit_device_to_host << std::endl
    << " run checkers (-c): " << do_check << std::endl
    << " simplified kalman filter (-k): " << do_simplified_kalman_filter << std::endl
    << " reserve MB (-m): " << reserve_mb << std::endl
    << " print memory usage (-p): " << print_memory_usage << std::endl
    << " verbosity (-v): " << verbosity << std::endl
    << " device: " << device_properties.name << std::endl
    << std::endl;

  std::cout << "MC check (compile opt): " << (mc_check_enabled ? "On" : "Off") << std::endl
    << " folder with MC truth input (-g): " << folder_name_MC << std::endl
    << " run checkers (-c): " << do_check << std::endl
    << std::endl;

  // Read folder contents
  std::vector<char> velopix_events;
  std::vector<unsigned int> velopix_event_offsets;
  verbose_cout << "Reading velopix raw events" << std::endl;
  read_folder(
    folder_name_velopix_raw,
    number_of_files,
    velopix_events,
    velopix_event_offsets );
  
  check_velopix_events( velopix_events, velopix_event_offsets, number_of_files );

  if ( !fileExists(folder_name_velopix_raw + "/geometry.bin") ) {
    error_cout << "no geometry file found in " << folder_name_velopix_raw << std::endl;
    return -1;
  }
  std::vector<char> geometry;
  readGeometry(folder_name_velopix_raw, geometry);

  // Copy data to pinned host memory
  const int number_of_events = velopix_event_offsets.size() - 1;
  char* host_velopix_events_pinned;
  uint* host_velopix_event_offsets_pinned;
  cudaCheck(cudaMallocHost((void**)&host_velopix_events_pinned, velopix_events.size()));
  cudaCheck(cudaMallocHost((void**)&host_velopix_event_offsets_pinned, velopix_event_offsets.size() * sizeof(uint)));
  std::copy_n(std::begin(velopix_events), velopix_events.size(), host_velopix_events_pinned);
  std::copy_n(std::begin(velopix_event_offsets), velopix_event_offsets.size(), host_velopix_event_offsets_pinned);

  std::vector<char> ut_events;
  std::vector<unsigned int> ut_event_offsets;
  verbose_cout << "Reading UT hits for " << number_of_events << " events " << std::endl;
  read_folder( folder_name_ut_hits, number_of_files,
  	      ut_events, ut_event_offsets );

  
  VeloUTTracking::HitsSoA *ut_hits_events = new VeloUTTracking::HitsSoA[number_of_events];
  uint32_t ut_n_hits_layers_events[number_of_events][VeloUTTracking::n_layers];
  read_ut_events_into_arrays( ut_hits_events, ut_n_hits_layers_events,
  			      ut_events, ut_event_offsets, number_of_events );

  std::vector<char> ft_events;
  std::vector<unsigned int> ft_event_offsets;
  verbose_cout << "Reading FT hits for " << number_of_events << " events " << std::endl;
  read_folder( folder_name_ft_hits, number_of_files,
              ft_events, ft_event_offsets );

  ForwardTracking::HitsSoAFwd *ft_hits_events = new ForwardTracking::HitsSoAFwd[number_of_events];
  uint32_t ft_n_hits_layers_events[number_of_events][ForwardTracking::n_layers];
  read_ft_events_into_arrays( ft_hits_events, ft_n_hits_layers_events,
                              ft_events, ft_event_offsets, number_of_events );

  //check_ut_events( ut_hits_events, ut_n_hits_layers_events, number_of_events );

  // Initialize detector constants on GPU
  initializeConstants();

  // Create streams
  StreamWrapper stream_wrapper;
  stream_wrapper.initialize_streams(
    tbb_threads,
    velopix_events,
    velopix_event_offsets,
    geometry,
    number_of_events,
    transmit_host_to_device,
    transmit_device_to_host,
    do_check,
    do_simplified_kalman_filter,
    print_memory_usage,
    folder_name_MC,
    reserve_mb
  );
  
  // Attempt to execute all in one go
  Timer t;
  tbb::parallel_for(
    static_cast<uint>(0),
    static_cast<uint>(tbb_threads),
    [&] (uint i) {
      stream_wrapper.run_stream(
        i,
        host_velopix_events_pinned,
        host_velopix_event_offsets_pinned,
        velopix_events.size(),
        velopix_event_offsets.size(),
	ut_hits_events,
	ut_n_hits_layers_events,
        ft_hits_events,
        ft_n_hits_layers_events,
        number_of_events,
        number_of_repetitions
      );
    }
  );
  t.stop();

  delete [] ut_hits_events;
  delete [] ft_hits_events; 
 
  std::cout << (number_of_events * tbb_threads * number_of_repetitions / t.get()) << " events/s" << std::endl
    << "Ran test for " << t.get() << " seconds" << std::endl;

  // Free and reset device
  // cudaCheck(cudaFreeHost(host_velopix_events_pinned));
  // cudaCheck(cudaFreeHost(host_velopix_event_offsets_pinned));
  cudaCheck(cudaDeviceReset());

  return 0;
}
