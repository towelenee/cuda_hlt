#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const char* host_events_pinned,
  const unsigned int* host_event_offsets_pinned,
  const unsigned int* host_hit_offsets_pinned,
  size_t host_events_pinned_size,
  size_t host_event_offsets_pinned_size,
  size_t host_hit_offsets_pinned_size,
  unsigned int start_event,
  unsigned int number_of_events,
  unsigned int number_of_repetitions
) {
  for (unsigned int repetitions=0; repetitions<number_of_repetitions; ++repetitions) {
    /////////////////////////
    // CalculatePhiAndSort //
    /////////////////////////

    // Optional transmission host to device
    if (transmit_host_to_device) {
      cudaCheck(cudaMemcpyAsync(dev_events, host_events_pinned, host_events_pinned_size, cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(dev_event_offsets, host_event_offsets_pinned, host_event_offsets_pinned_size * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(dev_hit_offsets, host_hit_offsets_pinned, host_hit_offsets_pinned_size * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    }

    // Invoke kernel
    calculatePhiAndSort();

    /////////////////////
    // SearchByTriplet //
    /////////////////////

    searchByTriplet();

    ////////////////////////
    // Consolidate tracks //
    ////////////////////////
    
    consolidateTracks();

    // Optional transmission device to host
    if (transmit_device_to_host) {
      cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, dev_atomics_storage, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaEventRecord(cuda_generic_event, stream);
      cudaEventSynchronize(cuda_generic_event);
      
      int total_number_of_tracks = 0;
      for (int i=0; i<number_of_events; ++i) {
        total_number_of_tracks += host_number_of_tracks_pinned[i];
      }

      cudaCheck(cudaMemcpyAsync(host_tracks_pinned, dev_tracklets, total_number_of_tracks * sizeof(Track), cudaMemcpyDeviceToHost, stream));
    }
  }

  return cudaSuccess;
}
