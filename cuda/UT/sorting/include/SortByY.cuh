#pragma once

#include "VeloUTDefinitions.cuh"
#include "UTDefinitions.cuh"
#include "Sorting.cuh"

__global__ void sort_by_y(
  uint32_t* dev_ut_hits,
  uint32_t* dev_ut_hit_offsets,
  uint* dev_hit_permutations,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  const float* dev_unique_sector_xs
 );
