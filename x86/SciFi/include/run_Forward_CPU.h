#pragma once

#include "Common.h"

#include "VeloDefinitions.cuh"
#include "TrackChecker.h"
#include "PrForward.h"
#include "Tools.h"
#include "VeloEventModel.cuh"

int run_forward_on_CPU (
  std::vector< trackChecker::Tracks >& ft_tracks_events,
  SciFi::HitsSoA * hits_layers_events,
  uint* host_velo_tracks_atomics,
  uint* host_velo_track_hit_number,
  uint* host_velo_states,
  VeloUTTracking::TrackUT * veloUT_tracks,
  const int * n_veloUT_tracks_events,
  const uint &number_of_events
);

