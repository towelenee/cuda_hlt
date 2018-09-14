#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.h"
#include "VeloUTDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"
#include "LinearFitting.h"
#include "HoughTransform.h"

/**
   Functions related to selecting hits on the x planes,
   which match to the VeloUT input track
 */

void collectAllXHits(
  SciFi::HitsSoA* hits_layers,
  std::vector<int>& allXHits, 
  const float xParams_seed[4],
  const float yParams_seed[4],
  const VeloState& velo_state,
  const float qop,
  int side);

void selectXCandidates(
  SciFi::HitsSoA* hits_layers,
  std::vector<int>& allXHits,
  const VeloUTTracking::TrackUT& veloUTTrack,
  std::vector<SciFi::Track>& outputTracks,
  const float zRef_track,
  const float xParams_seed[4],
  const float yParams_seed[4],
  const VeloState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  int side);

bool addHitsOnEmptyXLayers(
  SciFi::HitsSoA* hits_layers,
  std::vector<float> &trackParameters,
  const float xParams_seed[4],
  const float yParams_seed[4],
  bool fullFit,
  std::vector<unsigned int> &coordToFit,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  int side);