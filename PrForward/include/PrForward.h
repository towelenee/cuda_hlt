#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include "Logger.h"

#include "SystemOfUnits.h"
#include "TMVA_Forward_1.h"
#include "TMVA_Forward_2.h"
#include "SciFiDefinitions.cuh"
#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrVeloUT.cuh"
#include "PrForwardConstants.h"
#include "TrackUtils.h"
#include "LinearFitting.h"
#include "HitUtils.h"
#include "HoughTransform.h"
#include "FindXHits.h"
#include "FindStereoHits.h"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"

/** @class PrForward PrForward.h
   *
   *  - InputTracksName: Input location for VeloUT tracks
   *  - OutputTracksName: Output location for Forward tracks
   *  Based on code written by
   *  2012-03-20 : Olivier Callot
   *  2013-03-15 : Thomas Nikodem
   *  2015-02-13 : Sevda Esen [additional search in the triangles by Marian Stahl]
   *  2016-03-09 : Thomas Nikodem [complete restructuring]
   */


void PrForward(
  SciFi::HitsSoA *hits_layers_events,
  const Velo::Consolidated::States& velo_states,
  const uint event_tracks_offset,
  const VeloUTTracking::TrackUT * veloUT_tracks,
  const int n_veloUT_tracks,
  const SciFi::TMVA1& tmva1,
  const SciFi::TMVA2& tmva2,
  SciFi::Track outputTracks[SciFi::max_tracks],
  int& n_forward_tracks);
                                                      
void find_forward_tracks(
  SciFi::HitsSoA* hits_layers,  
  const VeloUTTracking::TrackUT& veloUTTrack,
  SciFi::Track outputTracks[SciFi::max_tracks],
  int& n_forward_tracks,
  const SciFi::TMVA1& tmva1,
  const SciFi::TMVA2& tmva2,
  const MiniState& velo_state);


void selectFullCandidates(
  SciFi::HitsSoA* hits_layers,
  SciFi::Track* candidate_tracks,
  int& n_candidate_tracks,
  SciFi::Track* selected_tracks,
  int& n_selected_tracks,
  const float xParams_seed[4],
  const float yParams_seed[4],
  MiniState velo_state,
  const float VeloUT_qOverP,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  const SciFi::TMVA1& tmva1,
  const SciFi::TMVA2& tmva2,
  bool secondLoop);

