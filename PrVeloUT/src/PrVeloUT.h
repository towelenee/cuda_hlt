# pragma once

#include <cmath>

// from cuda_hlt/checker/lib
#include "../checker/lib/include/Tracks.h"

// from Rec - PrKernel
#include "include/UTHitHandler.h"
#include "include/UTHitInfo.h"
#include "include/UTHit.h"

#include "PrUTMagnetTool.h"

// Math from ROOT
#include "CholeskyDecomp.h"

#include "include/VeloTypes.h"
#include "include/SystemOfUnits.h"

/** @class PrVeloUT PrVeloUT.h
   *
   *  PrVeloUT algorithm. This is just a wrapper,
   *  the actual pattern recognition is done in the 'PrVeloUTTool'.
   *
   *  - InputTracksName: Input location for Velo tracks
   *  - OutputTracksName: Output location for VeloTT tracks
   *  - TimingMeasurement: Do a timing measurement?
   *
   *  @author Mariusz Witek
   *  @date   2007-05-08
   *  @update for A-Team framework 2007-08-20 SHM
   *
   *  2017-03-01: Christoph Hasse (adapt to future framework)
   *  2018-05-05: Plácido Fernández
   */

struct TrackHelper{
  VeloState state;
  std::array<const Hit*, 4> bestHits = { nullptr, nullptr, nullptr, nullptr};
  std::array<float, 4> bestParams;
  float wb, invKinkVeloDist, xMidField;

  TrackHelper(
    const VeloState& miniState, 
    const float zKink, 
    const float sigmaVeloSlope, 
    const float maxPseudoChi2
  ):
    state(miniState),
    bestParams{{ 0.0, maxPseudoChi2, 0.0, 0.0 }}{
    xMidField = state.x + state.tx*(zKink-state.z);
    const float a = sigmaVeloSlope*(zKink - state.z);
    wb=1./(a*a);
    invKinkVeloDist = 1/(zKink-state.z);
  }
};

class PrVeloUT {

public:
  /// Standard constructor
  PrVeloUT();
  virtual int initialize() override;    ///< Algorithm initialization
  LHCb::Tracks operator()(const std::vector<Track>& inputTracks) const override;

private:

  const float m_minMomentum = 1.5*Gaudi::Units::GeV;
  const float m_minPT = 0.3*Gaudi::Units::GeV;
  const float m_maxPseudoChi2 = 1280.;
  const float m_yTol = 0.5  * Gaudi::Units::mm;
  const float m_yTolSlope = 0.08;
  const float m_hitTol1 = 6.0 * Gaudi::Units::mm;
  const float m_hitTol2 = 0.8 * Gaudi::Units::mm;
  const float m_deltaTx1 = 0.035;
  const float m_deltaTx2 = 0.018;
  const float m_maxXSlope = 0.350;
  const float m_maxYSlope = 0.300;
  const float m_centralHoleSize = 33. * Gaudi::Units::mm;
  const float m_intraLayerDist = 15.0 * Gaudi::Units::mm;
  const float m_overlapTol = 0.7 * Gaudi::Units::mm;
  const float m_passHoleSize = 40. * Gaudi::Units::mm;
  const int   m_minHighThres = 1;
  const bool  m_printVariables = false;
  const bool  m_passTracks = false;
  const bool  m_doTiming = false;

  // typedef MultiIndexedHitContainer<Hit, UT::Info::kNStations, UT::Info::kNLayers>::HitRange HitRange;

  bool getState(const Track* iTr, VeloState& trState, Track& outputTracks) const;

  bool getHits(std::array<std::vector<Hit>,4>& hitsInLayers,  const std::array<std::array<HitRange::const_iterator,85>,4>& iteratorsLayers,
               const UT::HitHandler* hh,
               const std::vector<float>& fudgeFactors, VeloState& trState ) const;

  bool formClusters(const std::array<std::vector<Hit>,4>& hitsInLayers, TrackHelper& helper) const;

  void prepareOutputTrack(const Track* veloTrack,
                          const TrackHelper& helper,
                          const std::array<std::vector<Hit>,4>& hitsInLayers,
                          std::vector<Track>& outputTracks,
                          const std::vector<float>& bdlTable) const;

  // ==============================================================================
  // -- Method that finds the hits in a given layer within a certain range
  // ==============================================================================
  inline void findHits( 
    const std::vector<Hit>& inputHits,
    const VeloState& myState, 
    const float xTolNormFact,
    const float invNormFact,
    std::vector<Hit>& outHits ) const 
  {
    const auto zInit = inputHits.at(0).zAtYEq0();
    const auto yApprox = myState.y + myState.ty * (zInit - myState.z);

    int pos = 0;
    for (auto& hit : inputHits) {
      if ( hit.isNotYCompatible(yApprox, m_yTol + m_yTolSlope * std::abs(xTolNormFact)) ) {
        ++pos;
      }
    }

    const auto xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
    const auto yyProto =       myState.y - myState.ty*myState.z;

    for (int i=pos; i<inputHits.size(); ++i) {

      const Hit& hit = inputHits[pos];

      const auto xx = hit.xAt(yApprox);
      const auto dx = xx - xOnTrackProto;

      if( dx < -xTolNormFact ) continue;
      if( dx >  xTolNormFact ) break;

      // -- Now refine the tolerance in Y
      if( hit.isNotYCompatible( yApprox, m_yTol + m_yTolSlope * std::abs(dx*invNormFact)) ) continue;

      const auto zz = hit.zAtYEq0();
      const auto yy = yyProto +  myState.ty*zz;
      const auto xx2 = hit.xAt(yy);

      // TODO avoid the copy - remove the const?
      Hit temp_hit = hit;
      temp_hit.m_second_x = xx2;
      temp_hit.m_second_z = zz;

      outHits.emplace_back(temp_hit);
    }
  }

  // ===========================================================================================
  // -- 2 helper functions for fit
  // -- Pseudo chi2 fit, templated for 3 or 4 hits
  // ===========================================================================================
  void addHit( float* mat, float* rhs, const Hit& hit) const {
    const float ui = hit.x;
    const float ci = hit.cosT();
    const float dz = 0.001*(hit.z - m_zMidUT);
    const float wi = hit->HitPtr.weight();
    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }

  void addChi2( const float xTTFit, const float xSlopeTTFit, float& chi2 , const Hit& hit) const {
    const float zd    = hit.z;
    const float xd    = xTTFit + xSlopeTTFit*(zd-m_zMidUT);
    const float du    = xd - hit.x;
    chi2 += (du*du)*hit.weight();
  }



  template <std::size_t N>
  void simpleFit(
    std::array<Hit,N>& hits, 
    TrackHelper& helper ) const 
  {
    assert( N==3||N==4 );

    // -- Scale the z-component, to not run into numerical problems
    // -- with floats
    const float zDiff = 0.001*(m_zKink-m_zMidUT);
    float mat[3] = { helper.wb, helper.wb*zDiff, helper.wb*zDiff*zDiff };
    float rhs[2] = { helper.wb* helper.xMidField, helper.wb*helper.xMidField*zDiff };

    const int nHighThres = std::count_if( hits.begin(),  hits.end(),
                                          []( const Hit& hit ){ return hit && hit.highThreshold(); });

    // -- Veto hit combinations with no high threshold hit
    // -- = likely spillover
    if( nHighThres < m_minHighThres ) return;

    std::for_each( hits.begin(), hits.end(), [&](const Hit& h) { this->addHit(mat,rhs,h); } );

    ROOT::Math::CholeskyDecomp<float, 2> decomp(mat);
    if( !decomp ) return;

    decomp.Solve(rhs);

    const float xSlopeTTFit = 0.001*rhs[1];
    const float xTTFit = rhs[0];

    // new VELO slope x
    const float xb = xTTFit+xSlopeTTFit*(m_zKink-m_zMidUT);
    const float xSlopeVeloFit = (xb-helper.state.x)*helper.invKinkVeloDist;
    const float chi2VeloSlope = (helper.state.tx - xSlopeVeloFit)*m_invSigmaVeloSlope;

    float chi2TT = chi2VeloSlope*chi2VeloSlope;

    std::for_each( hits.begin(), hits.end(), [&](const Hit& h) { this->addChi2(xTTFit,xSlopeTTFit, chi2TT, h); } );

    chi2TT /= (N + 1 - 2);

    if( chi2TT < helper.bestParams[1] ){

      // calculate q/p
      const float sinInX  = xSlopeVeloFit * std::sqrt(1.+xSlopeVeloFit*xSlopeVeloFit);
      const float sinOutX = xSlopeTTFit * std::sqrt(1.+xSlopeTTFit*xSlopeTTFit);
      const float qp = (sinInX-sinOutX);

      helper.bestParams = { qp, chi2TT, xTTFit,xSlopeTTFit };

      std::copy( hits.begin(), hits.end(), helper.bestHits.begin() );
      if( N == 3 ) { helper.bestHits[3] = nullptr ; }
    }

  }

  // ---

  // ITracksFromTrackR*   m_veloUTTool       = nullptr;             ///< The tool that does the actual pattern recognition
  // ISequencerTimerTool* m_timerTool        = nullptr;             ///< Timing tool
  // int                  m_veloUTTime       = 0;                   ///< Counter for timing tool
  PrUTMagnetTool       m_PrUTMagnetTool   = nullptr;             ///< Multipupose tool for Bdl and deflection
  float                m_zMidUT;
  float                m_distToMomentum;
  float                m_zKink;
  float                m_sigmaVeloSlope;
  float                m_invSigmaVeloSlope;


};
