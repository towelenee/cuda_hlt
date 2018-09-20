#include "HoughTransform.h"

//calculate xref for this plane
//in the c++ this is vectorized, undoing because no point before CUDA (but vectorization is obvious)
__host__ __device__ void xAtRef_SamePlaneHits(
  SciFi::HitsSoA* hits_layers,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int n_x_hits,
  float coordX[SciFi::Tracking::max_x_hits],
  const float xParams_seed[4],
  const SciFi::Tracking::Arrays& constArrays,
  MiniState velo_state, 
  int itH, int itEnd)
{
  //this is quite computationally expensive mind, should take care when porting
  float zHit    = hits_layers->m_z[allXHits[itH]]; //all hits in same layer
  float xFromVelo_Hit = straightLineExtend(xParams_seed,zHit);
  float zMagSlope = constArrays.zMagnetParams[2] * pow(velo_state.tx,2) +  constArrays.zMagnetParams[3] * pow(velo_state.ty,2);
  float dSlopeDivPart = 1.f / ( zHit - constArrays.zMagnetParams[0]);
  float dz      = 1.e-3f * ( zHit - SciFi::Tracking::zReference );
  
  while( itEnd>itH ){
    float xHit = hits_layers->m_x[allXHits[itH]];
    float dSlope  = ( xFromVelo_Hit - xHit ) * dSlopeDivPart;
    float zMag    = constArrays.zMagnetParams[0] + constArrays.zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
    float xMag    = xFromVelo_Hit + velo_state.tx * (zMag - zHit);
    float dxCoef  = dz * dz * ( constArrays.xParams[0] + dz * constArrays.xParams[1] ) * dSlope;
    float ratio   = (  SciFi::Tracking::zReference - zMag ) / ( zHit - zMag );
    coordX[itH] = xMag + ratio * (xHit + dxCoef  - xMag);
    itH++;
  }
}

__host__ __device__ int fitParabola(
  int* coordToFit,
  const int n_coordToFit,
  SciFi::HitsSoA* hits_layers,
  float trackParameters[SciFi::Tracking::nTrackParams],
  const bool xFit ) {

  //== Fit a cubic
  float s0   = 0.f; 
  float sz   = 0.f; 
  float sz2  = 0.f; 
  float sz3  = 0.f; 
  float sz4  = 0.f; 
  float sd   = 0.f; 
  float sdz  = 0.f; 
  float sdz2 = 0.f; 
  
  for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit) {
    int hit = coordToFit[i_hit];
    float d = trackToHitDistance(trackParameters, hits_layers, hit);
    if (!xFit)
      d *= - 1. / hits_layers->m_dxdy[hit];//TODO multiplication much faster than division!
    float w = hits_layers->m_w[hit];
    float z = .001f * ( hits_layers->m_z[hit] - SciFi::Tracking::zReference );
    s0   += w;
    sz   += w * z; 
    sz2  += w * z * z; 
    sz3  += w * z * z * z; 
    sz4  += w * z * z * z * z; 
    sd   += w * d; 
    sdz  += w * d * z; 
    sdz2 += w * d * z * z; 
  }    
  const float b1 = sz  * sz  - s0  * sz2; 
  const float c1 = sz2 * sz  - s0  * sz3; 
  const float d1 = sd  * sz  - s0  * sdz; 
  const float b2 = sz2 * sz2 - sz * sz3; 
  const float c2 = sz3 * sz2 - sz * sz4; 
  const float d2 = sdz * sz2 - sz * sdz2;
  const float den = (b1 * c2 - b2 * c1 );
  if(!(std::fabs(den) > 1e-5)) return false;
  const float db  = (d1 * c2 - d2 * c1 ) / den; 
  const float dc  = (d2 * b1 - d1 * b2 ) / den; 
  const float da  = ( sd - db * sz - dc * sz2) / s0;
  if (xFit) {
    trackParameters[0] += da;
    trackParameters[1] += db*1.e-3f;
    trackParameters[2] += dc*1.e-6f;
  } else {
    trackParameters[4] += da;
    trackParameters[5] += db*1.e-3f;
    trackParameters[6] += dc*1.e-6f;
  }

  return true;
}

__host__ __device__ bool fitXProjection(
  SciFi::HitsSoA *hits_layers,
  float trackParameters[SciFi::Tracking::nTrackParams],
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars)
{

  if (planeCounter.nbDifferent < pars.minXHits) return false;
  bool doFit = true;
  while ( doFit ) {

    fitParabola( coordToFit, n_coordToFit, hits_layers, trackParameters, true );
    
    float maxChi2 = 0.f; 
    float totChi2 = 0.f;  
    //int   nDoF = -3; // fitted 3 parameters
    int  nDoF = -3;
    const bool notMultiple = planeCounter.nbDifferent == n_coordToFit;

    int worst = n_coordToFit;
    for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit ) {
      int hit = coordToFit[i_hit];
      float d = trackToHitDistance(trackParameters, hits_layers, hit);
      float chi2 = d*d*hits_layers->m_w[hit];
      totChi2 += chi2;
      ++nDoF;
      if ( chi2 > maxChi2 && ( notMultiple || planeCounter.nbInPlane( hits_layers->m_planeCode[hit]/2 ) > 1 ) ) {
        maxChi2 = chi2;
        worst   = i_hit; 
      }    
    }    
    if ( nDoF < 1 )return false;
    trackParameters[7] = totChi2;
    trackParameters[8] = (float) nDoF;

    if ( worst == n_coordToFit ) {
      return true;
    }    
    doFit = false;
    if ( totChi2/nDoF > SciFi::Tracking::maxChi2PerDoF  ||
         maxChi2 > SciFi::Tracking::maxChi2XProjection ) {
      removeOutlier( hits_layers, planeCounter, coordToFit, n_coordToFit, coordToFit[worst]);
      if (planeCounter.nbDifferent < pars.minXHits + pars.minStereoHits) return false;
      doFit = true;
    }    
  }
  return true;
}
 

__host__ __device__ bool fitYProjection(
  SciFi::HitsSoA* hits_layers,
  SciFi::Tracking::Track& track,
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  PlaneCounter& planeCounter,
  MiniState velo_state,
  const SciFi::Tracking::Arrays& constArrays,
  SciFi::Tracking::HitSearchCuts& pars)
{
  
  float maxChi2 = 1.e9f;
  bool parabola = false; //first linear than parabola
  //== Fit a line
  const float txs  = track.trackParams[0]; // simplify overgeneral c++ calculation
  const float tsxz = velo_state.x + (SciFi::Tracking::zReference - velo_state.z) * velo_state.tx; 
  const float tolYMag = SciFi::Tracking::tolYMag + SciFi::Tracking::tolYMagSlope * fabs(txs-tsxz);
  const float wMag   = 1./(tolYMag * tolYMag );

  bool doFit = true;
  while ( doFit ) {
    //Use position in magnet as constrain in fit
    //although because wMag is quite small only little influence...
    float zMag  = zMagnet(velo_state, constArrays);
    const float tys = track.trackParams[4]+(zMag-SciFi::Tracking::zReference)*track.trackParams[5];
    const float tsyz = velo_state.y + (zMag-velo_state.z)*velo_state.ty;
    const float dyMag = tys-tsyz;
    zMag -= SciFi::Tracking::zReference;
    float s0   = wMag;
    float sz   = wMag * zMag;
    float sz2  = wMag * zMag * zMag;
    float sd   = wMag * dyMag;
    float sdz  = wMag * dyMag * zMag;
   
    if ( parabola ) {

      // position in magnet not used for parabola fit, hardly any influence on efficiency
      fitParabola( stereoHits, n_stereoHits, hits_layers, track.trackParams, false );
      
    } else { // straight line fit

      for ( int i_hit = 0; i_hit < n_stereoHits; ++i_hit ) {
        int hit = stereoHits[i_hit];
        const float d = - trackToHitDistance(track.trackParams, hits_layers, hit) / 
                          hits_layers->m_dxdy[hit];//TODO multiplication much faster than division!
        const float w = hits_layers->m_w[hit];
        const float z = hits_layers->m_z[hit] - SciFi::Tracking::zReference;
	s0   += w;
        sz   += w * z; 
        sz2  += w * z * z;
        sd   += w * d;
        sdz  += w * d * z;
      }
      const float den = (s0 * sz2 - sz * sz );
      if(!(std::fabs(den) > 1e-5)) { 
        return false;
      }
      const float da  = (sd * sz2 - sdz * sz ) / den;
      const float db  = (sdz * s0 - sd  * sz ) / den;
      track.trackParams[4] += da;
      track.trackParams[5] += db;
    }//fit end, now doing outlier removal

    int worst = n_stereoHits;
    maxChi2 = 0.;
    for ( int i_hit = 0; i_hit < n_stereoHits; ++i_hit ) {
      int hit = stereoHits[i_hit];
      float d = trackToHitDistance(track.trackParams, hits_layers, hit);
      float chi2 = d*d*hits_layers->m_w[hit];
      if ( chi2 > maxChi2 ) {
        maxChi2 = chi2;
        worst   = i_hit;
      }
    }

    if ( maxChi2 < SciFi::Tracking::maxChi2StereoLinear && !parabola ) {
      parabola = true;
      maxChi2 = 1.e9f;
      continue;
    }

    if ( maxChi2 > SciFi::Tracking::maxChi2Stereo ) {
      removeOutlier( hits_layers, planeCounter, stereoHits, n_stereoHits, stereoHits[worst] );
      if ( planeCounter.nbDifferent < pars.minStereoHits ) {
        return false;
      }
      continue;
    }
    break;
  }
  return true;
} 