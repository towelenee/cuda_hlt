#pragma once

#include "TMVA_Forward.h"

namespace SciFi {
  
namespace Tracking {

  inline void TMVA2_InitTransform_1( TMVA& tmva )
  {
      // Normalization transformation, initialisation
     tmva.fMin_1[0][0] = 10;
     tmva.fMax_1[0][0] = 12;
     tmva.fMin_1[1][0] = 10;
     tmva.fMax_1[1][0] = 12;
     tmva.fMin_1[2][0] = 10;
     tmva.fMax_1[2][0] = 12;
     tmva.fMin_1[0][1] = -0.000751986168325;
     tmva.fMax_1[0][1] = 0.000730101251975;
     tmva.fMin_1[1][1] = -0.000889532791916;
     tmva.fMax_1[1][1] = 0.000888269452844;
     tmva.fMin_1[2][1] = -0.000889532791916;
     tmva.fMax_1[2][1] = 0.000888269452844;
     tmva.fMin_1[0][2] = -0.000719623989426;
     tmva.fMax_1[0][2] = 0.000641609425656;
     tmva.fMin_1[1][2] = -0.00094475003425;
     tmva.fMax_1[1][2] = 0.000963236612733;
     tmva.fMin_1[2][2] = -0.00094475003425;
     tmva.fMax_1[2][2] = 0.000963236612733;
     tmva.fMin_1[0][3] = 0.000119972355606;
     tmva.fMax_1[0][3] = 0.171912387013;
     tmva.fMin_1[1][3] = 9.59152966971e-05;
     tmva.fMax_1[1][3] = 0.161721900105;
     tmva.fMin_1[2][3] = 9.59152966971e-05;
     tmva.fMax_1[2][3] = 0.171912387013;
     tmva.fMin_1[0][4] = -0.359134197235;
     tmva.fMax_1[0][4] = 0.203789561987;
     tmva.fMin_1[1][4] = -0.537947356701;
     tmva.fMax_1[1][4] = 0.509115695953;
     tmva.fMin_1[2][4] = -0.537947356701;
     tmva.fMax_1[2][4] = 0.509115695953;
     tmva.fMin_1[0][5] = -0.0192034840584;
     tmva.fMax_1[0][5] = 0.019477725029;
     tmva.fMin_1[1][5] = -0.0266773104668;
     tmva.fMax_1[1][5] = 0.0260969996452;
     tmva.fMin_1[2][5] = -0.0266773104668;
     tmva.fMax_1[2][5] = 0.0260969996452;
     tmva.fMin_1[0][6] = -191.514282227;
     tmva.fMax_1[0][6] = 149.822387695;
     tmva.fMin_1[1][6] = -177.097839355;
     tmva.fMax_1[1][6] = 195.880737305;
     tmva.fMin_1[2][6] = -191.514282227;
     tmva.fMax_1[2][6] = 195.880737305;
  }
 
  // initialize internal variables
  inline void TMVA2_Initialize( TMVA& tmva )
  {
     // weight matrix from layer 0 to 1
     tmva.fWeightMatrix0to1[0][0] = -0.363682357093314;
     tmva.fWeightMatrix0to1[1][0] = -0.0489472213380593;
     tmva.fWeightMatrix0to1[2][0] = 0.0260684359162034;
     tmva.fWeightMatrix0to1[3][0] = -0.0183195135886001;
     tmva.fWeightMatrix0to1[4][0] = -0.0435597940879404;
     tmva.fWeightMatrix0to1[5][0] = -0.409695697569545;
     tmva.fWeightMatrix0to1[6][0] = 0.083823608603888;
     tmva.fWeightMatrix0to1[7][0] = 2.7884563303233;
     tmva.fWeightMatrix0to1[8][0] = -0.0327817823630005;
     tmva.fWeightMatrix0to1[9][0] = -0.152166863476521;
     tmva.fWeightMatrix0to1[0][1] = -8.02569746070074;
     tmva.fWeightMatrix0to1[1][1] = -1.57566743031828;
     tmva.fWeightMatrix0to1[2][1] = -8.74113304358939;
     tmva.fWeightMatrix0to1[3][1] = -15.1660820931205;
     tmva.fWeightMatrix0to1[4][1] = -3.7244859171386;
     tmva.fWeightMatrix0to1[5][1] = -5.85502710533626;
     tmva.fWeightMatrix0to1[6][1] = -1.21441194925294;
     tmva.fWeightMatrix0to1[7][1] = 0.159766452843058;
     tmva.fWeightMatrix0to1[8][1] = -5.13496215182161;
     tmva.fWeightMatrix0to1[9][1] = -1.9665293685506;
     tmva.fWeightMatrix0to1[0][2] = -3.87259609677237;
     tmva.fWeightMatrix0to1[1][2] = -13.7463874097183;
     tmva.fWeightMatrix0to1[2][2] = -4.66960984157945e-05;
     tmva.fWeightMatrix0to1[3][2] = -0.184966358069029;
     tmva.fWeightMatrix0to1[4][2] = -2.98033381620024;
     tmva.fWeightMatrix0to1[5][2] = -0.646993912152853;
     tmva.fWeightMatrix0to1[6][2] = 27.1148466663061;
     tmva.fWeightMatrix0to1[7][2] = 0.197114374537912;
     tmva.fWeightMatrix0to1[8][2] = -3.91787190684213;
     tmva.fWeightMatrix0to1[9][2] = -0.512554051079575;
     tmva.fWeightMatrix0to1[0][3] = 4.95541028582906;
     tmva.fWeightMatrix0to1[1][3] = -0.982579271133555;
     tmva.fWeightMatrix0to1[2][3] = 2.25583004410242;
     tmva.fWeightMatrix0to1[3][3] = -2.48764727093388;
     tmva.fWeightMatrix0to1[4][3] = -9.71388568465769;
     tmva.fWeightMatrix0to1[5][3] = 4.39989922354738;
     tmva.fWeightMatrix0to1[6][3] = -1.99668262303782;
     tmva.fWeightMatrix0to1[7][3] = 9.12999515405744;
     tmva.fWeightMatrix0to1[8][3] = -7.6391652044467;
     tmva.fWeightMatrix0to1[9][3] = 1.9417568161743;
     tmva.fWeightMatrix0to1[0][4] = 0.693020864624262;
     tmva.fWeightMatrix0to1[1][4] = 1.03896782947591;
     tmva.fWeightMatrix0to1[2][4] = 0.42106865840627;
     tmva.fWeightMatrix0to1[3][4] = -1.10974045620275;
     tmva.fWeightMatrix0to1[4][4] = -0.10554329100342;
     tmva.fWeightMatrix0to1[5][4] = -19.467568592815;
     tmva.fWeightMatrix0to1[6][4] = 0.541954630475377;
     tmva.fWeightMatrix0to1[7][4] = 0.672219357358115;
     tmva.fWeightMatrix0to1[8][4] = 2.54286051035417;
     tmva.fWeightMatrix0to1[9][4] = 36.8873831511543;
     tmva.fWeightMatrix0to1[0][5] = 3.29431313466074;
     tmva.fWeightMatrix0to1[1][5] = 11.664575333949;
     tmva.fWeightMatrix0to1[2][5] = 3.43714144134435;
     tmva.fWeightMatrix0to1[3][5] = 1.85706790726995;
     tmva.fWeightMatrix0to1[4][5] = -3.07833615212947;
     tmva.fWeightMatrix0to1[5][5] = -0.886964190370458;
     tmva.fWeightMatrix0to1[6][5] = -1.40606434245474;
     tmva.fWeightMatrix0to1[7][5] = -0.0894519016668799;
     tmva.fWeightMatrix0to1[8][5] = -3.24990700112274;
     tmva.fWeightMatrix0to1[9][5] = -0.167564019764147;
     tmva.fWeightMatrix0to1[0][6] = 0.0338826751202223;
     tmva.fWeightMatrix0to1[1][6] = -1.92783872591996;
     tmva.fWeightMatrix0to1[2][6] = 1.1775088910565;
     tmva.fWeightMatrix0to1[3][6] = -0.168732040045552;
     tmva.fWeightMatrix0to1[4][6] = 14.7029039282644;
     tmva.fWeightMatrix0to1[5][6] = 1.15730572575181;
     tmva.fWeightMatrix0to1[6][6] = 0.784103977726078;
     tmva.fWeightMatrix0to1[7][6] = -0.486152580769206;
     tmva.fWeightMatrix0to1[8][6] = -8.15354484425858;
     tmva.fWeightMatrix0to1[9][6] = -1.83517996941011;
     tmva.fWeightMatrix0to1[0][7] = -1.47411332800668;
     tmva.fWeightMatrix0to1[1][7] = -1.41692905398734;
     tmva.fWeightMatrix0to1[2][7] = -0.102182533294306;
     tmva.fWeightMatrix0to1[3][7] = -2.4103624442092;
     tmva.fWeightMatrix0to1[4][7] = -7.57107586584993;
     tmva.fWeightMatrix0to1[5][7] = 6.29607332908425;
     tmva.fWeightMatrix0to1[6][7] = -1.71383034420142;
     tmva.fWeightMatrix0to1[7][7] = 6.14511047364846;
     tmva.fWeightMatrix0to1[8][7] = -3.93791385193052;
     tmva.fWeightMatrix0to1[9][7] = 1.48072208584235;
     // weight matrix from layer 1 to 2
     tmva.fWeightMatrix1to2[0][0] = 4.12677463873458;
     tmva.fWeightMatrix1to2[1][0] = 0.95694975193799;
     tmva.fWeightMatrix1to2[2][0] = 2.68236250583668;
     tmva.fWeightMatrix1to2[3][0] = -1.60535968633654;
     tmva.fWeightMatrix1to2[4][0] = -5.37872915673067;
     tmva.fWeightMatrix1to2[5][0] = -0.664854638721318;
     tmva.fWeightMatrix1to2[6][0] = 4.16620047731897;
     tmva.fWeightMatrix1to2[7][0] = 0.799035782689849;
     tmva.fWeightMatrix1to2[8][0] = -0.225088888088133;
     tmva.fWeightMatrix1to2[0][1] = 2.09374462359668;
     tmva.fWeightMatrix1to2[1][1] = -1.51413159263093;
     tmva.fWeightMatrix1to2[2][1] = -2.14016618792657;
     tmva.fWeightMatrix1to2[3][1] = -0.158046673411734;
     tmva.fWeightMatrix1to2[4][1] = 0.373618282718242;
     tmva.fWeightMatrix1to2[5][1] = 1.95941843491082;
     tmva.fWeightMatrix1to2[6][1] = -0.160149534664951;
     tmva.fWeightMatrix1to2[7][1] = 0.0942699565784549;
     tmva.fWeightMatrix1to2[8][1] = -2.28998444992079;
     tmva.fWeightMatrix1to2[0][2] = -2.03361729487953;
     tmva.fWeightMatrix1to2[1][2] = 0.246934403528477;
     tmva.fWeightMatrix1to2[2][2] = 1.23912692639827;
     tmva.fWeightMatrix1to2[3][2] = -1.24584976309099;
     tmva.fWeightMatrix1to2[4][2] = 0.232791533134803;
     tmva.fWeightMatrix1to2[5][2] = -0.0984830387133716;
     tmva.fWeightMatrix1to2[6][2] = -1.35662278212464;
     tmva.fWeightMatrix1to2[7][2] = -0.618030631899878;
     tmva.fWeightMatrix1to2[8][2] = 3.31051938706427;
     tmva.fWeightMatrix1to2[0][3] = -2.24216091836216;
     tmva.fWeightMatrix1to2[1][3] = -2.0741279159171;
     tmva.fWeightMatrix1to2[2][3] = 1.58905650835314;
     tmva.fWeightMatrix1to2[3][3] = -2.24329177699312;
     tmva.fWeightMatrix1to2[4][3] = -1.32954165411477;
     tmva.fWeightMatrix1to2[5][3] = -0.962141953588846;
     tmva.fWeightMatrix1to2[6][3] = 0.553473134087373;
     tmva.fWeightMatrix1to2[7][3] = -0.221879659733536;
     tmva.fWeightMatrix1to2[8][3] = 1.37972245088897;
     tmva.fWeightMatrix1to2[0][4] = 1.61803204646002;
     tmva.fWeightMatrix1to2[1][4] = 0.149486276956427;
     tmva.fWeightMatrix1to2[2][4] = 0.241106175418173;
     tmva.fWeightMatrix1to2[3][4] = -0.871515215778542;
     tmva.fWeightMatrix1to2[4][4] = -2.15288250173632;
     tmva.fWeightMatrix1to2[5][4] = 0.704719162938588;
     tmva.fWeightMatrix1to2[6][4] = -1.85276053353347;
     tmva.fWeightMatrix1to2[7][4] = 0.152763398436487;
     tmva.fWeightMatrix1to2[8][4] = -2.1381756643797;
     tmva.fWeightMatrix1to2[0][5] = 0.451165224481025;
     tmva.fWeightMatrix1to2[1][5] = -0.197440200947197;
     tmva.fWeightMatrix1to2[2][5] = 0.799551550055436;
     tmva.fWeightMatrix1to2[3][5] = -1.18507878989424;
     tmva.fWeightMatrix1to2[4][5] = 0.736442854852866;
     tmva.fWeightMatrix1to2[5][5] = -0.778210248681898;
     tmva.fWeightMatrix1to2[6][5] = 1.0184279963991;
     tmva.fWeightMatrix1to2[7][5] = 3.02152020947048;
     tmva.fWeightMatrix1to2[8][5] = 1.44488951951168;
     tmva.fWeightMatrix1to2[0][6] = 2.11855113889747;
     tmva.fWeightMatrix1to2[1][6] = 1.13030384892463;
     tmva.fWeightMatrix1to2[2][6] = -0.409760081875493;
     tmva.fWeightMatrix1to2[3][6] = -1.14736280086342;
     tmva.fWeightMatrix1to2[4][6] = -0.177061437738292;
     tmva.fWeightMatrix1to2[5][6] = 0.949169465138031;
     tmva.fWeightMatrix1to2[6][6] = -0.3285355519552;
     tmva.fWeightMatrix1to2[7][6] = -0.0793765092190401;
     tmva.fWeightMatrix1to2[8][6] = -2.37773041909011;
     tmva.fWeightMatrix1to2[0][7] = -0.750831940769777;
     tmva.fWeightMatrix1to2[1][7] = 0.178009942769106;
     tmva.fWeightMatrix1to2[2][7] = -1.02237486835902;
     tmva.fWeightMatrix1to2[3][7] = 0.967964323610706;
     tmva.fWeightMatrix1to2[4][7] = -4.11405845648907;
     tmva.fWeightMatrix1to2[5][7] = 0.970851461678289;
     tmva.fWeightMatrix1to2[6][7] = -0.314542140867855;
     tmva.fWeightMatrix1to2[7][7] = 0.0267235503842247;
     tmva.fWeightMatrix1to2[8][7] = 0.777583791043133;
     tmva.fWeightMatrix1to2[0][8] = -0.0766468428753576;
     tmva.fWeightMatrix1to2[1][8] = -1.29672432665597;
     tmva.fWeightMatrix1to2[2][8] = -1.07091840218592;
     tmva.fWeightMatrix1to2[3][8] = -0.557257692690983;
     tmva.fWeightMatrix1to2[4][8] = 2.56309454383188;
     tmva.fWeightMatrix1to2[5][8] = 2.59834254898381;
     tmva.fWeightMatrix1to2[6][8] = 0.451467225870388;
     tmva.fWeightMatrix1to2[7][8] = -0.271472478418115;
     tmva.fWeightMatrix1to2[8][8] = 0.322847172776879;
     tmva.fWeightMatrix1to2[0][9] = 1.50513181890796;
     tmva.fWeightMatrix1to2[1][9] = -1.74229624893957;
     tmva.fWeightMatrix1to2[2][9] = -1.45656061974725;
     tmva.fWeightMatrix1to2[3][9] = -0.963316737364389;
     tmva.fWeightMatrix1to2[4][9] = -0.501558620705004;
     tmva.fWeightMatrix1to2[5][9] = -1.08768021305589;
     tmva.fWeightMatrix1to2[6][9] = 1.10289926307001;
     tmva.fWeightMatrix1to2[7][9] = -7.54827898955444;
     tmva.fWeightMatrix1to2[8][9] = -3.97260031491365;
     tmva.fWeightMatrix1to2[0][10] = -4.25444399915558;
     tmva.fWeightMatrix1to2[1][10] = 2.7785088087099;
     tmva.fWeightMatrix1to2[2][10] = -0.855599304985592;
     tmva.fWeightMatrix1to2[3][10] = 6.31732858642786;
     tmva.fWeightMatrix1to2[4][10] = -6.38927142032414;
     tmva.fWeightMatrix1to2[5][10] = -8.68408961949098;
     tmva.fWeightMatrix1to2[6][10] = -0.966303858068243;
     tmva.fWeightMatrix1to2[7][10] = 0.956475297634228;
     tmva.fWeightMatrix1to2[8][10] = 4.9830363220156;
     // weight matrix from layer 2 to 3
     tmva.fWeightMatrix2to3[0][0] = -0.359705221916529;
     tmva.fWeightMatrix2to3[1][0] = -1.33108343881925;
     tmva.fWeightMatrix2to3[2][0] = -0.254173419631061;
     tmva.fWeightMatrix2to3[3][0] = 0.844269213532595;
     tmva.fWeightMatrix2to3[4][0] = 0.229231897608563;
     tmva.fWeightMatrix2to3[5][0] = -0.824395661056368;
     tmva.fWeightMatrix2to3[6][0] = 0.305257835074692;
     tmva.fWeightMatrix2to3[0][1] = 0.0488169282722191;
     tmva.fWeightMatrix2to3[1][1] = 0.195648713620308;
     tmva.fWeightMatrix2to3[2][1] = 0.147800158020385;
     tmva.fWeightMatrix2to3[3][1] = -0.980206859398799;
     tmva.fWeightMatrix2to3[4][1] = -0.733677868364601;
     tmva.fWeightMatrix2to3[5][1] = 0.941940627461568;
     tmva.fWeightMatrix2to3[6][1] = -0.231109764490389;
     tmva.fWeightMatrix2to3[0][2] = 0.844470069798132;
     tmva.fWeightMatrix2to3[1][2] = -0.900908680241791;
     tmva.fWeightMatrix2to3[2][2] = 0.594168030958934;
     tmva.fWeightMatrix2to3[3][2] = -1.25436099188948;
     tmva.fWeightMatrix2to3[4][2] = -0.351556746084157;
     tmva.fWeightMatrix2to3[5][2] = 0.187005315757542;
     tmva.fWeightMatrix2to3[6][2] = 0.213681768782238;
     tmva.fWeightMatrix2to3[0][3] = -2.33420223621305;
     tmva.fWeightMatrix2to3[1][3] = -0.593859756778516;
     tmva.fWeightMatrix2to3[2][3] = 0.671475416265333;
     tmva.fWeightMatrix2to3[3][3] = -0.721341813887584;
     tmva.fWeightMatrix2to3[4][3] = 0.506023812700976;
     tmva.fWeightMatrix2to3[5][3] = 1.67673691416136;
     tmva.fWeightMatrix2to3[6][3] = -2.124662366003;
     tmva.fWeightMatrix2to3[0][4] = 0.381704742406321;
     tmva.fWeightMatrix2to3[1][4] = -1.10907997416722;
     tmva.fWeightMatrix2to3[2][4] = 0.305601285490167;
     tmva.fWeightMatrix2to3[3][4] = 1.8290628215884;
     tmva.fWeightMatrix2to3[4][4] = 1.59684613442337;
     tmva.fWeightMatrix2to3[5][4] = -1.63529379893428;
     tmva.fWeightMatrix2to3[6][4] = -1.10871648278323;
     tmva.fWeightMatrix2to3[0][5] = 0.887026509690821;
     tmva.fWeightMatrix2to3[1][5] = -2.41063403571978;
     tmva.fWeightMatrix2to3[2][5] = 0.876840414959149;
     tmva.fWeightMatrix2to3[3][5] = -0.41423276945129;
     tmva.fWeightMatrix2to3[4][5] = 0.695524010798871;
     tmva.fWeightMatrix2to3[5][5] = -0.332151547886367;
     tmva.fWeightMatrix2to3[6][5] = -0.954094195036293;
     tmva.fWeightMatrix2to3[0][6] = -0.87673131357239;
     tmva.fWeightMatrix2to3[1][6] = -0.589817661983923;
     tmva.fWeightMatrix2to3[2][6] = 0.607084417156363;
     tmva.fWeightMatrix2to3[3][6] = -0.24324056006996;
     tmva.fWeightMatrix2to3[4][6] = 0.553432299251619;
     tmva.fWeightMatrix2to3[5][6] = -2.00310191419927;
     tmva.fWeightMatrix2to3[6][6] = -0.972887313360374;
     tmva.fWeightMatrix2to3[0][7] = -0.00952422797350496;
     tmva.fWeightMatrix2to3[1][7] = -1.15043208722106;
     tmva.fWeightMatrix2to3[2][7] = 0.646482949790799;
     tmva.fWeightMatrix2to3[3][7] = 0.541747178992036;
     tmva.fWeightMatrix2to3[4][7] = 0.856232431477478;
     tmva.fWeightMatrix2to3[5][7] = 0.101887676904865;
     tmva.fWeightMatrix2to3[6][7] = -0.109826032567475;
     tmva.fWeightMatrix2to3[0][8] = -0.330563362034855;
     tmva.fWeightMatrix2to3[1][8] = 0.485458770361598;
     tmva.fWeightMatrix2to3[2][8] = -0.783414671398449;
     tmva.fWeightMatrix2to3[3][8] = -0.891742600447044;
     tmva.fWeightMatrix2to3[4][8] = -1.22573773056992;
     tmva.fWeightMatrix2to3[5][8] = -0.861200300490884;
     tmva.fWeightMatrix2to3[6][8] = -2.77863046899075;
     tmva.fWeightMatrix2to3[0][9] = -3.21025190679825;
     tmva.fWeightMatrix2to3[1][9] = 5.05459734017371;
     tmva.fWeightMatrix2to3[2][9] = 1.42880862651111;
     tmva.fWeightMatrix2to3[3][9] = 1.88916122723334;
     tmva.fWeightMatrix2to3[4][9] = -4.61241600524382;
     tmva.fWeightMatrix2to3[5][9] = -3.90792395129795;
     tmva.fWeightMatrix2to3[6][9] = 1.51275256453558;
     // weight matrix from layer 3 to 4
     tmva.fWeightMatrix3to4[0][0] = 0.490641348474588;
     tmva.fWeightMatrix3to4[0][1] = 0.483067099717087;
     tmva.fWeightMatrix3to4[0][2] = -0.541242670164575;
     tmva.fWeightMatrix3to4[0][3] = -0.58053622359768;
     tmva.fWeightMatrix3to4[0][4] = 0.425277394090215;
     tmva.fWeightMatrix3to4[0][5] = 0.469384698563024;
     tmva.fWeightMatrix3to4[0][6] = -0.472826385903612;
     tmva.fWeightMatrix3to4[0][7] = 1.75985681009234;
  }

  inline void TMVA2_Init( TMVA& tmva ) {
    // initialize constants
    TMVA2_Initialize(tmva);
    
    // initialize transformation
    TMVA2_InitTransform_1(tmva);
  }
    

} // namespace Tracking
 
} // namespace SciFi
