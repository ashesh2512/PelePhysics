#ifndef MECHANISM_CUDA_H
#define MECHANISM_CUDA_H

#include "mechanism.H"

// get mean internal energy in mass units
__device__ __forceinline__ void
CKUBMS_CUDA(
  const amrex::Real T, volatile const amrex::Real y[], amrex::Real& ubms, volatile amrex::Real scratch[], const int tidx)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  for (int i = tidx; i < NUM_SPECIES; i += 32) {
    scratch[i] = y[i] *
                (+1.50000000e+00 + 3.52666409e-13 * T - 6.65306547e-16 * T2 +
                 5.75204080e-19 * T3 - 1.85546466e-22 * T4 + 2.54736599e+04 * invT) *
                0.9920634920634921;
  }
  __syncwarp();

  // Hard coded for warp sizes of 32
  scratch[tidx] += scratch[tidx + 32];
  scratch[tidx] += scratch[tidx + 16];
  scratch[tidx] += scratch[tidx + 8];
  scratch[tidx] += scratch[tidx + 4];
  scratch[tidx] += scratch[tidx + 2];
  scratch[tidx] += scratch[tidx + 1];

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  ubms = scratch[0] * RT;
}

// Returns the mean specific heat at CV (Eq. 36)
__device__ __forceinline__ void
CKCVBS_CUDA(
  const amrex::Real T, volatile const amrex::Real y[], amrex::Real& cvbs, volatile amrex::Real scratch[], const int tidx)
{
  amrex::Real result = 0.0;
  // compute Cv/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  for (int i = tidx; i < NUM_SPECIES; i += 32) {
    scratch[i] = y[i] *
                 (+1.50000000e+00 + 7.05332819e-13 * T - 1.99591964e-15 * T2 +
                  2.30081632e-18 * T3 - 9.27732332e-22 * T4) *
                 0.9920634920634921;
  }
  __syncwarp();

  // Hard coded for warp sizes of 32
  scratch[tidx] += scratch[tidx + 32];
  scratch[tidx] += scratch[tidx + 16];
  scratch[tidx] += scratch[tidx + 8];
  scratch[tidx] += scratch[tidx + 4];
  scratch[tidx] += scratch[tidx + 2];
  scratch[tidx] += scratch[tidx + 1];

  cvbs = scratch[0] * 8.31446261815324e+07;
}

// compute the e/(RT) at the given temperature
__device__ __forceinline__ void
speciesInternalEnergy_CUDA(
  volatile amrex::Real* species, const amrex::Real T, const int tidx)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  for (int i = tidx; i < NUM_SPECIES; i += 32) {
    species[i] = +2.16826710e+00 - 1.63965942e-03 * T + 2.21435465e-06 * T2 -
                 1.53201656e-09 * T3 + 4.22531942e-13 * T4 +
                 2.91222592e+04 * invT;
  }
}

__device__ __forceinline__ void
productionRate_CUDA(
  volatile amrex::Real* wdot,
  volatile const amrex::Real* sc,
  const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 53; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 53; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[53];
  gibbs(g_RT, T);

  {
    // reaction 4:  H + O2 (+M) <=> HO2 (+M)
    const amrex::Real k_f = 5116000 * exp((0.44) * logT);
    amrex::Real Corr = mixture + (10.89) * sc[6] + (-0.15) * sc[8] +
                       (0.0900000000000001) * sc[16] + (1.18) * sc[17];
    const amrex::Real redP = Corr / k_f * 63280000 * exp(-1.4 * logT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(0.5 * exp(-T * 1e+30) + 0.5 * exp(-T * 1e-30) + 0.0);
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[8]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] - g_RT[4] + g_RT[8])) * (refC) * (sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 17:  2 OH (+M) <=> H2O2 (+M)
    const amrex::Real k_f = 111000000 * exp((-0.37) * logT);
    amrex::Real Corr =
      mixture + sc[5] + (5) * sc[6] + (0.75) * sc[16] + (2.6) * sc[17];
    const amrex::Real redP =
      Corr / k_f * 201000 * exp(-0.584 * logT - -1153.87581469022 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.2654 * exp(-T * 0.0106382978723404) +
      0.7346 * exp(-T * 0.000569476082004556) + exp(-5182 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * ((sc[3] * sc[3]));
    const amrex::Real qr =
      Corr * k_f * exp(-(2.000000 * g_RT[3] - g_RT[7])) * (refC) * (sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= 2.000000 * qdot;
    wdot[7] += qdot;
  }

  {
    // reaction 37:  CO + H2 (+M) <=> CH2O (+M)
    const amrex::Real k_f = 43 * exp((1.5) * logT - (40056.0465980557) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 5.07e+15 * exp(-3.42 * logT - 42446.3257606281 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.0679999999999999 * exp(-T * 0.0050761421319797) +
      0.932 * exp(-T * 0.000649350649350649) + exp(-10300 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[5] * sc[16]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[5] - g_RT[14] + g_RT[16])) * (refC) * (sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[5] -= qdot;
    wdot[14] += qdot;
    wdot[16] -= qdot;
  }

  {
    // reaction 38:  H + HCO (+M) <=> CH2O (+M)
    const amrex::Real k_f =
      1090000 * exp((0.48) * logT - (-130.836333109227) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 1350000000000 * exp(-2.57 * logT - 717.083748771726 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.2176 * exp(-T * 0.003690036900369) +
      0.7824 * exp(-T * 0.000362976406533575) + exp(-6570 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[13]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[13] - g_RT[14])) * (refC) * (sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[13] -= qdot;
    wdot[14] += qdot;
  }

  {
    // reaction 39:  CH2 + H (+M) <=> CH3 (+M)
    const amrex::Real k_f = 25000000000 * exp((-0.8) * logT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 3.2e+15 * exp(-3.14 * logT - 618.956498939805 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.32 * exp(-T * 0.0128205128205128) +
      0.68 * exp(-T * 0.00050125313283208) + exp(-5590 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[9]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[9] - g_RT[11])) * (refC) * (sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[9] -= qdot;
    wdot[11] += qdot;
  }

  {
    // reaction 58:  CH2O + H (+M) <=> CH3O (+M)
    const amrex::Real k_f =
      540000 * exp((0.454) * logT - (1308.36333109227) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 2.2e+18 * exp(-4.8 * logT - 2797.88466187424 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.242 * exp(-T * 0.0106382978723404) +
      0.758 * exp(-T * 0.000643086816720257) + exp(-4200 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[14]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[14] - g_RT[15])) * (refC) * (sc[15]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[14] -= qdot;
    wdot[15] += qdot;
  }

  {
    // reaction 64:  CH3 + H (+M) <=> CH4 (+M)
    const amrex::Real k_f =
      12700000000 * exp((-0.63) * logT - (192.731983003208) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 2.477e+21 * exp(-4.76 * logT - 1227.84866456352 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.217 * exp(-T * 0.0135135135135135) +
      0.783 * exp(-T * 0.000340020401224073) + exp(-6964 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[11]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[11] - g_RT[12])) * (refC) * (sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
  }

  {
    // reaction 77:  2 CH3 (+M) <=> C2H6 (+M)
    const amrex::Real k_f =
      21200000000 * exp((-0.97) * logT - (311.994332798926) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 1.77e+38 * exp(-9.67 * logT - 3130.00766130536 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.4675 * exp(-T * 0.00662251655629139) +
      0.5325 * exp(-T * 0.000963391136801541) + exp(-4970 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * ((sc[11] * sc[11]));
    const amrex::Real qr =
      Corr * k_f * exp(-(2.000000 * g_RT[11] - g_RT[22])) * (refC) * (sc[22]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= 2.000000 * qdot;
    wdot[22] += qdot;
  }

  {
    // reaction 90:  C2H3 (+M) <=> C2H2 + H (+M)
    const amrex::Real k_f =
      38600000 * exp((1.62) * logT - (18643.2716780664) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[18] + (2) * sc[20] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 2.565e+21 * exp(-3.4 * logT - 18014.5125184767 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      -0.9816 * exp(-T * 0.000185745862510913) +
      1.9816 * exp(-T * 0.232926488400261) + exp(0.0795 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[19]);
    const amrex::Real qr = Corr * k_f * exp(-(-g_RT[1] - g_RT[18] + g_RT[19])) *
                           (refCinv) * (sc[1] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[18] += qdot;
    wdot[19] -= qdot;
  }

  {
    // reaction 95:  C2H3 + H (+M) <=> C2H4 (+M)
    const amrex::Real k_f =
      6080000 * exp((0.27) * logT - (140.900666425322) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[18] + (2) * sc[20] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 1.4e+18 * exp(-3.86 * logT - 1670.67933047167 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.218 * exp(-T * 0.00481927710843374) +
      0.782 * exp(-T * 0.000375516334960571) + exp(-6095 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[19]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[19] - g_RT[20])) * (refC) * (sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 107:  C2H3 + CH3 (+M) <=> C3H6 (+M)
    const amrex::Real k_f = 25000000;
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[18] + (2) * sc[20] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 4.27e+46 * exp(-11.94 * logT - 4916.32618157895 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.825 * exp(-T * 0.000745934656124124) +
      0.175 * exp(-T * 1.66666666666667e-05) + exp(-10139.8 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[11] * sc[19]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[11] + g_RT[19] - g_RT[25])) * (refC) * (sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[19] -= qdot;
    wdot[25] += qdot;
  }

  {
    // reaction 113:  C2H4 + H (+M) <=> C2H5 (+M)
    const amrex::Real k_f =
      3975 * exp((1.28) * logT - (649.466525387545) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 4715000 * exp(-380.059419015673 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.24 * exp(-T * 0.025) + 0.76 * exp(-T * 0.000975609756097561) + 0.0);
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[20]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[20] - g_RT[21])) * (refC) * (sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 126:  C2H5 + H (+M) <=> C2H6 (+M)
    const amrex::Real k_f =
      521000000000 * exp((-0.99) * logT - (795.082331971457) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 1.99e+29 * exp(-7.08 * logT - 3364.00341090455 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.1578 * exp(-T * 0.008) + 0.8422 * exp(-T * 0.000450653447498873) +
      exp(-6882 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[21]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[21] - g_RT[22])) * (refC) * (sc[22]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[21] -= qdot;
    wdot[22] += qdot;
  }

  {
    // reaction 134:  C2H3 + C2H5 (+M) <=> C4H81 (+M)
    const amrex::Real k_f = 15000000;
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 1.55e+44 * exp(-11.79 * logT - 4521.1501339225 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.802 * exp(-T * 0.000439000834101585) +
      0.198 * exp(-T * 1.66666666666667e-05) + exp(-5723.2 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[19] * sc[21]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[19] + g_RT[21] - g_RT[29])) * (refC) * (sc[29]);
    const amrex::Real qdot = qf - qr;
    wdot[19] -= qdot;
    wdot[21] -= qdot;
    wdot[29] += qdot;
  }

  {
    // reaction 141:  H + aC3H5 (+M) <=> C3H6 (+M)
    const amrex::Real k_f = 200000000;
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 1.33e+48 * exp(-12 * logT - 3003.09641818941 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.98 * exp(-T * 0.000911909538573774) +
      0.02 * exp(-T * 0.000911909538573774) + exp(-6859.5 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[24]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[24] - g_RT[25])) * (refC) * (sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[24] -= qdot;
    wdot[25] += qdot;
  }

  {
    // reaction 148:  CH3 + aC3H5 (+M) <=> C4H81 (+M)
    const amrex::Real k_f =
      100000000 * exp((-0.32) * logT - (-131.993731440578) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 3.91e+48 * exp(-12.81 * logT - 3145.1041612795 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.896 * exp(-T * 0.00062266500622665) +
      0.104 * exp(-T * 1.66666666666667e-05) + exp(-6118.4 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[11] * sc[24]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[11] + g_RT[24] - g_RT[29])) * (refC) * (sc[29]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[24] -= qdot;
    wdot[29] += qdot;
  }

  {
    // reaction 149:  C3H6 + H (+M) <=> nC3H7 (+M)
    const amrex::Real k_f = 13300000 * exp(-(1640.83858218945) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 6.26e+26 * exp(-6.66 * logT - 3522.51666063304 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(+1 * exp(-T * 0.000763358778625954) + exp(-48097 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[25]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[25] - g_RT[26])) * (refC) * (sc[26]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[25] -= qdot;
    wdot[26] += qdot;
  }

  {
    // reaction 168:  C4H7 + H (+M) <=> C4H81 (+M)
    const amrex::Real k_f = 36000000;
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 3.01e+36 * exp(-9.32 * logT - 2935.56474163841 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.502 * exp(-T * 0.00076103500761035) +
      0.498 * exp(-T * 0.00076103500761035) + exp(-50000 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[28]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[28] - g_RT[29])) * (refC) * (sc[29]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[28] -= qdot;
    wdot[29] += qdot;
  }

  {
    // reaction 172:  C4H81 + H (+M) <=> pC4H9 (+M)
    const amrex::Real k_f = 13300000 * exp(-(1640.83858218945) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 6.26e+26 * exp(-6.66 * logT - 3522.51666063304 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(+1 * exp(-T * 0.000763358778625954) + exp(-48097 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[29]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[29] - g_RT[30])) * (refC) * (sc[30]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[29] -= qdot;
    wdot[30] += qdot;
  }

  {
    // reaction 192:  C5H10 + H (+M) <=> PXC5H11 (+M)
    const amrex::Real k_f = 13300000 * exp(-(1640.83858218945) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 6.26e+26 * exp(-6.66 * logT - 3522.51666063304 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(+1 * exp(-T * 0.000763358778625954) + exp(-48097 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[32]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[32] - g_RT[33])) * (refC) * (sc[33]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[32] -= qdot;
    wdot[33] += qdot;
  }

  {
    // reaction 196:  C6H12 + H (+M) <=> PXC6H13 (+M)
    const amrex::Real k_f = 13300000 * exp(-(1640.83858218945) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 6.26e+26 * exp(-6.66 * logT - 3522.51666063304 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(+1 * exp(-T * 0.000763358778625954) + exp(-48097 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[34]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[34] - g_RT[35])) * (refC) * (sc[35]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[34] -= qdot;
    wdot[35] += qdot;
  }

  {
    // reaction 200:  C7H14 + H (+M) <=> PXC7H15 (+M)
    const amrex::Real k_f = 13300000 * exp(-(1640.83858218945) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 6.26e+26 * exp(-6.66 * logT - 3522.51666063304 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(+1 * exp(-T * 0.000763358778625954) + exp(-48097 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[36]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[36] - g_RT[37])) * (refC) * (sc[37]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[36] -= qdot;
    wdot[37] += qdot;
  }

  {
    // reaction 204:  C8H16 + H (+M) <=> PXC8H17 (+M)
    const amrex::Real k_f = 13300000 * exp(-(1640.83858218945) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 6.26e+26 * exp(-6.66 * logT - 3522.51666063304 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(+1 * exp(-T * 0.000763358778625954) + exp(-48097 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[38]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[38] - g_RT[39])) * (refC) * (sc[39]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[38] -= qdot;
    wdot[39] += qdot;
  }

  {
    // reaction 208:  C9H18 + H (+M) <=> PXC9H19 (+M)
    const amrex::Real k_f = 13300000 * exp(-(1640.83858218945) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 6.26e+26 * exp(-6.66 * logT - 3522.51666063304 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(+1 * exp(-T * 0.000763358778625954) + exp(-48097 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[40]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[40] - g_RT[41])) * (refC) * (sc[41]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[40] -= qdot;
    wdot[41] += qdot;
  }

  {
    // reaction 212:  C10H20 + H (+M) <=> PXC10H21 (+M)
    const amrex::Real k_f = 13300000 * exp(-(1640.83858218945) * invT);
    amrex::Real Corr = mixture + sc[5] + (5) * sc[6] + sc[12] + (0.5) * sc[16] +
                       sc[17] + (2) * sc[22];
    const amrex::Real redP =
      Corr / k_f * 6.26e+26 * exp(-6.66 * logT - 3522.51666063304 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(+1 * exp(-T * 0.000763358778625954) + exp(-48097 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[42]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[42] - g_RT[43])) * (refC) * (sc[43]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[42] -= qdot;
    wdot[43] += qdot;
  }

  {
    // reaction 28:  CO + O (+M) <=> CO2 (+M)
    const amrex::Real k_f = 13620 * exp(-(1199.66853127845) * invT);
    amrex::Real Corr =
      mixture + sc[5] + (11) * sc[6] + (0.75) * sc[16] + (2.6) * sc[17];
    const amrex::Real redP =
      Corr / k_f * 1173000000000 * exp(-2.79 * logT - 2108.98104638758 * invT);
    Corr = redP / (1.0 + redP);
    const amrex::Real qf = Corr * k_f * (sc[2] * sc[16]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[2] + g_RT[16] - g_RT[17])) * (refC) * (sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[16] -= qdot;
    wdot[17] += qdot;
  }

  {
    // reaction 18:  2 H + M <=> H2 + M
    const amrex::Real k_f = 1780000 * exp((-1) * logT);
    const amrex::Real Corr =
      mixture + (-1) * sc[5] + (-1) * sc[6] + (-1) * sc[17];
    const amrex::Real qf = Corr * k_f * ((sc[1] * sc[1]));
    const amrex::Real qr =
      Corr * k_f * exp(-(2.000000 * g_RT[1] - g_RT[5])) * (refC) * (sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= 2.000000 * qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 19:  H + OH + M <=> H2O + M
    const amrex::Real k_f = 44000000000 * exp((-2) * logT);
    const amrex::Real Corr =
      mixture + sc[5] + (5.3) * sc[6] + (0.75) * sc[16] + (2.6) * sc[17];
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[3]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[3] - g_RT[6])) * (refC) * (sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[3] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 20:  2 O + M <=> O2 + M
    const amrex::Real k_f = 120000 * exp((-1) * logT);
    const amrex::Real Corr = mixture + (1.4) * sc[5] + (14.4) * sc[6] +
                             (0.75) * sc[16] + (2.6) * sc[17];
    const amrex::Real qf = Corr * k_f * ((sc[2] * sc[2]));
    const amrex::Real qr =
      Corr * k_f * exp(-(2.000000 * g_RT[2] - g_RT[8])) * (refC) * (sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= 2.000000 * qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 21:  2 H + H2 <=> H2 + H2
    const amrex::Real k_f = 90000 * exp((-0.6) * logT);
    const amrex::Real qf = k_f * ((sc[1] * sc[1]) * sc[5]);
    const amrex::Real qr =
      k_f * exp(-(2.000000 * g_RT[1] - g_RT[5])) * (refC) * ((sc[5] * sc[5]));
    const amrex::Real qdot = qf - qr;
    wdot[1] -= 2.000000 * qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 22:  2 H + H2O <=> H2 + H2O
    const amrex::Real k_f = 56240000 * exp((-1.25) * logT);
    const amrex::Real qf = k_f * ((sc[1] * sc[1]) * sc[6]);
    const amrex::Real qr =
      k_f * exp(-(2.000000 * g_RT[1] - g_RT[5])) * (refC) * (sc[5] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= 2.000000 * qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 23:  2 H + CO2 <=> H2 + CO2
    const amrex::Real k_f = 550000000 * exp((-2) * logT);
    const amrex::Real qf = k_f * ((sc[1] * sc[1]) * sc[17]);
    const amrex::Real qr =
      k_f * exp(-(2.000000 * g_RT[1] - g_RT[5])) * (refC) * (sc[5] * sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= 2.000000 * qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 24:  H + O + M <=> OH + M
    const amrex::Real k_f = 9428000 * exp((-1) * logT);
    const amrex::Real Corr =
      mixture + sc[5] + (11) * sc[6] + (0.75) * sc[16] + (2.6) * sc[17];
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[2]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[2] - g_RT[3])) * (refC) * (sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
  }

  {
    // reaction 30:  HCO + M <=> CO + H + M
    const amrex::Real k_f =
      187000000000 * exp((-1) * logT - (8554.68331868023) * invT);
    const amrex::Real Corr =
      mixture + sc[5] + (-1) * sc[6] + (0.75) * sc[16] + (2.6) * sc[17];
    const amrex::Real qf = Corr * k_f * (sc[13]);
    const amrex::Real qr = Corr * k_f * exp(-(-g_RT[1] + g_RT[13] - g_RT[16])) *
                           (refCinv) * (sc[1] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 36:  HCO + H2O <=> CO + H + H2O
    const amrex::Real k_f =
      2244000000000 * exp((-1) * logT - (8554.68331868023) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[13]);
    const amrex::Real qr = k_f * exp(-(-g_RT[1] + g_RT[13] - g_RT[16])) *
                           (refCinv) * (sc[1] * sc[6] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 0:  H + O2 <=> O + OH
    const amrex::Real k_f = 97560000 * exp(-(7468.87259020676) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[8])) * (sc[2] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[3] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 1:  H2 + O <=> H + OH
    const amrex::Real k_f =
      0.04589 * exp((2.7) * logT - (3150.13632793755) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[5]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[5])) * (sc[1] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[5] -= qdot;
  }

  {
    // reaction 2:  H2 + OH <=> H + H2O
    const amrex::Real k_f =
      102.4 * exp((1.6) * logT - (1659.74946449039) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[5]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[3] + g_RT[5] - g_RT[6])) * (sc[1] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[5] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 3:  2 OH <=> H2O + O
    const amrex::Real k_f =
      0.03973 * exp((2.4) * logT - (-1061.78716484796) * invT);
    const amrex::Real qf = k_f * ((sc[3] * sc[3]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[2] + 2.000000 * g_RT[3] - g_RT[6])) * (sc[2] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[3] -= 2.000000 * qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 5:  H + HO2 <=> 2 OH
    const amrex::Real k_f = 74850000 * exp(-(148.448916412392) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - 2.000000 * g_RT[3] + g_RT[4])) * ((sc[3] * sc[3]));
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[3] += 2.000000 * qdot;
    wdot[4] -= qdot;
  }

  {
    // reaction 6:  H2 + O2 <=> H + HO2
    const amrex::Real k_f =
      0.5916 * exp((2.433) * logT - (26923.0980538841) * invT);
    const amrex::Real qf = k_f * (sc[5] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] - g_RT[4] + g_RT[5] + g_RT[8])) * (sc[1] * sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 7:  HO2 + OH <=> H2O + O2
    const amrex::Real k_f = 28910000 * exp(-(-252.569476734047) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] + g_RT[4] - g_RT[6] - g_RT[8])) * (sc[6] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 8:  H + HO2 <=> H2O + O
    const amrex::Real k_f = 3970000 * exp(-(337.658382754967) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[2] + g_RT[4] - g_RT[6])) * (sc[2] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 9:  HO2 + O <=> O2 + OH
    const amrex::Real k_f = 40000000;
    const amrex::Real qf = k_f * (sc[2] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] + g_RT[4] - g_RT[8])) * (sc[3] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 10:  2 HO2 <=> H2O2 + O2
    const amrex::Real k_f = 130000 * exp(-(-820.243165261693) * invT);
    const amrex::Real qf = k_f * ((sc[4] * sc[4]));
    const amrex::Real qr =
      k_f * exp(-(2.000000 * g_RT[4] - g_RT[7] - g_RT[8])) * (sc[7] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= 2.000000 * qdot;
    wdot[7] += qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 11:  2 HO2 <=> H2O2 + O2
    const amrex::Real k_f = 365800000 * exp(-(6038.59998965664) * invT);
    const amrex::Real qf = k_f * ((sc[4] * sc[4]));
    const amrex::Real qr =
      k_f * exp(-(2.000000 * g_RT[4] - g_RT[7] - g_RT[8])) * (sc[7] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= 2.000000 * qdot;
    wdot[7] += qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 12:  H + H2O2 <=> H2O + OH
    const amrex::Real k_f = 24100000 * exp(-(1997.77016324474) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[3] - g_RT[6] + g_RT[7])) * (sc[3] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 13:  H + H2O2 <=> H2 + HO2
    const amrex::Real k_f = 6.05 * exp((2) * logT - (2616.72666218454) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[4] - g_RT[5] + g_RT[7])) * (sc[4] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 14:  H2O2 + O <=> HO2 + OH
    const amrex::Real k_f = 9.63 * exp((2) * logT - (1997.77016324474) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[7])) * (sc[3] * sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 15:  H2O2 + OH <=> H2O + HO2
    const amrex::Real k_f = 2000000 * exp(-(214.873516298615) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[4] - g_RT[6] + g_RT[7])) * (sc[4] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 16:  H2O2 + OH <=> H2O + HO2
    const amrex::Real k_f =
      2.67e+35 * exp((-7) * logT - (18920.9466342575) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[4] - g_RT[6] + g_RT[7])) * (sc[4] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 25:  CO + OH <=> CO2 + H
    const amrex::Real k_f =
      0.07046 * exp((2.053) * logT - (-178.979071526765) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[16]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[3] + g_RT[16] - g_RT[17])) * (sc[1] * sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[16] -= qdot;
    wdot[17] += qdot;
  }

  {
    // reaction 26:  CO + OH <=> CO2 + H
    const amrex::Real k_f =
      5757000 * exp((-0.664) * logT - (166.98238621398) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[16]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[3] + g_RT[16] - g_RT[17])) * (sc[1] * sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[16] -= qdot;
    wdot[17] += qdot;
  }

  {
    // reaction 27:  CO + HO2 <=> CO2 + OH
    const amrex::Real k_f =
      0.157 * exp((2.18) * logT - (9029.02038003442) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[16]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] + g_RT[16] - g_RT[17])) * (sc[3] * sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[16] -= qdot;
    wdot[17] += qdot;
  }

  {
    // reaction 29:  CO + O2 <=> CO2 + O
    const amrex::Real k_f = 1119000 * exp(-(24003.4349588851) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[16]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[2] + g_RT[8] + g_RT[16] - g_RT[17])) * (sc[2] * sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[8] -= qdot;
    wdot[16] -= qdot;
    wdot[17] += qdot;
  }

  {
    // reaction 31:  H + HCO <=> CO + H2
    const amrex::Real k_f = 120000000;
    const amrex::Real qf = k_f * (sc[1] * sc[13]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] + g_RT[13] - g_RT[16])) * (sc[5] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 32:  HCO + O <=> CO + OH
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[2] * sc[13]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] + g_RT[13] - g_RT[16])) * (sc[3] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 33:  HCO + O <=> CO2 + H
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[2] * sc[13]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[13] - g_RT[17])) * (sc[1] * sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[13] -= qdot;
    wdot[17] += qdot;
  }

  {
    // reaction 34:  HCO + OH <=> CO + H2O
    const amrex::Real k_f = 30200000;
    const amrex::Real qf = k_f * (sc[3] * sc[13]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] + g_RT[13] - g_RT[16])) * (sc[6] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 35:  HCO + O2 <=> CO + HO2
    const amrex::Real k_f =
      12040 * exp((0.807) * logT - (-365.838516040031) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[13]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] + g_RT[13] - g_RT[16])) * (sc[4] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 40:  CH2 + O <=> H + HCO
    const amrex::Real k_f = 80000000;
    const amrex::Real qf = k_f * (sc[2] * sc[9]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[9] - g_RT[13])) * (sc[1] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[9] -= qdot;
    wdot[13] += qdot;
  }

  {
    // reaction 41:  CH2 + OH <=> CH2O + H
    const amrex::Real k_f = 20000000;
    const amrex::Real qf = k_f * (sc[3] * sc[9]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[3] + g_RT[9] - g_RT[14])) * (sc[1] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[9] -= qdot;
    wdot[14] += qdot;
  }

  {
    // reaction 42:  CH2 + H2 <=> CH3 + H
    const amrex::Real k_f = 0.5 * exp((2) * logT - (3638.25649376812) * invT);
    const amrex::Real qf = k_f * (sc[5] * sc[9]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[5] + g_RT[9] - g_RT[11])) * (sc[1] * sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[5] -= qdot;
    wdot[9] -= qdot;
    wdot[11] += qdot;
  }

  {
    // reaction 43:  CH2 + O2 <=> HCO + OH
    const amrex::Real k_f = 10600000 * exp(-(754.82499870708) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[9]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[8] + g_RT[9] - g_RT[13])) * (sc[3] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[8] -= qdot;
    wdot[9] -= qdot;
    wdot[13] += qdot;
  }

  {
    // reaction 44:  CH2 + O2 <=> CO2 + 2 H
    const amrex::Real k_f = 2640000 * exp(-(754.82499870708) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[9]);
    const amrex::Real qr =
      k_f * exp(-(-2.000000 * g_RT[1] + g_RT[8] + g_RT[9] - g_RT[17])) *
      (refCinv) * ((sc[1] * sc[1]) * sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += 2.000000 * qdot;
    wdot[8] -= qdot;
    wdot[9] -= qdot;
    wdot[17] += qdot;
  }

  {
    // reaction 45:  CH2 + HO2 <=> CH2O + OH
    const amrex::Real k_f = 20000000;
    const amrex::Real qf = k_f * (sc[4] * sc[9]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] + g_RT[9] - g_RT[14])) * (sc[3] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;
    wdot[14] += qdot;
  }

  {
    // reaction 46:  2 CH2 <=> C2H2 + H2
    const amrex::Real k_f = 32000000;
    const amrex::Real qf = k_f * ((sc[9] * sc[9]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[5] + 2.000000 * g_RT[9] - g_RT[18])) * (sc[5] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[9] -= 2.000000 * qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 47:  CH2* + N2 <=> CH2 + N2
    const amrex::Real k_f = 15000000 * exp(-(301.929999482832) * invT);
    const amrex::Real qf = k_f * (sc[10] * sc[52]);
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[9] + g_RT[10] + g_RT[52] - g_RT[52])) *
                           (sc[9] * sc[52]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[52] -= qdot;
    wdot[52] += qdot;
  }

  {
    // reaction 48:  CH2* + O <=> CO + H2
    const amrex::Real k_f = 15000000;
    const amrex::Real qf = k_f * (sc[2] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[5] + g_RT[10] - g_RT[16])) * (sc[5] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[5] += qdot;
    wdot[10] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 49:  CH2* + O <=> H + HCO
    const amrex::Real k_f = 15000000;
    const amrex::Real qf = k_f * (sc[2] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[10] - g_RT[13])) * (sc[1] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;
  }

  {
    // reaction 50:  CH2* + OH <=> CH2O + H
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[3] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[3] + g_RT[10] - g_RT[14])) * (sc[1] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[10] -= qdot;
    wdot[14] += qdot;
  }

  {
    // reaction 51:  CH2* + H2 <=> CH3 + H
    const amrex::Real k_f = 70000000;
    const amrex::Real qf = k_f * (sc[5] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[5] + g_RT[10] - g_RT[11])) * (sc[1] * sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[5] -= qdot;
    wdot[10] -= qdot;
    wdot[11] += qdot;
  }

  {
    // reaction 52:  CH2* + O2 <=> CO + H + OH
    const amrex::Real k_f = 28000000;
    const amrex::Real qf = k_f * (sc[8] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] - g_RT[3] + g_RT[8] + g_RT[10] - g_RT[16])) *
      (refCinv) * (sc[1] * sc[3] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[3] += qdot;
    wdot[8] -= qdot;
    wdot[10] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 53:  CH2* + O2 <=> CO + H2O
    const amrex::Real k_f = 12000000;
    const amrex::Real qf = k_f * (sc[8] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[6] + g_RT[8] + g_RT[10] - g_RT[16])) * (sc[6] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[6] += qdot;
    wdot[8] -= qdot;
    wdot[10] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 54:  CH2* + H2O <=> CH2 + H2O
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[6] * sc[10]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[6] - g_RT[6] - g_RT[9] + g_RT[10])) * (sc[6] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[6] -= qdot;
    wdot[6] += qdot;
    wdot[9] += qdot;
    wdot[10] -= qdot;
  }

  {
    // reaction 55:  CH2* + CO <=> CH2 + CO
    const amrex::Real k_f = 9000000;
    const amrex::Real qf = k_f * (sc[10] * sc[16]);
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[9] + g_RT[10] + g_RT[16] - g_RT[16])) *
                           (sc[9] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[16] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 56:  CH2* + CO2 <=> CH2 + CO2
    const amrex::Real k_f = 7000000;
    const amrex::Real qf = k_f * (sc[10] * sc[17]);
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[9] + g_RT[10] + g_RT[17] - g_RT[17])) *
                           (sc[9] * sc[17]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[17] -= qdot;
    wdot[17] += qdot;
  }

  {
    // reaction 57:  CH2* + CO2 <=> CH2O + CO
    const amrex::Real k_f = 14000000;
    const amrex::Real qf = k_f * (sc[10] * sc[17]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[10] - g_RT[14] - g_RT[16] + g_RT[17])) *
                           (sc[14] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[10] -= qdot;
    wdot[14] += qdot;
    wdot[16] += qdot;
    wdot[17] -= qdot;
  }

  {
    // reaction 59:  CH2O + H <=> H2 + HCO
    const amrex::Real k_f =
      23000 * exp((1.05) * logT - (1648.03458051046) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[14]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[13] + g_RT[14])) * (sc[5] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 60:  CH2O + O <=> HCO + OH
    const amrex::Real k_f = 39000000 * exp(-(1781.38699694871) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[14]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[13] + g_RT[14])) * (sc[3] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 61:  CH2O + OH <=> H2O + HCO
    const amrex::Real k_f =
      3430 * exp((1.18) * logT - (-224.93784961471) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[14]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[13] + g_RT[14])) * (sc[6] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 62:  CH2O + O2 <=> HCO + HO2
    const amrex::Real k_f = 100000000 * exp(-(20128.6666321888) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[14]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] - g_RT[13] + g_RT[14])) * (sc[4] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 63:  CH2O + HO2 <=> H2O2 + HCO
    const amrex::Real k_f = 1000000 * exp(-(4025.73332643776) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[14]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[7] - g_RT[13] + g_RT[14])) * (sc[7] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 65:  CH3 + O <=> CH2O + H
    const amrex::Real k_f = 84300000;
    const amrex::Real qf = k_f * (sc[2] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[11] - g_RT[14])) * (sc[1] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[11] -= qdot;
    wdot[14] += qdot;
  }

  {
    // reaction 66:  CH3 + OH <=> CH2 + H2O
    const amrex::Real k_f = 56 * exp((1.6) * logT - (2727.43432866158) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[9] + g_RT[11])) * (sc[6] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[9] += qdot;
    wdot[11] -= qdot;
  }

  {
    // reaction 67:  CH3 + OH <=> CH2* + H2O
    const amrex::Real k_f = 25010000;
    const amrex::Real qf = k_f * (sc[3] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[10] + g_RT[11])) * (sc[6] * sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[10] += qdot;
    wdot[11] -= qdot;
  }

  {
    // reaction 68:  CH3 + O2 <=> CH3O + O
    const amrex::Real k_f = 30830000 * exp(-(14492.6399751759) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[2] + g_RT[8] + g_RT[11] - g_RT[15])) * (sc[2] * sc[15]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[8] -= qdot;
    wdot[11] -= qdot;
    wdot[15] += qdot;
  }

  {
    // reaction 69:  CH3 + O2 <=> CH2O + OH
    const amrex::Real k_f = 36000 * exp(-(4498.75699229419) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[8] + g_RT[11] - g_RT[14])) * (sc[3] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[8] -= qdot;
    wdot[11] -= qdot;
    wdot[14] += qdot;
  }

  {
    // reaction 70:  CH3 + HO2 <=> CH4 + O2
    const amrex::Real k_f = 1000000;
    const amrex::Real qf = k_f * (sc[4] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[8] + g_RT[11] - g_RT[12])) * (sc[8] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[8] += qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
  }

  {
    // reaction 71:  CH3 + HO2 <=> CH3O + OH
    const amrex::Real k_f = 13400000;
    const amrex::Real qf = k_f * (sc[4] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] + g_RT[11] - g_RT[15])) * (sc[3] * sc[15]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[11] -= qdot;
    wdot[15] += qdot;
  }

  {
    // reaction 72:  CH3 + H2O2 <=> CH4 + HO2
    const amrex::Real k_f =
      0.0245 * exp((2.47) * logT - (2606.66232886845) * invT);
    const amrex::Real qf = k_f * (sc[7] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[7] + g_RT[11] - g_RT[12])) * (sc[4] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[7] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
  }

  {
    // reaction 73:  CH3 + HCO <=> CH4 + CO
    const amrex::Real k_f = 8480000;
    const amrex::Real qf = k_f * (sc[11] * sc[13]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] + g_RT[13] - g_RT[16])) *
                           (sc[12] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 74:  CH2O + CH3 <=> CH4 + HCO
    const amrex::Real k_f =
      0.00332 * exp((2.81) * logT - (2948.84966161566) * invT);
    const amrex::Real qf = k_f * (sc[11] * sc[14]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] - g_RT[13] + g_RT[14])) *
                           (sc[12] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 75:  CH2 + CH3 <=> C2H4 + H
    const amrex::Real k_f = 40000000;
    const amrex::Real qf = k_f * (sc[9] * sc[11]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[9] + g_RT[11] - g_RT[20])) * (sc[1] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[9] -= qdot;
    wdot[11] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 76:  CH2* + CH3 <=> C2H4 + H
    const amrex::Real k_f = 12000000 * exp(-(-286.83349950869) * invT);
    const amrex::Real qf = k_f * (sc[10] * sc[11]);
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[1] + g_RT[10] + g_RT[11] - g_RT[20])) *
                           (sc[1] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[10] -= qdot;
    wdot[11] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 78:  2 CH3 <=> C2H5 + H
    const amrex::Real k_f =
      4990000 * exp((0.1) * logT - (5334.09665753003) * invT);
    const amrex::Real qf = k_f * ((sc[11] * sc[11]));
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[1] + 2.000000 * g_RT[11] - g_RT[21])) *
                           (sc[1] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[11] -= 2.000000 * qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 79:  CH3O + H <=> CH2O + H2
    const amrex::Real k_f = 20000000;
    const amrex::Real qf = k_f * (sc[1] * sc[15]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[14] + g_RT[15])) * (sc[5] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[14] += qdot;
    wdot[15] -= qdot;
  }

  {
    // reaction 80:  CH3O + H <=> CH3 + OH
    const amrex::Real k_f = 32000000;
    const amrex::Real qf = k_f * (sc[1] * sc[15]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[3] - g_RT[11] + g_RT[15])) * (sc[3] * sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[11] += qdot;
    wdot[15] -= qdot;
  }

  {
    // reaction 81:  CH3O + H <=> CH2* + H2O
    const amrex::Real k_f = 16000000;
    const amrex::Real qf = k_f * (sc[1] * sc[15]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[6] - g_RT[10] + g_RT[15])) * (sc[6] * sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[6] += qdot;
    wdot[10] += qdot;
    wdot[15] -= qdot;
  }

  {
    // reaction 82:  CH3O + O <=> CH2O + OH
    const amrex::Real k_f = 10000000;
    const amrex::Real qf = k_f * (sc[2] * sc[15]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[14] + g_RT[15])) * (sc[3] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[14] += qdot;
    wdot[15] -= qdot;
  }

  {
    // reaction 83:  CH3O + OH <=> CH2O + H2O
    const amrex::Real k_f = 5000000;
    const amrex::Real qf = k_f * (sc[3] * sc[15]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[14] + g_RT[15])) * (sc[6] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[14] += qdot;
    wdot[15] -= qdot;
  }

  {
    // reaction 84:  CH3O + O2 <=> CH2O + HO2
    const amrex::Real k_f =
      4.28e-19 * exp((7.6) * logT - (-1776.35483029066) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[15]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] - g_RT[14] + g_RT[15])) * (sc[4] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[14] += qdot;
    wdot[15] -= qdot;
  }

  {
    // reaction 85:  CH4 + H <=> CH3 + H2
    const amrex::Real k_f =
      660 * exp((1.62) * logT - (5454.86865732316) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[12]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[11] + g_RT[12])) * (sc[5] * sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;
  }

  {
    // reaction 86:  CH4 + O <=> CH3 + OH
    const amrex::Real k_f =
      1020 * exp((1.5) * logT - (4327.66332592059) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[12]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[11] + g_RT[12])) * (sc[3] * sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;
  }

  {
    // reaction 87:  CH4 + OH <=> CH3 + H2O
    const amrex::Real k_f = 100 * exp((1.6) * logT - (1570.03599731073) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[12]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[11] + g_RT[12])) * (sc[6] * sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;
  }

  {
    // reaction 88:  CH2 + CH4 <=> 2 CH3
    const amrex::Real k_f = 2.46 * exp((2) * logT - (4161.60182620503) * invT);
    const amrex::Real qf = k_f * (sc[9] * sc[12]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[9] - 2.000000 * g_RT[11] + g_RT[12])) *
                           ((sc[11] * sc[11]));
    const amrex::Real qdot = qf - qr;
    wdot[9] -= qdot;
    wdot[11] += 2.000000 * qdot;
    wdot[12] -= qdot;
  }

  {
    // reaction 89:  CH2* + CH4 <=> 2 CH3
    const amrex::Real k_f = 16000000 * exp(-(-286.83349950869) * invT);
    const amrex::Real qf = k_f * (sc[10] * sc[12]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[10] - 2.000000 * g_RT[11] + g_RT[12])) *
                           ((sc[11] * sc[11]));
    const amrex::Real qdot = qf - qr;
    wdot[10] -= qdot;
    wdot[11] += 2.000000 * qdot;
    wdot[12] -= qdot;
  }

  {
    // reaction 91:  C2H2 + O <=> CH2 + CO
    const amrex::Real k_f = 4.08 * exp((2) * logT - (956.111665028967) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[18]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[9] - g_RT[16] + g_RT[18])) * (sc[9] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[9] += qdot;
    wdot[16] += qdot;
    wdot[18] -= qdot;
  }

  {
    // reaction 92:  C2H2 + OH <=> CH3 + CO
    const amrex::Real k_f =
      4.83e-10 * exp((4) * logT - (-1006.43333160944) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[18]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[3] - g_RT[11] - g_RT[16] + g_RT[18])) *
                           (sc[11] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[11] += qdot;
    wdot[16] += qdot;
    wdot[18] -= qdot;
  }

  {
    // reaction 93:  C2H2 + HCO <=> C2H3 + CO
    const amrex::Real k_f = 10 * exp((2) * logT - (3019.29999482832) * invT);
    const amrex::Real qf = k_f * (sc[13] * sc[18]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[13] - g_RT[16] + g_RT[18] - g_RT[19])) *
                           (sc[16] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[13] -= qdot;
    wdot[16] += qdot;
    wdot[18] -= qdot;
    wdot[19] += qdot;
  }

  {
    // reaction 94:  C2H2 + CH3 <=> aC3H5
    const amrex::Real k_f =
      2.68e+47 * exp((-12.82) * logT - (17979.9314692026) * invT);
    const amrex::Real qf = k_f * (sc[11] * sc[18]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[11] + g_RT[18] - g_RT[24])) * (refC) * (sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[18] -= qdot;
    wdot[24] += qdot;
  }

  {
    // reaction 96:  C2H3 + H <=> C2H2 + H2
    const amrex::Real k_f = 90000000;
    const amrex::Real qf = k_f * (sc[1] * sc[19]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[18] + g_RT[19])) * (sc[5] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[18] += qdot;
    wdot[19] -= qdot;
  }

  {
    // reaction 97:  C2H3 + O <=> CH3 + CO
    const amrex::Real k_f = 48000000;
    const amrex::Real qf = k_f * (sc[2] * sc[19]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[2] - g_RT[11] - g_RT[16] + g_RT[19])) *
                           (sc[11] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[11] += qdot;
    wdot[16] += qdot;
    wdot[19] -= qdot;
  }

  {
    // reaction 98:  C2H3 + OH <=> C2H2 + H2O
    const amrex::Real k_f = 30110000;
    const amrex::Real qf = k_f * (sc[3] * sc[19]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[18] + g_RT[19])) * (sc[6] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[18] += qdot;
    wdot[19] -= qdot;
  }

  {
    // reaction 99:  C2H3 + O2 <=> C2H2 + HO2
    const amrex::Real k_f =
      1.34 * exp((1.61) * logT - (-192.93326966953) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[19]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] - g_RT[18] + g_RT[19])) * (sc[4] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[18] += qdot;
    wdot[19] -= qdot;
  }

  {
    // reaction 100:  C2H3 + O2 <=> CH2CHO + O
    const amrex::Real k_f =
      300000 * exp((0.29) * logT - (5.53538332385192) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[19]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[2] + g_RT[8] + g_RT[19] - g_RT[23])) * (sc[2] * sc[23]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[8] -= qdot;
    wdot[19] -= qdot;
    wdot[23] += qdot;
  }

  {
    // reaction 101:  C2H3 + O2 <=> CH2O + HCO
    const amrex::Real k_f =
      46000000000 * exp((-1.39) * logT - (508.248832462767) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[19]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[8] - g_RT[13] - g_RT[14] + g_RT[19])) *
                           (sc[13] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[8] -= qdot;
    wdot[13] += qdot;
    wdot[14] += qdot;
    wdot[19] -= qdot;
  }

  {
    // reaction 102:  C2H3 + HO2 <=> CH2CHO + OH
    const amrex::Real k_f = 10000000;
    const amrex::Real qf = k_f * (sc[4] * sc[19]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] + g_RT[19] - g_RT[23])) * (sc[3] * sc[23]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[19] -= qdot;
    wdot[23] += qdot;
  }

  {
    // reaction 103:  C2H3 + H2O2 <=> C2H4 + HO2
    const amrex::Real k_f = 12100 * exp(-(-299.917132819613) * invT);
    const amrex::Real qf = k_f * (sc[7] * sc[19]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[7] + g_RT[19] - g_RT[20])) * (sc[4] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[7] -= qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 104:  C2H3 + HCO <=> C2H4 + CO
    const amrex::Real k_f = 90330000;
    const amrex::Real qf = k_f * (sc[13] * sc[19]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[13] - g_RT[16] + g_RT[19] - g_RT[20])) *
                           (sc[16] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[13] -= qdot;
    wdot[16] += qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 105:  C2H3 + HCO <=> C2H3CHO
    const amrex::Real k_f = 18000000;
    const amrex::Real qf = k_f * (sc[13] * sc[19]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[13] + g_RT[19] - g_RT[27])) * (refC) * (sc[27]);
    const amrex::Real qdot = qf - qr;
    wdot[13] -= qdot;
    wdot[19] -= qdot;
    wdot[27] += qdot;
  }

  {
    // reaction 106:  C2H3 + CH3 <=> C2H2 + CH4
    const amrex::Real k_f = 392000;
    const amrex::Real qf = k_f * (sc[11] * sc[19]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] - g_RT[18] + g_RT[19])) *
                           (sc[12] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[18] += qdot;
    wdot[19] -= qdot;
  }

  {
    // reaction 108:  C2H3 + CH3 <=> H + aC3H5
    const amrex::Real k_f =
      1.5e+18 * exp((-2.83) * logT - (9368.88788395227) * invT);
    const amrex::Real qf = k_f * (sc[11] * sc[19]);
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[1] + g_RT[11] + g_RT[19] - g_RT[24])) *
                           (sc[1] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[11] -= qdot;
    wdot[19] -= qdot;
    wdot[24] += qdot;
  }

  {
    // reaction 109:  2 C2H3 <=> C2H2 + C2H4
    const amrex::Real k_f = 960000;
    const amrex::Real qf = k_f * ((sc[19] * sc[19]));
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[18] + 2.000000 * g_RT[19] - g_RT[20])) *
                           (sc[18] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[18] += qdot;
    wdot[19] -= 2.000000 * qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 110:  CH2CHO <=> CH3 + CO
    const amrex::Real k_f =
      7.8e+41 * exp((-9.147) * logT - (23600.8616262414) * invT);
    const amrex::Real qf = k_f * (sc[23]);
    const amrex::Real qr = k_f * exp(-(-g_RT[11] - g_RT[16] + g_RT[23])) *
                           (refCinv) * (sc[11] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[11] += qdot;
    wdot[16] += qdot;
    wdot[23] -= qdot;
  }

  {
    // reaction 111:  CH2CHO + H <=> CH3 + HCO
    const amrex::Real k_f = 90000000;
    const amrex::Real qf = k_f * (sc[1] * sc[23]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[11] - g_RT[13] + g_RT[23])) *
                           (sc[11] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[13] += qdot;
    wdot[23] -= qdot;
  }

  {
    // reaction 112:  CH2CHO + O2 <=> CH2O + CO + OH
    const amrex::Real k_f = 18000;
    const amrex::Real qf = k_f * (sc[8] * sc[23]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[8] - g_RT[14] - g_RT[16] + g_RT[23])) *
      (refCinv) * (sc[3] * sc[14] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[8] -= qdot;
    wdot[14] += qdot;
    wdot[16] += qdot;
    wdot[23] -= qdot;
  }

  {
    // reaction 114:  C2H4 + H <=> C2H3 + H2
    const amrex::Real k_f =
      50.7 * exp((1.9) * logT - (6516.65582217112) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[20]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[19] + g_RT[20])) * (sc[5] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[19] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 115:  C2H4 + O <=> C2H3 + OH
    const amrex::Real k_f =
      15.1 * exp((1.9) * logT - (1882.03033010965) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[20]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[19] + g_RT[20])) * (sc[3] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[19] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 116:  C2H4 + O <=> CH3 + HCO
    const amrex::Real k_f =
      19.2 * exp((1.83) * logT - (110.707666477038) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[20]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[2] - g_RT[11] - g_RT[13] + g_RT[20])) *
                           (sc[11] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[11] += qdot;
    wdot[13] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 117:  C2H4 + O <=> CH2 + CH2O
    const amrex::Real k_f =
      0.384 * exp((1.83) * logT - (110.707666477038) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[20]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[9] - g_RT[14] + g_RT[20])) * (sc[9] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[9] += qdot;
    wdot[14] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 118:  C2H4 + OH <=> C2H3 + H2O
    const amrex::Real k_f = 3.6 * exp((2) * logT - (1258.0416645118) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[20]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[19] + g_RT[20])) * (sc[6] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[19] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 119:  C2H4 + HCO <=> C2H5 + CO
    const amrex::Real k_f = 10 * exp((2) * logT - (4025.73332643776) * invT);
    const amrex::Real qf = k_f * (sc[13] * sc[20]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[13] - g_RT[16] + g_RT[20] - g_RT[21])) *
                           (sc[16] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[13] -= qdot;
    wdot[16] += qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 120:  C2H4 + CH2 <=> H + aC3H5
    const amrex::Real k_f = 20000000 * exp(-(3019.29999482832) * invT);
    const amrex::Real qf = k_f * (sc[9] * sc[20]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[9] + g_RT[20] - g_RT[24])) * (sc[1] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[9] -= qdot;
    wdot[20] -= qdot;
    wdot[24] += qdot;
  }

  {
    // reaction 121:  C2H4 + CH2* <=> H + aC3H5
    const amrex::Real k_f = 50000000;
    const amrex::Real qf = k_f * (sc[10] * sc[20]);
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[1] + g_RT[10] + g_RT[20] - g_RT[24])) *
                           (sc[1] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[10] -= qdot;
    wdot[20] -= qdot;
    wdot[24] += qdot;
  }

  {
    // reaction 122:  C2H4 + CH3 <=> C2H3 + CH4
    const amrex::Real k_f = 0.227 * exp((2) * logT - (4629.59332540342) * invT);
    const amrex::Real qf = k_f * (sc[11] * sc[20]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] - g_RT[19] + g_RT[20])) *
                           (sc[12] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[19] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 123:  nC3H7 <=> C2H4 + CH3
    const amrex::Real k_f = 96000000000000 * exp(-(15611.2604302599) * invT);
    const amrex::Real qf = k_f * (sc[26]);
    const amrex::Real qr = k_f * exp(-(-g_RT[11] - g_RT[20] + g_RT[26])) *
                           (refCinv) * (sc[11] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[11] += qdot;
    wdot[20] += qdot;
    wdot[26] -= qdot;
  }

  {
    // reaction 124:  C2H4 + O2 <=> C2H3 + HO2
    const amrex::Real k_f = 42200000 * exp(-(30595.573280927) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[20]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] - g_RT[19] + g_RT[20])) * (sc[4] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[19] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 125:  C2H3 + C2H4 <=> C4H7
    const amrex::Real k_f =
      7.93e+32 * exp((-8.47) * logT - (7155.74098774311) * invT);
    const amrex::Real qf = k_f * (sc[19] * sc[20]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[19] + g_RT[20] - g_RT[28])) * (refC) * (sc[28]);
    const amrex::Real qdot = qf - qr;
    wdot[19] -= qdot;
    wdot[20] -= qdot;
    wdot[28] += qdot;
  }

  {
    // reaction 127:  C2H5 + H <=> C2H4 + H2
    const amrex::Real k_f = 2000000;
    const amrex::Real qf = k_f * (sc[1] * sc[21]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[20] + g_RT[21])) * (sc[5] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[20] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 128:  C2H5 + O <=> CH2O + CH3
    const amrex::Real k_f = 16040000;
    const amrex::Real qf = k_f * (sc[2] * sc[21]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[2] - g_RT[11] - g_RT[14] + g_RT[21])) *
                           (sc[11] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[11] += qdot;
    wdot[14] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 129:  C2H5 + O2 <=> C2H4 + HO2
    const amrex::Real k_f = 20000;
    const amrex::Real qf = k_f * (sc[8] * sc[21]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] - g_RT[20] + g_RT[21])) * (sc[4] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[20] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 130:  C2H5 + HO2 <=> C2H6 + O2
    const amrex::Real k_f = 300000;
    const amrex::Real qf = k_f * (sc[4] * sc[21]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[8] + g_RT[21] - g_RT[22])) * (sc[8] * sc[22]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[8] += qdot;
    wdot[21] -= qdot;
    wdot[22] += qdot;
  }

  {
    // reaction 131:  C2H5 + HO2 <=> C2H4 + H2O2
    const amrex::Real k_f = 300000;
    const amrex::Real qf = k_f * (sc[4] * sc[21]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[7] - g_RT[20] + g_RT[21])) * (sc[7] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[20] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 132:  C2H5 + HO2 <=> CH2O + CH3 + OH
    const amrex::Real k_f = 24000000;
    const amrex::Real qf = k_f * (sc[4] * sc[21]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[11] - g_RT[14] + g_RT[21])) *
      (refCinv) * (sc[3] * sc[11] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[14] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 133:  C2H5 + H2O2 <=> C2H6 + HO2
    const amrex::Real k_f = 8700 * exp(-(490.133032493797) * invT);
    const amrex::Real qf = k_f * (sc[7] * sc[21]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[7] + g_RT[21] - g_RT[22])) * (sc[4] * sc[22]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[7] -= qdot;
    wdot[21] -= qdot;
    wdot[22] += qdot;
  }

  {
    // reaction 135:  C2H3 + C2H5 <=> CH3 + aC3H5
    const amrex::Real k_f =
      3.9e+26 * exp((-5.22) * logT - (9937.0194996458) * invT);
    const amrex::Real qf = k_f * (sc[19] * sc[21]);
    const amrex::Real qr = k_f *
                           exp(-(-g_RT[11] + g_RT[19] + g_RT[21] - g_RT[24])) *
                           (sc[11] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[11] += qdot;
    wdot[19] -= qdot;
    wdot[21] -= qdot;
    wdot[24] += qdot;
  }

  {
    // reaction 136:  C2H6 + H <=> C2H5 + H2
    const amrex::Real k_f = 115 * exp((1.9) * logT - (3789.22149350954) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[22]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[21] + g_RT[22])) * (sc[5] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[21] += qdot;
    wdot[22] -= qdot;
  }

  {
    // reaction 137:  C2H6 + O <=> C2H5 + OH
    const amrex::Real k_f =
      89.8 * exp((1.92) * logT - (2863.30282842885) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[22]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[21] + g_RT[22])) * (sc[3] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[21] += qdot;
    wdot[22] -= qdot;
  }

  {
    // reaction 138:  C2H6 + OH <=> C2H5 + H2O
    const amrex::Real k_f =
      3.54 * exp((2.12) * logT - (437.798499250106) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[22]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[21] + g_RT[22])) * (sc[6] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[21] += qdot;
    wdot[22] -= qdot;
  }

  {
    // reaction 139:  C2H6 + CH2* <=> C2H5 + CH3
    const amrex::Real k_f = 40000000 * exp(-(-276.769166192596) * invT);
    const amrex::Real qf = k_f * (sc[10] * sc[22]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[10] - g_RT[11] - g_RT[21] + g_RT[22])) *
                           (sc[11] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[10] -= qdot;
    wdot[11] += qdot;
    wdot[21] += qdot;
    wdot[22] -= qdot;
  }

  {
    // reaction 140:  C2H6 + CH3 <=> C2H5 + CH4
    const amrex::Real k_f =
      6.14 * exp((1.74) * logT - (5258.61415765932) * invT);
    const amrex::Real qf = k_f * (sc[11] * sc[22]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] - g_RT[21] + g_RT[22])) *
                           (sc[12] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[21] += qdot;
    wdot[22] -= qdot;
  }

  {
    // reaction 142:  O + aC3H5 <=> C2H3CHO + H
    const amrex::Real k_f = 60000000;
    const amrex::Real qf = k_f * (sc[2] * sc[24]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[24] - g_RT[27])) * (sc[1] * sc[27]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[24] -= qdot;
    wdot[27] += qdot;
  }

  {
    // reaction 143:  OH + aC3H5 <=> C2H3CHO + 2 H
    const amrex::Real k_f =
      4.2e+26 * exp((-5.16) * logT - (15159.905274033) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[24]);
    const amrex::Real qr =
      k_f * exp(-(-2.000000 * g_RT[1] + g_RT[3] + g_RT[24] - g_RT[27])) *
      (refCinv) * ((sc[1] * sc[1]) * sc[27]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += 2.000000 * qdot;
    wdot[3] -= qdot;
    wdot[24] -= qdot;
    wdot[27] += qdot;
  }

  {
    // reaction 144:  O2 + aC3H5 <=> C2H3CHO + OH
    const amrex::Real k_f =
      18200000 * exp((-0.41) * logT - (11503.0297636301) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[24]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[8] + g_RT[24] - g_RT[27])) * (sc[3] * sc[27]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[8] -= qdot;
    wdot[24] -= qdot;
    wdot[27] += qdot;
  }

  {
    // reaction 145:  HO2 + aC3H5 <=> C3H6 + O2
    const amrex::Real k_f = 2660000;
    const amrex::Real qf = k_f * (sc[4] * sc[24]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[8] + g_RT[24] - g_RT[25])) * (sc[8] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[8] += qdot;
    wdot[24] -= qdot;
    wdot[25] += qdot;
  }

  {
    // reaction 146:  HO2 + aC3H5 <=> C2H3 + CH2O + OH
    const amrex::Real k_f = 6600000;
    const amrex::Real qf = k_f * (sc[4] * sc[24]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[14] - g_RT[19] + g_RT[24])) *
      (refCinv) * (sc[3] * sc[14] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[19] += qdot;
    wdot[24] -= qdot;
  }

  {
    // reaction 147:  HCO + aC3H5 <=> C3H6 + CO
    const amrex::Real k_f = 60000000;
    const amrex::Real qf = k_f * (sc[13] * sc[24]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[13] - g_RT[16] + g_RT[24] - g_RT[25])) *
                           (sc[16] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[13] -= qdot;
    wdot[16] += qdot;
    wdot[24] -= qdot;
    wdot[25] += qdot;
  }

  {
    // reaction 150:  C3H6 + H <=> C2H4 + CH3
    const amrex::Real k_f =
      8e+15 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[25]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[11] - g_RT[20] + g_RT[25])) *
                           (sc[11] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[20] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 151:  C3H6 + H <=> H2 + aC3H5
    const amrex::Real k_f =
      0.173 * exp((2.5) * logT - (1253.00949785375) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[25]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[24] + g_RT[25])) * (sc[5] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 152:  C3H6 + O <=> C2H3CHO + 2 H
    const amrex::Real k_f = 40 * exp((1.65) * logT - (164.551849718143) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[25]);
    const amrex::Real qr =
      k_f * exp(-(-2.000000 * g_RT[1] + g_RT[2] + g_RT[25] - g_RT[27])) *
      (refCinv) * ((sc[1] * sc[1]) * sc[27]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += 2.000000 * qdot;
    wdot[2] -= qdot;
    wdot[25] -= qdot;
    wdot[27] += qdot;
  }

  {
    // reaction 153:  C3H6 + O <=> C2H5 + HCO
    const amrex::Real k_f =
      35 * exp((1.65) * logT - (-489.126599162188) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[25]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[2] - g_RT[13] - g_RT[21] + g_RT[25])) *
                           (sc[13] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[13] += qdot;
    wdot[21] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 154:  C3H6 + O <=> OH + aC3H5
    const amrex::Real k_f =
      180000 * exp((0.7) * logT - (2958.91399493175) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[25]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[24] + g_RT[25])) * (sc[3] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 155:  C3H6 + OH <=> H2O + aC3H5
    const amrex::Real k_f = 3.1 * exp((2) * logT - (-149.958566409806) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[25]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[24] + g_RT[25])) * (sc[6] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 156:  C3H6 + HO2 <=> H2O2 + aC3H5
    const amrex::Real k_f =
      0.0096 * exp((2.6) * logT - (6999.74382134365) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[25]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[7] - g_RT[24] + g_RT[25])) * (sc[7] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 157:  C3H6 + CH3 <=> CH4 + aC3H5
    const amrex::Real k_f =
      2.2e-06 * exp((3.5) * logT - (2855.75457844178) * invT);
    const amrex::Real qf = k_f * (sc[11] * sc[25]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] - g_RT[24] + g_RT[25])) *
                           (sc[12] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 158:  C2H3CHO + H <=> C2H4 + HCO
    const amrex::Real k_f =
      108000 * exp((0.454) * logT - (2928.72099498347) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[27]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[13] - g_RT[20] + g_RT[27])) *
                           (sc[13] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[13] += qdot;
    wdot[20] += qdot;
    wdot[27] -= qdot;
  }

  {
    // reaction 159:  C2H3CHO + O <=> C2H3 + CO + OH
    const amrex::Real k_f = 30000000 * exp(-(1781.38699694871) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[27]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[16] - g_RT[19] + g_RT[27])) *
      (refCinv) * (sc[3] * sc[16] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[16] += qdot;
    wdot[19] += qdot;
    wdot[27] -= qdot;
  }

  {
    // reaction 160:  C2H3CHO + OH <=> C2H3 + CO + H2O
    const amrex::Real k_f =
      3430 * exp((1.18) * logT - (-224.93784961471) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[27]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[16] - g_RT[19] + g_RT[27])) *
      (refCinv) * (sc[6] * sc[16] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[16] += qdot;
    wdot[19] += qdot;
    wdot[27] -= qdot;
  }

  {
    // reaction 161:  H + nC3H7 <=> C2H5 + CH3
    const amrex::Real k_f =
      3.7e+18 * exp((-2.92) * logT - (6292.72440588802) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[26]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[11] - g_RT[21] + g_RT[26])) *
                           (sc[11] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[21] += qdot;
    wdot[26] -= qdot;
  }

  {
    // reaction 162:  H + nC3H7 <=> C3H6 + H2
    const amrex::Real k_f = 1800000;
    const amrex::Real qf = k_f * (sc[1] * sc[26]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[25] + g_RT[26])) * (sc[5] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[25] += qdot;
    wdot[26] -= qdot;
  }

  {
    // reaction 163:  O + nC3H7 <=> C2H5 + CH2O
    const amrex::Real k_f = 96000000;
    const amrex::Real qf = k_f * (sc[2] * sc[26]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[2] - g_RT[14] - g_RT[21] + g_RT[26])) *
                           (sc[14] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[14] += qdot;
    wdot[21] += qdot;
    wdot[26] -= qdot;
  }

  {
    // reaction 164:  OH + nC3H7 <=> C3H6 + H2O
    const amrex::Real k_f = 24000000;
    const amrex::Real qf = k_f * (sc[3] * sc[26]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[25] + g_RT[26])) * (sc[6] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[25] += qdot;
    wdot[26] -= qdot;
  }

  {
    // reaction 165:  O2 + nC3H7 <=> C3H6 + HO2
    const amrex::Real k_f = 90000;
    const amrex::Real qf = k_f * (sc[8] * sc[26]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] - g_RT[25] + g_RT[26])) * (sc[4] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[25] += qdot;
    wdot[26] -= qdot;
  }

  {
    // reaction 166:  HO2 + nC3H7 <=> C2H5 + CH2O + OH
    const amrex::Real k_f = 24000000;
    const amrex::Real qf = k_f * (sc[4] * sc[26]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[14] - g_RT[21] + g_RT[26])) *
      (refCinv) * (sc[3] * sc[14] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[21] += qdot;
    wdot[26] -= qdot;
  }

  {
    // reaction 167:  CH3 + nC3H7 <=> C3H6 + CH4
    const amrex::Real k_f = 11000000;
    const amrex::Real qf = k_f * (sc[11] * sc[26]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] - g_RT[25] + g_RT[26])) *
                           (sc[12] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[25] += qdot;
    wdot[26] -= qdot;
  }

  {
    // reaction 169:  C4H7 + H <=> CH3 + aC3H5
    const amrex::Real k_f =
      2e+15 * exp((-2) * logT - (5535.38332385192) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[28]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[11] - g_RT[24] + g_RT[28])) *
                           (sc[11] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[24] += qdot;
    wdot[28] -= qdot;
  }

  {
    // reaction 170:  C4H7 + HO2 <=> CH2O + OH + aC3H5
    const amrex::Real k_f = 24000000;
    const amrex::Real qf = k_f * (sc[4] * sc[28]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[14] - g_RT[24] + g_RT[28])) *
      (refCinv) * (sc[3] * sc[14] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[24] += qdot;
    wdot[28] -= qdot;
  }

  {
    // reaction 171:  C4H7 + HCO <=> C4H81 + CO
    const amrex::Real k_f = 60000000;
    const amrex::Real qf = k_f * (sc[13] * sc[28]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[13] - g_RT[16] + g_RT[28] - g_RT[29])) *
                           (sc[16] * sc[29]);
    const amrex::Real qdot = qf - qr;
    wdot[13] -= qdot;
    wdot[16] += qdot;
    wdot[28] -= qdot;
    wdot[29] += qdot;
  }

  {
    // reaction 173:  C4H81 + H <=> C2H4 + C2H5
    const amrex::Real k_f =
      1.6e+16 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[29]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[20] - g_RT[21] + g_RT[29])) *
                           (sc[20] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[20] += qdot;
    wdot[21] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 174:  C4H81 + H <=> C3H6 + CH3
    const amrex::Real k_f =
      3.2e+16 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[29]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[11] - g_RT[25] + g_RT[29])) *
                           (sc[11] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[25] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 175:  C4H81 + H <=> C4H7 + H2
    const amrex::Real k_f =
      0.65 * exp((2.54) * logT - (3399.73179417669) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[29]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[28] + g_RT[29])) * (sc[5] * sc[28]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 176:  C4H81 + O <=> HCO + nC3H7
    const amrex::Real k_f =
      330 * exp((1.45) * logT - (-202.293099653497) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[29]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[2] - g_RT[13] - g_RT[26] + g_RT[29])) *
                           (sc[13] * sc[26]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[13] += qdot;
    wdot[26] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 177:  C4H81 + O <=> C4H7 + OH
    const amrex::Real k_f = 15000000 * exp(-(2898.52799503519) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[29]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[28] + g_RT[29])) * (sc[3] * sc[28]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 178:  C4H81 + O <=> C4H7 + OH
    const amrex::Real k_f = 26000000 * exp(-(2249.3784961471) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[29]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[28] + g_RT[29])) * (sc[3] * sc[28]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 179:  C4H81 + OH <=> C4H7 + H2O
    const amrex::Real k_f =
      0.0007 * exp((2.66) * logT - (265.195182879087) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[29]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[28] + g_RT[29])) * (sc[6] * sc[28]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 180:  C4H81 + O2 <=> C4H7 + HO2
    const amrex::Real k_f = 20000000 * exp(-(25628.8247894344) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[29]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] - g_RT[28] + g_RT[29])) * (sc[4] * sc[28]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 181:  C4H81 + HO2 <=> C4H7 + H2O2
    const amrex::Real k_f = 1000000 * exp(-(7216.12698763968) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[29]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[7] - g_RT[28] + g_RT[29])) * (sc[7] * sc[28]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 182:  C4H81 + CH3 <=> C4H7 + CH4
    const amrex::Real k_f =
      4.5e-07 * exp((3.65) * logT - (3599.50881050116) * invT);
    const amrex::Real qf = k_f * (sc[11] * sc[29]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] - g_RT[28] + g_RT[29])) *
                           (sc[12] * sc[28]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;
  }

  {
    // reaction 183:  H + pC4H9 <=> 2 C2H5
    const amrex::Real k_f =
      3.7e+18 * exp((-2.92) * logT - (6292.72440588802) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[30]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - 2.000000 * g_RT[21] + g_RT[30])) *
                           ((sc[21] * sc[21]));
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[21] += 2.000000 * qdot;
    wdot[30] -= qdot;
  }

  {
    // reaction 184:  H + pC4H9 <=> C4H81 + H2
    const amrex::Real k_f = 1800000;
    const amrex::Real qf = k_f * (sc[1] * sc[30]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[5] - g_RT[29] + g_RT[30])) * (sc[5] * sc[29]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[29] += qdot;
    wdot[30] -= qdot;
  }

  {
    // reaction 185:  O + pC4H9 <=> CH2O + nC3H7
    const amrex::Real k_f = 96000000;
    const amrex::Real qf = k_f * (sc[2] * sc[30]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[2] - g_RT[14] - g_RT[26] + g_RT[30])) *
                           (sc[14] * sc[26]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[14] += qdot;
    wdot[26] += qdot;
    wdot[30] -= qdot;
  }

  {
    // reaction 186:  OH + pC4H9 <=> C4H81 + H2O
    const amrex::Real k_f = 24000000;
    const amrex::Real qf = k_f * (sc[3] * sc[30]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - g_RT[6] - g_RT[29] + g_RT[30])) * (sc[6] * sc[29]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[29] += qdot;
    wdot[30] -= qdot;
  }

  {
    // reaction 187:  O2 + pC4H9 <=> C4H81 + HO2
    const amrex::Real k_f = 270000;
    const amrex::Real qf = k_f * (sc[8] * sc[30]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[4] + g_RT[8] - g_RT[29] + g_RT[30])) * (sc[4] * sc[29]);
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[29] += qdot;
    wdot[30] -= qdot;
  }

  {
    // reaction 188:  HO2 + pC4H9 <=> CH2O + OH + nC3H7
    const amrex::Real k_f = 24000000;
    const amrex::Real qf = k_f * (sc[4] * sc[30]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[14] - g_RT[26] + g_RT[30])) *
      (refCinv) * (sc[3] * sc[14] * sc[26]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[26] += qdot;
    wdot[30] -= qdot;
  }

  {
    // reaction 189:  CH3 + pC4H9 <=> C4H81 + CH4
    const amrex::Real k_f = 11000000;
    const amrex::Real qf = k_f * (sc[11] * sc[30]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[11] - g_RT[12] - g_RT[29] + g_RT[30])) *
                           (sc[12] * sc[29]);
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[29] += qdot;
    wdot[30] -= qdot;
  }

  {
    // reaction 190:  C5H9 => C2H4 + aC3H5
    const amrex::Real k_f = 25000000000000 * exp(-(15106.1214767918) * invT);
    const amrex::Real qf = k_f * (sc[31]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[20] += qdot;
    wdot[24] += qdot;
    wdot[31] -= qdot;
  }

  {
    // reaction 191:  C5H9 => C2H3 + C3H6
    const amrex::Real k_f = 25000000000000 * exp(-(15106.1214767918) * invT);
    const amrex::Real qf = k_f * (sc[31]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[19] += qdot;
    wdot[25] += qdot;
    wdot[31] -= qdot;
  }

  {
    // reaction 193:  C5H10 + H <=> C2H4 + nC3H7
    const amrex::Real k_f =
      8e+15 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[32]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[20] - g_RT[26] + g_RT[32])) *
                           (sc[20] * sc[26]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[20] += qdot;
    wdot[26] += qdot;
    wdot[32] -= qdot;
  }

  {
    // reaction 194:  C5H10 + H <=> C2H5 + C3H6
    const amrex::Real k_f =
      1.6e+16 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[32]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[21] - g_RT[25] + g_RT[32])) *
                           (sc[21] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[21] += qdot;
    wdot[25] += qdot;
    wdot[32] -= qdot;
  }

  {
    // reaction 195:  C2H4 + nC3H7 <=> PXC5H11
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[20] * sc[26]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[20] + g_RT[26] - g_RT[33])) * (refC) * (sc[33]);
    const amrex::Real qdot = qf - qr;
    wdot[20] -= qdot;
    wdot[26] -= qdot;
    wdot[33] += qdot;
  }

  {
    // reaction 197:  C6H12 + H <=> C2H4 + pC4H9
    const amrex::Real k_f =
      8e+15 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[34]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[20] - g_RT[30] + g_RT[34])) *
                           (sc[20] * sc[30]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[20] += qdot;
    wdot[30] += qdot;
    wdot[34] -= qdot;
  }

  {
    // reaction 198:  C6H12 + H <=> C3H6 + nC3H7
    const amrex::Real k_f =
      1.6e+16 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[34]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[25] - g_RT[26] + g_RT[34])) *
                           (sc[25] * sc[26]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[25] += qdot;
    wdot[26] += qdot;
    wdot[34] -= qdot;
  }

  {
    // reaction 199:  C2H4 + pC4H9 <=> PXC6H13
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[20] * sc[30]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[20] + g_RT[30] - g_RT[35])) * (refC) * (sc[35]);
    const amrex::Real qdot = qf - qr;
    wdot[20] -= qdot;
    wdot[30] -= qdot;
    wdot[35] += qdot;
  }

  {
    // reaction 201:  C7H14 + H <=> C2H4 + PXC5H11
    const amrex::Real k_f =
      8e+15 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[36]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[20] - g_RT[33] + g_RT[36])) *
                           (sc[20] * sc[33]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[20] += qdot;
    wdot[33] += qdot;
    wdot[36] -= qdot;
  }

  {
    // reaction 202:  C7H14 + H <=> C3H6 + pC4H9
    const amrex::Real k_f =
      1.6e+16 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[36]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[25] - g_RT[30] + g_RT[36])) *
                           (sc[25] * sc[30]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[25] += qdot;
    wdot[30] += qdot;
    wdot[36] -= qdot;
  }

  {
    // reaction 203:  C2H4 + PXC5H11 <=> PXC7H15
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[20] * sc[33]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[20] + g_RT[33] - g_RT[37])) * (refC) * (sc[37]);
    const amrex::Real qdot = qf - qr;
    wdot[20] -= qdot;
    wdot[33] -= qdot;
    wdot[37] += qdot;
  }

  {
    // reaction 205:  C8H16 + H <=> C2H4 + PXC6H13
    const amrex::Real k_f =
      8e+15 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[38]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[20] - g_RT[35] + g_RT[38])) *
                           (sc[20] * sc[35]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[20] += qdot;
    wdot[35] += qdot;
    wdot[38] -= qdot;
  }

  {
    // reaction 206:  C8H16 + H <=> C3H6 + PXC5H11
    const amrex::Real k_f =
      1.6e+16 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[38]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[25] - g_RT[33] + g_RT[38])) *
                           (sc[25] * sc[33]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[25] += qdot;
    wdot[33] += qdot;
    wdot[38] -= qdot;
  }

  {
    // reaction 207:  C2H4 + PXC6H13 <=> PXC8H17
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[20] * sc[35]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[20] + g_RT[35] - g_RT[39])) * (refC) * (sc[39]);
    const amrex::Real qdot = qf - qr;
    wdot[20] -= qdot;
    wdot[35] -= qdot;
    wdot[39] += qdot;
  }

  {
    // reaction 209:  C9H18 + H <=> C2H4 + PXC7H15
    const amrex::Real k_f =
      8e+15 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[40]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[20] - g_RT[37] + g_RT[40])) *
                           (sc[20] * sc[37]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[20] += qdot;
    wdot[37] += qdot;
    wdot[40] -= qdot;
  }

  {
    // reaction 210:  C9H18 + H <=> C3H6 + PXC6H13
    const amrex::Real k_f =
      1.6e+16 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[40]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[25] - g_RT[35] + g_RT[40])) *
                           (sc[25] * sc[35]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[25] += qdot;
    wdot[35] += qdot;
    wdot[40] -= qdot;
  }

  {
    // reaction 211:  C2H4 + PXC7H15 <=> PXC9H19
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[20] * sc[37]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[20] + g_RT[37] - g_RT[41])) * (refC) * (sc[41]);
    const amrex::Real qdot = qf - qr;
    wdot[20] -= qdot;
    wdot[37] -= qdot;
    wdot[41] += qdot;
  }

  {
    // reaction 213:  C10H20 + H <=> C2H4 + PXC8H17
    const amrex::Real k_f =
      8e+15 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[42]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[20] - g_RT[39] + g_RT[42])) *
                           (sc[20] * sc[39]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[20] += qdot;
    wdot[39] += qdot;
    wdot[42] -= qdot;
  }

  {
    // reaction 214:  C10H20 + H <=> C3H6 + PXC7H15
    const amrex::Real k_f =
      1.6e+16 * exp((-2.39) * logT - (5625.96232369677) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[42]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[1] - g_RT[25] - g_RT[37] + g_RT[42])) *
                           (sc[25] * sc[37]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[25] += qdot;
    wdot[37] += qdot;
    wdot[42] -= qdot;
  }

  {
    // reaction 215:  C2H4 + PXC8H17 <=> PXC10H21
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[20] * sc[39]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[20] + g_RT[39] - g_RT[43])) * (refC) * (sc[43]);
    const amrex::Real qdot = qf - qr;
    wdot[20] -= qdot;
    wdot[39] -= qdot;
    wdot[43] += qdot;
  }

  {
    // reaction 216:  C12H24 <=> C5H9 + PXC7H15
    const amrex::Real k_f = 3.5e+16 * exp(-(35696.6303005228) * invT);
    const amrex::Real qf = k_f * (sc[47]);
    const amrex::Real qr = k_f * exp(-(-g_RT[31] - g_RT[37] + g_RT[47])) *
                           (refCinv) * (sc[31] * sc[37]);
    const amrex::Real qdot = qf - qr;
    wdot[31] += qdot;
    wdot[37] += qdot;
    wdot[47] -= qdot;
  }

  {
    // reaction 217:  C2H4 + PXC10H21 <=> PXC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[20] * sc[43]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[20] + g_RT[43] - g_RT[44])) * (refC) * (sc[44]);
    const amrex::Real qdot = qf - qr;
    wdot[20] -= qdot;
    wdot[43] -= qdot;
    wdot[44] += qdot;
  }

  {
    // reaction 218:  PXC12H25 <=> S3XC12H25
    const amrex::Real k_f =
      3670000000000 * exp((-0.6) * logT - (7246.31998758796) * invT);
    const amrex::Real qf = k_f * (sc[44]);
    const amrex::Real qr = k_f * exp(-(g_RT[44] - g_RT[46])) * (sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[44] -= qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 219:  C3H6 + PXC9H19 <=> SXC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[25] * sc[41]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[25] + g_RT[41] - g_RT[45])) * (refC) * (sc[45]);
    const amrex::Real qdot = qf - qr;
    wdot[25] -= qdot;
    wdot[41] -= qdot;
    wdot[45] += qdot;
  }

  {
    // reaction 220:  C4H81 + PXC8H17 <=> SXC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[29] * sc[39]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[29] + g_RT[39] - g_RT[45])) * (refC) * (sc[45]);
    const amrex::Real qdot = qf - qr;
    wdot[29] -= qdot;
    wdot[39] -= qdot;
    wdot[45] += qdot;
  }

  {
    // reaction 221:  C5H10 + PXC7H15 <=> S3XC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[32] * sc[37]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[32] + g_RT[37] - g_RT[46])) * (refC) * (sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[32] -= qdot;
    wdot[37] -= qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 222:  C10H20 + C2H5 <=> S3XC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[21] * sc[42]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[21] + g_RT[42] - g_RT[46])) * (refC) * (sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[21] -= qdot;
    wdot[42] -= qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 223:  C6H12 + PXC6H13 <=> S3XC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[34] * sc[35]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[34] + g_RT[35] - g_RT[46])) * (refC) * (sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[34] -= qdot;
    wdot[35] -= qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 224:  C9H18 + nC3H7 <=> S3XC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[26] * sc[40]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[26] + g_RT[40] - g_RT[46])) * (refC) * (sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[26] -= qdot;
    wdot[40] -= qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 225:  C7H14 + PXC5H11 <=> S3XC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[33] * sc[36]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[33] + g_RT[36] - g_RT[46])) * (refC) * (sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[33] -= qdot;
    wdot[36] -= qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 226:  C8H16 + pC4H9 <=> S3XC12H25
    const amrex::Real k_f = 300000 * exp(-(3673.48166037445) * invT);
    const amrex::Real qf = k_f * (sc[30] * sc[38]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[30] + g_RT[38] - g_RT[46])) * (refC) * (sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[30] -= qdot;
    wdot[38] -= qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 227:  C2H5 + PXC10H21 <=> NC12H26
    const amrex::Real k_f = 188000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * (sc[21] * sc[43]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[21] + g_RT[43])) * (refC) * (sc[0]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[21] -= qdot;
    wdot[43] -= qdot;
  }

  {
    // reaction 228:  PXC9H19 + nC3H7 <=> NC12H26
    const amrex::Real k_f = 188000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * (sc[26] * sc[41]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[26] + g_RT[41])) * (refC) * (sc[0]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[26] -= qdot;
    wdot[41] -= qdot;
  }

  {
    // reaction 229:  PXC8H17 + pC4H9 <=> NC12H26
    const amrex::Real k_f = 188000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * (sc[30] * sc[39]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[30] + g_RT[39])) * (refC) * (sc[0]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[30] -= qdot;
    wdot[39] -= qdot;
  }

  {
    // reaction 230:  PXC5H11 + PXC7H15 <=> NC12H26
    const amrex::Real k_f = 188000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * (sc[33] * sc[37]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[33] + g_RT[37])) * (refC) * (sc[0]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[33] -= qdot;
    wdot[37] -= qdot;
  }

  {
    // reaction 231:  2 PXC6H13 <=> NC12H26
    const amrex::Real k_f = 188000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * ((sc[35] * sc[35]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + 2.000000 * g_RT[35])) * (refC) * (sc[0]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[35] -= 2.000000 * qdot;
  }

  {
    // reaction 232:  H + NC12H26 <=> H2 + PXC12H25
    const amrex::Real k_f =
      1.3 * exp((2.54) * logT - (3399.73179417669) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[1]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[1] - g_RT[5] - g_RT[44])) * (sc[5] * sc[44]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[44] += qdot;
  }

  {
    // reaction 233:  H + NC12H26 <=> H2 + SXC12H25
    const amrex::Real k_f = 2.6 * exp((2.4) * logT - (2249.8817128129) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[1]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[1] - g_RT[5] - g_RT[45])) * (sc[5] * sc[45]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[45] += qdot;
  }

  {
    // reaction 234:  H + NC12H26 <=> H2 + S3XC12H25
    const amrex::Real k_f = 3.9 * exp((2.4) * logT - (2249.8817128129) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[1]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[1] - g_RT[5] - g_RT[46])) * (sc[5] * sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] -= qdot;
    wdot[5] += qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 235:  NC12H26 + O <=> OH + PXC12H25
    const amrex::Real k_f =
      0.19 * exp((2.68) * logT - (1869.95313013034) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[2]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[2] - g_RT[3] - g_RT[44])) * (sc[3] * sc[44]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[44] += qdot;
  }

  {
    // reaction 236:  NC12H26 + O <=> OH + SXC12H25
    const amrex::Real k_f =
      0.0952 * exp((2.71) * logT - (1059.77429818474) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[2]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[2] - g_RT[3] - g_RT[45])) * (sc[3] * sc[45]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[45] += qdot;
  }

  {
    // reaction 237:  NC12H26 + O <=> OH + S3XC12H25
    const amrex::Real k_f =
      0.1428 * exp((2.71) * logT - (1059.77429818474) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[2]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[2] - g_RT[3] - g_RT[46])) * (sc[3] * sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 238:  NC12H26 + OH <=> H2O + PXC12H25
    const amrex::Real k_f =
      0.0034 * exp((2.66) * logT - (265.195182879087) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[3]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[3] - g_RT[6] - g_RT[44])) * (sc[6] * sc[44]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[44] += qdot;
  }

  {
    // reaction 239:  NC12H26 + OH <=> H2O + SXC12H25
    const amrex::Real k_f =
      0.074 * exp((2.39) * logT - (197.764149661255) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[3]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[3] - g_RT[6] - g_RT[45])) * (sc[6] * sc[45]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[45] += qdot;
  }

  {
    // reaction 240:  NC12H26 + OH <=> H2O + S3XC12H25
    const amrex::Real k_f =
      0.101 * exp((2.39) * logT - (197.764149661255) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[3]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[3] - g_RT[6] - g_RT[46])) * (sc[6] * sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 241:  NC12H26 + O2 <=> HO2 + PXC12H25
    const amrex::Real k_f = 40000000 * exp(-(25628.8247894344) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[4] + g_RT[8] - g_RT[44])) * (sc[4] * sc[44]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[44] += qdot;
  }

  {
    // reaction 242:  NC12H26 + O2 <=> HO2 + SXC12H25
    const amrex::Real k_f = 80000000 * exp(-(23948.0811256466) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[4] + g_RT[8] - g_RT[45])) * (sc[4] * sc[45]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[45] += qdot;
  }

  {
    // reaction 243:  NC12H26 + O2 <=> HO2 + S3XC12H25
    const amrex::Real k_f = 120000000 * exp(-(23948.0811256466) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[4] + g_RT[8] - g_RT[46])) * (sc[4] * sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 244:  HO2 + NC12H26 <=> H2O2 + PXC12H25
    const amrex::Real k_f =
      0.0676 * exp((2.55) * logT - (8298.04281911983) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[4] - g_RT[7] - g_RT[44])) * (sc[7] * sc[44]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[44] += qdot;
  }

  {
    // reaction 245:  HO2 + NC12H26 <=> H2O2 + SXC12H25
    const amrex::Real k_f =
      0.089 * exp((2.6) * logT - (6999.74382134365) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[4] - g_RT[7] - g_RT[45])) * (sc[7] * sc[45]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[45] += qdot;
  }

  {
    // reaction 246:  HO2 + NC12H26 <=> H2O2 + S3XC12H25
    const amrex::Real k_f =
      0.0885 * exp((2.6) * logT - (6999.74382134365) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] + g_RT[4] - g_RT[7] - g_RT[46])) * (sc[7] * sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 247:  CH3 + NC12H26 <=> CH4 + PXC12H25
    const amrex::Real k_f =
      1.81e-06 * exp((3.65) * logT - (3599.50881050116) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[11]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[0] + g_RT[11] - g_RT[12] - g_RT[44])) *
                           (sc[12] * sc[44]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[44] += qdot;
  }

  {
    // reaction 248:  CH3 + NC12H26 <=> CH4 + SXC12H25
    const amrex::Real k_f =
      6e-06 * exp((3.46) * logT - (2757.62732860986) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[11]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[0] + g_RT[11] - g_RT[12] - g_RT[45])) *
                           (sc[12] * sc[45]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[45] += qdot;
  }

  {
    // reaction 249:  CH3 + NC12H26 <=> CH4 + S3XC12H25
    const amrex::Real k_f =
      9e-06 * exp((3.46) * logT - (2757.62732860986) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[11]);
    const amrex::Real qr = k_f *
                           exp(-(g_RT[0] + g_RT[11] - g_RT[12] - g_RT[46])) *
                           (sc[12] * sc[46]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
    wdot[46] += qdot;
  }

  {
    // reaction 250:  O2 + PXC12H25 => C12H25O2
    const amrex::Real k_f = 50000000;
    const amrex::Real qf = k_f * (sc[8] * sc[44]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[8] -= qdot;
    wdot[44] -= qdot;
    wdot[48] += qdot;
  }

  {
    // reaction 251:  C12H25O2 => O2 + PXC12H25
    const amrex::Real k_f = 27500000000000 * exp(-(13788.1366430493) * invT);
    const amrex::Real qf = k_f * (sc[48]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[8] += qdot;
    wdot[44] += qdot;
    wdot[48] -= qdot;
  }

  {
    // reaction 252:  O2 + SXC12H25 => C12H25O2
    const amrex::Real k_f = 50000000;
    const amrex::Real qf = k_f * (sc[8] * sc[45]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[8] -= qdot;
    wdot[45] -= qdot;
    wdot[48] += qdot;
  }

  {
    // reaction 253:  C12H25O2 => O2 + SXC12H25
    const amrex::Real k_f = 27500000000000 * exp(-(13788.1366430493) * invT);
    const amrex::Real qf = k_f * (sc[48]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[8] += qdot;
    wdot[45] += qdot;
    wdot[48] -= qdot;
  }

  {
    // reaction 254:  O2 + S3XC12H25 => C12H25O2
    const amrex::Real k_f = 50000000;
    const amrex::Real qf = k_f * (sc[8] * sc[46]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[8] -= qdot;
    wdot[46] -= qdot;
    wdot[48] += qdot;
  }

  {
    // reaction 255:  C12H25O2 => O2 + S3XC12H25
    const amrex::Real k_f = 27500000000000 * exp(-(13788.1366430493) * invT);
    const amrex::Real qf = k_f * (sc[48]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[8] += qdot;
    wdot[46] += qdot;
    wdot[48] -= qdot;
  }

  {
    // reaction 256:  C12H25O2 => C12OOH
    const amrex::Real k_f = 1510000000000 * exp(-(9561.11665028967) * invT);
    const amrex::Real qf = k_f * (sc[48]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[48] -= qdot;
    wdot[49] += qdot;
  }

  {
    // reaction 257:  C12OOH => C12H25O2
    const amrex::Real k_f = 100000000000 * exp(-(5786.99165675428) * invT);
    const amrex::Real qf = k_f * (sc[49]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[48] += qdot;
    wdot[49] -= qdot;
  }

  {
    // reaction 258:  O2 + PXC12H25 => C12H24 + HO2
    const amrex::Real k_f = 350000 * exp(-(3019.29999482832) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[44]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[44] -= qdot;
    wdot[47] += qdot;
  }

  {
    // reaction 259:  C12H24 + HO2 => O2 + PXC12H25
    const amrex::Real k_f = 316000 * exp(-(9812.72498319203) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[47]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[8] += qdot;
    wdot[44] += qdot;
    wdot[47] -= qdot;
  }

  {
    // reaction 260:  O2 + SXC12H25 => C12H24 + HO2
    const amrex::Real k_f = 350000 * exp(-(3019.29999482832) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[45]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[45] -= qdot;
    wdot[47] += qdot;
  }

  {
    // reaction 261:  C12H24 + HO2 => O2 + SXC12H25
    const amrex::Real k_f = 316000 * exp(-(9812.72498319203) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[47]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[8] += qdot;
    wdot[45] += qdot;
    wdot[47] -= qdot;
  }

  {
    // reaction 262:  O2 + S3XC12H25 => C12H24 + HO2
    const amrex::Real k_f = 350000 * exp(-(3019.29999482832) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[46]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[46] -= qdot;
    wdot[47] += qdot;
  }

  {
    // reaction 263:  C12H24 + HO2 => O2 + S3XC12H25
    const amrex::Real k_f = 316000 * exp(-(9812.72498319203) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[47]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[8] += qdot;
    wdot[46] += qdot;
    wdot[47] -= qdot;
  }

  {
    // reaction 264:  C12OOH + O2 => O2C12H24OOH
    const amrex::Real k_f = 46000;
    const amrex::Real qf = k_f * (sc[8] * sc[49]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[8] -= qdot;
    wdot[49] -= qdot;
    wdot[50] += qdot;
  }

  {
    // reaction 265:  O2C12H24OOH => C12OOH + O2
    const amrex::Real k_f = 25100000000000 * exp(-(13788.1366430493) * invT);
    const amrex::Real qf = k_f * (sc[50]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[8] += qdot;
    wdot[49] += qdot;
    wdot[50] -= qdot;
  }

  {
    // reaction 266:  O2C12H24OOH <=> OC12H23OOH + OH
    const amrex::Real k_f = 89000000000 * exp(-(8554.68331868023) * invT);
    const amrex::Real qf = k_f * (sc[50]);
    const amrex::Real qr = k_f * exp(-(-g_RT[3] + g_RT[50] - g_RT[51])) *
                           (refCinv) * (sc[3] * sc[51]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[50] -= qdot;
    wdot[51] += qdot;
  }

  {
    // reaction 267:  OC12H23OOH => 3 C2H4 + C2H5 + 2 CH2CHO + OH
    const amrex::Real k_f = 1.8e+15 * exp(-(21167.8090470755) * invT);
    const amrex::Real qf = k_f * (sc[51]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[20] += 3.000000 * qdot;
    wdot[21] += qdot;
    wdot[23] += 2.000000 * qdot;
    wdot[51] -= qdot;
  }
}

//  get temperature given internal energy in mass units and mass fracs
__device__ __forceinline__ void
GET_T_GIVEN_EY_CUDA(
  const amrex::Real e,
  volatile const amrex::Real y[],
  volatile amrex::Real scratch[64],
  amrex::Real& t,
  int& ierr,
  const int tidx)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real e1, emin, emax, cv, t1, dt;
  CKUBMS_CUDA(tmin, y, emin, scratch, tidx);
  CKUBMS_CUDA(tmax, y, emax, scratch, tidx);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS_CUDA(tmin, y, cv, scratch, tidx);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS_CUDA(tmax, y, cv, scratch, tidx);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  // int count = 0;
  // for (int i = 0; i < maxiter; ++i) 
  {
    // count = count+1;
    CKUBMS_CUDA(t1, y, e1, scratch, tidx);
    CKCVBS_CUDA(t1, y, cv, scratch, tidx);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      // break;
    }
    t1 += dt;
  }
  // printf("Count: %d \n", count);
  t = t1;
  ierr = 0;
}

// Returns internal energy in mass units (Eq 30.)
__device__ __forceinline__ void
CKUMS_CUDA(const amrex::Real T, volatile amrex::Real ums[NUM_SPECIES], const int tidx)
{
  speciesInternalEnergy_CUDA(ums, T, tidx);

  for (int i = tidx; i < NUM_SPECIES; i += 32) {
    ums[i] *= 8.31446261815324e+07 * T * imw(i);
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
__device__ __forceinline__ void
CKYTCR_CUDA(
  const amrex::Real rho,
  amrex::Real /*T*/,
  volatile const amrex::Real y[NUM_SPECIES],
  volatile amrex::Real c[64],
  const int tidx)
{
  for (int i = tidx; i < NUM_SPECIES; i += 32) {
    c[i] = rho * y[i] * imw(i);
  }
}

// compute the production rate for each species
__device__ __forceinline__ void
CKWC_CUDA(
  const amrex::Real T, volatile amrex::Real C[64], volatile amrex::Real wdot[NUM_SPECIES], const int tidx)
{
  // convert to SI
  for (int i = tidx; i < NUM_SPECIES; i += 32) {
    C[i] *= 1.0e6;
  }
  __syncwarp();

  if (tidx == 0) {
    productionRate_CUDA(wdot, C, T);
  }
  __syncwarp();

  // convert to chemkin units
  for (int i = tidx; i < NUM_SPECIES; i += 32) {
    C[i] *= 1.0e-6;
    wdot[i] *= 1.0e-6;
  }
}

#endif