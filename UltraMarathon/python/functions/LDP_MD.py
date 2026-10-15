import numpy as np


class LDP_MD:
    """Multi-distance lidar data processing (online, one call per time step) with two outputs.

    Estimates the rotor-effective wind speed (REWS) from the line-of-sight wind speeds of all range gates.
    Same inputs as LDP_v3, but for all gates:
        v_0L, v_FF = ldp.step(isValid[i_t, :], beamID[i_t], lineOfSightWindSpeed[i_t, :], dt)
    v_0L: REWS at the rotor at the current time, unfiltered (discipline 1, smallest SDES)
    v_FF: REWS at the rotor at t + T_lead, adaptive low-pass (discipline 2, feedforward control)

    Processing (see studies/MultiDistanceLDP, steps s01-s14):
    1. new measurement when beamID changes; only valid measurements are used
    2. u estimate per beam and gate: v_los / cos(AngleToCenterline)
    3. advection to the rotor: arrival time t_a = t_m + tau(x), tau from an induction zone model
       (Troldborg & Meyer Forsting) and the advection speed c_conv * U_adv * shear(z). U_adv is the
       spatial mean of the induction corrected, held values of all beams and gates (optionally low-pass
       filtered with T_mean), shear(z) a power law over height per beam and gate
    4. signal of every beam and gate in rotor time: linear interpolation between two arrivals if the
       later measurement is already known, otherwise hold (blade ghost free: a missing beam keeps
       its own information instead of changing the composition of the mean)
    5. mean over the 4 beams per gate, weighted sum over the gates
    6. control output: estimate for the rotor at t + T_lead (uses part of the preview) and adaptive first
       order low-pass with omega_c = k_cutoff * U_LPF (U_LPF: spatial mean, low-pass with T_mean_LPF)
    """

    def __init__(self, LDP):
        self.p              = LDP
        d                   = np.asarray(LDP['GateDistances'], float)
        x                   = d + LDP['X_Lidar']                    # upstream distance from rotor [m]
        self.n_beams        = LDP['NumberOfBeams']
        self.n_gates        = len(x)
        self.weights        = np.asarray(LDP['Weights'], float) / np.sum(LDP['Weights'])
        self.cos_angle      = np.cos(np.deg2rad(LDP['AngleToCenterline']))
        R                   = LDP['RotorDiameter'] / 2
        a                   = LDP['InductionFactor']
        self.gain_induction = 1 - a * self._induction_shape(x / R)
        # travel time for an advection speed of 1 m/s: int_0^x dx' / (1 - a f(x'))
        s                   = np.linspace(0, 1, 400)
        xx                  = x[:, None] * s
        self.tau_unit       = np.trapezoid(1 / (1 - a * self._induction_shape(xx / R)), xx, axis=1)
        # shear over height: advection speed factor per beam and gate (power law, 1 at the lidar height)
        H                   = LDP['HubHeight']
        z                   = LDP['Z_Lidar'] + np.asarray(LDP['BeamSignZ'], float)[:, None] \
            * d[None, :] * np.tan(np.deg2rad(LDP['AngleVertical']))  # height relative to hub [m]
        self.shear          = ((H + z) / (H + LDP['Z_Lidar'])) ** LDP['ShearExponent']
        self.T_leads        = np.array([0.0, LDP['T_lead']])         # query times: v_0L, v_FF
        self.reset()

    @staticmethod
    def _induction_shape(xi):
        return 1 - xi / np.sqrt(1 + xi ** 2)

    def _mean_over_beams(self, u):
        """Mean over the available (non-NaN) beams per gate; U_init for gates without any value (start-up)."""
        ok  = ~np.isnan(u)
        n   = ok.sum(axis=0)
        s   = np.where(ok, u, 0.0).sum(axis=0)
        return np.where(n > 0, s / np.maximum(n, 1), self.p['U_init'])

    @staticmethod
    def _tustin_lowpass(y_last, x, x_last, wdt):
        return ((2 - wdt) * y_last + wdt * (x + x_last)) / (2 + wdt)

    def reset(self):
        nb, ng, N           = self.n_beams, self.n_gates, self.p['BufferSize']
        self.n_step         = 0
        self.first_call     = True
        self.PreviousBeamID = -1
        self.u_held         = np.full((nb, ng), np.nan)             # last valid u per beam and gate
        self.U_input_last   = self.p['U_init']
        self.U_adv          = self.p['U_init']                      # advection speed (spatial mean)
        self.U_LPF          = self.p['U_init']                      # speed for the low-pass corner frequency
        # ring buffer per beam and gate of valid measurements: arrival time and value
        self.buf_ta         = np.zeros((nb, ng, N))
        self.buf_u          = np.zeros((nb, ng, N))
        self.n_written      = np.zeros((nb, ng), int)               # number of stored measurements
        self.i_arrived      = np.full((2, nb, ng), -1)              # last arrived measurement per query time
        self.ta_last        = np.full((nb, ng), -np.inf)            # last arrival time (monotonic)
        self.LPF_last       = None                                  # state of the output low-pass

    def _rotor_time_signal(self, i_q, t_q):
        """Advance to the last measurement arrived at t_q and interpolate (else hold) for every beam and gate."""
        N = self.p['BufferSize']
        while True:
            nxt     = self.i_arrived[i_q] + 1
            has_nxt = nxt < self.n_written
            ta_nxt  = np.take_along_axis(self.buf_ta, (nxt % N)[..., None], axis=2)[..., 0]
            advance = has_nxt & (ta_nxt <= t_q)
            if not np.any(advance):
                break
            self.i_arrived[i_q] = np.where(advance, nxt, self.i_arrived[i_q])
        i_a     = self.i_arrived[i_q]
        arrived = i_a >= 0
        k0      = (np.maximum(i_a, 0) % N)[..., None]
        k1      = ((i_a + 1) % N)[..., None]
        ta0     = np.take_along_axis(self.buf_ta, k0, axis=2)[..., 0]
        u0      = np.take_along_axis(self.buf_u, k0, axis=2)[..., 0]
        ta1     = np.take_along_axis(self.buf_ta, k1, axis=2)[..., 0]
        u1      = np.take_along_axis(self.buf_u, k1, axis=2)[..., 0]
        known   = arrived & (i_a + 1 < self.n_written)               # next measurement already taken
        w       = np.where(known, (t_q - ta0) / np.maximum(ta1 - ta0, 1e-9), 0.0)
        u_rt    = np.where(arrived, u0 + np.clip(w, 0, 1) * (u1 - u0), np.nan)
        return float(self.weights @ self._mean_over_beams(u_rt))     # 5. beams -> gates -> REWS

    def step(self, isValid, beamID, lineOfSightWindSpeed, dt):
        if self.first_call:
            self.first_call = False
        else:
            self.n_step += 1
        t = self.n_step * dt                                        # no accumulation of round-off
        N = self.p['BufferSize']

        # 1./2. new measurement: update held values
        new_measurement = beamID != self.PreviousBeamID
        if new_measurement:
            i_b         = int(beamID) - 1
            valid       = np.asarray(isValid, bool)
            u_new       = np.asarray(lineOfSightWindSpeed, float) / self.cos_angle
            self.u_held[i_b, valid] = u_new[valid]
            self.PreviousBeamID = beamID

        # spatial mean of the induction corrected held values; optional temporal low-pass (Tustin)
        U_input     = np.mean(self._mean_over_beams(self.u_held) / self.gain_induction)
        if self.p['T_mean'] > 0:
            self.U_adv = self._tustin_lowpass(self.U_adv, U_input, self.U_input_last, dt / self.p['T_mean'])
        else:
            self.U_adv = U_input
        self.U_LPF  = self._tustin_lowpass(self.U_LPF, U_input, self.U_input_last, dt / self.p['T_mean_LPF'])
        self.U_input_last = U_input

        # 3. store the new valid measurements with their arrival time at the rotor
        if new_measurement and np.any(valid):
            t_a                     = t + self.tau_unit / (self.p['c_conv'] * self.U_adv * self.shear[i_b])
            t_a                     = np.maximum(t_a, self.ta_last[i_b])
            g                       = np.flatnonzero(valid)
            k                       = self.n_written[i_b, g] % N
            self.buf_ta[i_b, g, k]  = t_a[g]
            self.buf_u[i_b, g, k]   = u_new[g]
            self.ta_last[i_b, g]    = t_a[g]
            self.n_written[i_b, g]  += 1

        # 4./5. REWS in rotor time for both query times
        REWS_0      = self._rotor_time_signal(0, t + self.T_leads[0])   # discipline 1: rotor, now
        REWS_lead   = self._rotor_time_signal(1, t + self.T_leads[1])   # discipline 2: rotor at t + T_lead

        # 6. adaptive low-pass (Tustin) of the control output: omega_c = k_cutoff * U_LPF
        if self.LPF_last is None:
            self.LPF_last = (REWS_lead, REWS_lead)                  # (input, output) of last step
        x_last, y_last = self.LPF_last
        REWS_FF     = self._tustin_lowpass(y_last, REWS_lead, x_last, self.p['k_cutoff'] * self.U_LPF * dt)
        self.LPF_last = (REWS_lead, REWS_FF)
        return REWS_0, REWS_FF
