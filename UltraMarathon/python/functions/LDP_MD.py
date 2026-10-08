import numpy as np


class LDP_MD:
    """Multi-distance lidar data processing (online, one call per time step).

    Estimates the rotor-effective wind speed (REWS) at the rotor at the current time from the
    line-of-sight wind speeds of all range gates. Same inputs as LDP_v3, but for all gates:
        v_0L = ldp.step(isValid[i_t, :], beamID[i_t], lineOfSightWindSpeed[i_t, :], dt)

    Processing (see studies/MultiDistanceLDP, steps s01-s05):
    1. new measurement when beamID changes; only valid measurements are used
    2. u estimate per beam and gate: v_los / cos(AngleToCenterline)
    3. advection to the rotor: arrival time t_a = t_m + tau(x), tau from an induction zone model
       (Troldborg & Meyer Forsting) and the advection speed c_conv * U_inf; U_inf is estimated
       online with a first order low-pass of the induction corrected, held gate means
    4. signal of every beam and gate in rotor time: linear interpolation between two arrivals if the
       later measurement is already known, otherwise hold (blade ghost free: a missing beam keeps
       its own information instead of changing the composition of the mean)
    5. mean over the 4 beams per gate, weighted sum over the gates
    6. optional (for control): estimate for the rotor at t + T_lead (uses part of the preview) and
       adaptive first order low-pass with omega_c = k_cutoff * U_inf
    """

    def __init__(self, LDP):
        self.p              = LDP
        x                   = np.asarray(LDP['GateDistances'], float) + LDP['X_Lidar']  # upstream of rotor [m]
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

    def reset(self):
        nb, ng, N           = self.n_beams, self.n_gates, self.p['BufferSize']
        self.n_step         = 0
        self.first_call     = True
        self.PreviousBeamID = -1
        self.u_held         = np.full((nb, ng), np.nan)             # last valid u per beam and gate
        self.U_inf          = self.p['U_init']                      # online free stream estimate
        self.U_input_last   = self.p['U_init']
        # ring buffer per beam and gate of valid measurements: measurement time, arrival time, value
        self.buf_tm         = np.zeros((nb, ng, N))
        self.buf_ta         = np.zeros((nb, ng, N))
        self.buf_u          = np.zeros((nb, ng, N))
        self.n_written      = np.zeros((nb, ng), int)               # number of stored measurements
        self.i_arrived      = np.full((nb, ng), -1)                 # index of last arrived measurement
        self.ta_last        = np.full((nb, ng), -np.inf)            # last arrival time (monotonic)
        self.LPF_last       = None                                  # state of the output low-pass

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

        # online free stream estimate: Tustin first order low-pass, time constant T_mean
        u_gate_held = self._mean_over_beams(self.u_held)
        U_input     = np.mean(u_gate_held / self.gain_induction)
        wdt         = dt / self.p['T_mean']
        self.U_inf  = ((2 - wdt) * self.U_inf + wdt * (U_input + self.U_input_last)) / (2 + wdt)
        self.U_input_last = U_input

        # 3. store the new valid measurements with their arrival time at the rotor
        if new_measurement and np.any(valid):
            t_a                     = t + self.tau_unit / (self.p['c_conv'] * self.U_inf)
            t_a                     = np.maximum(t_a, self.ta_last[i_b])
            g                       = np.flatnonzero(valid)
            k                       = self.n_written[i_b, g] % N
            self.buf_tm[i_b, g, k]  = t
            self.buf_ta[i_b, g, k]  = t_a[g]
            self.buf_u[i_b, g, k]   = u_new[g]
            self.ta_last[i_b, g]    = t_a[g]
            self.n_written[i_b, g]  += 1

        # 4. advance to the last measurement arrived at the rotor at t + T_lead (all beams and gates at once)
        t_q = t + self.p.get('T_lead', 0.0)
        while True:
            nxt     = self.i_arrived + 1
            has_nxt = nxt < self.n_written
            ta_nxt  = np.take_along_axis(self.buf_ta, (nxt % N)[..., None], axis=2)[..., 0]
            advance = has_nxt & (ta_nxt <= t_q)
            if not np.any(advance):
                break
            self.i_arrived = np.where(advance, nxt, self.i_arrived)

        # interpolate between last arrived and next (already measured) value, else hold
        arrived = self.i_arrived >= 0
        k0      = (np.maximum(self.i_arrived, 0) % N)[..., None]
        k1      = ((self.i_arrived + 1) % N)[..., None]
        ta0     = np.take_along_axis(self.buf_ta, k0, axis=2)[..., 0]
        u0      = np.take_along_axis(self.buf_u, k0, axis=2)[..., 0]
        ta1     = np.take_along_axis(self.buf_ta, k1, axis=2)[..., 0]
        u1      = np.take_along_axis(self.buf_u, k1, axis=2)[..., 0]
        known   = arrived & (self.i_arrived + 1 < self.n_written)
        w       = np.where(known, (t_q - ta0) / np.maximum(ta1 - ta0, 1e-9), 0.0)
        u_rt    = np.where(arrived, u0 + np.clip(w, 0, 1) * (u1 - u0), np.nan)

        # 5. mean over beams, weighted sum over gates
        u_gate = self._mean_over_beams(u_rt)
        REWS   = float(self.weights @ u_gate)

        # 6. optional adaptive low-pass (Tustin): corner frequency omega_c = k_cutoff * U_inf
        if self.p.get('FlagLPF', 0):
            if self.LPF_last is None:
                self.LPF_last = (REWS, REWS)                        # (input, output) of last step
            wdt         = self.p['k_cutoff'] * self.U_inf * dt
            x_last, y_last = self.LPF_last
            REWS_f      = ((2 - wdt) * y_last + wdt * (REWS + x_last)) / (2 + wdt)
            self.LPF_last = (REWS, REWS_f)
            return REWS_f
        return REWS
