function [REWS_0,REWS_FF] = LDP_MD(isValid,beamID,lineOfSightWindSpeed,DT,LDP)
% Multi-distance lidar data processing (online, one call per time step) with two outputs.
% Matlab version of LDP_MD.py (development in python/studies/MultiDistanceLDP).
% Estimates the rotor-effective wind speed (REWS) from the line-of-sight
% wind speeds of all range gates. Same inputs as LDP_v3, but for all gates:
%   [v_0L(i_t),v_FF(i_t)] = LDP_MD(isValid(i_t,:),beamID(i_t),lineOfSightWindSpeed(i_t,:),dt,LDP_MD_Parameter);
% v_0L: REWS at the rotor at the current time, unfiltered (discipline 1, smallest SDES)
% v_FF: REWS at the rotor at t + T_lead, adaptive low-pass (discipline 2, feedforward control)
% Reset with: clear LDP_MD
%
% Processing:
% 1. new measurement when beamID changes; only valid measurements are used
% 2. u estimate per beam and gate: v_los / cos(AngleToCenterline)
% 3. advection to the rotor: arrival time t_a = t_m + tau(x), tau from an
%    induction zone model (Troldborg & Meyer Forsting) and the advection
%    speed c_conv*U_adv*shear(z). U_adv is the spatial mean of the induction
%    corrected, held values of all beams and gates (optionally low-pass
%    filtered with T_mean), shear(z) a power law over height per beam and gate
% 4. signal of every beam and gate in rotor time: linear interpolation
%    between two arrivals if the later measurement is already known,
%    otherwise hold (blade ghost free: a missing beam keeps its own
%    information instead of changing the composition of the mean)
% 5. mean over the 4 beams per gate, weighted sum over the gates
% 6. control output: estimate for the rotor at t + T_lead (uses part of the
%    preview) and adaptive first order low-pass with omega_c = k_cutoff*U_LPF
%    (U_LPF: spatial mean, low-pass with T_mean_LPF)

% persistent variables
persistent S
if isempty(S)
    S = Initialize(LDP);
else
    S.n_step = S.n_step + 1;
end
t   = S.n_step*DT;                                          % no accumulation of round-off
N   = LDP.BufferSize;

% 1./2. new measurement: update held values
NewMeasurement = beamID ~= S.PreviousBeamID;
if NewMeasurement
    i_b                     = beamID;
    valid                   = logical(isValid(:)');
    u_new                   = lineOfSightWindSpeed(:)'/cosd(LDP.AngleToCenterline);
    S.u_held(i_b,valid)     = u_new(valid);
    S.PreviousBeamID        = beamID;
end

% spatial mean of the induction corrected held values; optional temporal low-pass (Tustin)
U_input         = mean(MeanOverBeams(S.u_held,LDP.U_init)./S.gain_induction);
if LDP.T_mean > 0
    S.U_adv     = TustinLowpass(S.U_adv,U_input,S.U_input_last,DT/LDP.T_mean);
else
    S.U_adv     = U_input;
end
S.U_LPF         = TustinLowpass(S.U_LPF,U_input,S.U_input_last,DT/LDP.T_mean_LPF);
S.U_input_last  = U_input;

% 3. store the new valid measurements with their arrival time at the rotor
if NewMeasurement && any(valid)
    t_a                         = t + S.tau_unit./(LDP.c_conv*S.U_adv*S.shear(i_b,:));
    t_a                         = max(t_a,S.ta_last(i_b,:));
    for i_g = find(valid)
        S.n_written(i_b,i_g)    = S.n_written(i_b,i_g) + 1;
        k                       = mod(S.n_written(i_b,i_g)-1,N) + 1;  % slot in ring buffer
        S.buf_ta(i_b,i_g,k)     = t_a(i_g);
        S.buf_u(i_b,i_g,k)      = u_new(i_g);
        S.ta_last(i_b,i_g)      = t_a(i_g);
    end
end

% 4./5. REWS in rotor time for both query times
[REWS_0,S]      = RotorTimeSignal(S,1,t,LDP);               % discipline 1: rotor, now
[REWS_lead,S]   = RotorTimeSignal(S,2,t+LDP.T_lead,LDP);    % discipline 2: rotor at t + T_lead

% 6. adaptive low-pass (Tustin) of the control output: omega_c = k_cutoff * U_LPF
if isempty(S.LPF_InputLast)
    S.LPF_InputLast     = REWS_lead;
    S.LPF_OutputLast    = REWS_lead;
end
REWS_FF             = TustinLowpass(S.LPF_OutputLast,REWS_lead,S.LPF_InputLast,LDP.k_cutoff*S.U_LPF*DT);
S.LPF_InputLast     = REWS_lead;
S.LPF_OutputLast    = REWS_FF;

end


function [REWS,S] = RotorTimeSignal(S,i_q,t_q,LDP)
% advance to the last measurement arrived at t_q and interpolate (else hold)
% for every beam and gate; n_arrived(:,:,i_q): number of arrived measurements (0 = none)
N       = LDP.BufferSize;
nb      = LDP.NumberOfBeams;
ng      = S.n_gates;
Lin     = reshape(1:nb*ng,nb,ng);                           % linear index of (beam, gate)
n_arr   = S.n_arrived(:,:,i_q);
while true
    nxt     = n_arr + 1;
    has_nxt = nxt <= S.n_written;
    ta_nxt  = S.buf_ta(Lin + (mod(nxt-1,N))*nb*ng);
    advance = has_nxt & (ta_nxt <= t_q);
    if ~any(advance(:))
        break
    end
    n_arr(advance) = nxt(advance);
end
S.n_arrived(:,:,i_q) = n_arr;

arrived = n_arr >= 1;
idx0    = Lin + (mod(max(n_arr,1)-1,N))*nb*ng;
idx1    = Lin + (mod(n_arr,N))*nb*ng;
ta0     = S.buf_ta(idx0);
u0      = S.buf_u(idx0);
ta1     = S.buf_ta(idx1);
u1      = S.buf_u(idx1);
known   = arrived & (n_arr + 1 <= S.n_written);             % next measurement already taken
w       = zeros(nb,ng);
w(known)= (t_q - ta0(known))./max(ta1(known)-ta0(known),1e-9);
u_rt    = u0 + min(max(w,0),1).*(u1-u0);
u_rt(~arrived) = NaN;

% 5. mean over beams, weighted sum over gates
REWS    = S.weights*MeanOverBeams(u_rt,LDP.U_init)';
end


function S = Initialize(LDP)
% Initialize the persistent state: geometry, induction model, shear and buffers

d                   = LDP.GateDistances(:)';
x                   = d + LDP.X_Lidar;                      % [m]   upstream distance of the gates from the rotor
R                   = LDP.RotorDiameter/2;
a                   = LDP.InductionFactor;
nb                  = LDP.NumberOfBeams;
ng                  = numel(x);
N                   = LDP.BufferSize;

S.n_gates           = ng;
S.weights           = LDP.Weights(:)'/sum(LDP.Weights);
S.gain_induction    = 1 - a*InductionShape(x/R);
% travel time for an advection speed of 1 m/s: int_0^x dx' / (1 - a f(x'))
s                   = linspace(0,1,400);
S.tau_unit          = zeros(1,ng);
for i_g = 1:ng
    xx              = x(i_g)*s;
    S.tau_unit(i_g) = trapz(xx,1./(1-a*InductionShape(xx/R)));
end
% shear over height: advection speed factor per beam and gate (power law, 1 at the lidar height)
H                   = LDP.HubHeight;
z                   = LDP.Z_Lidar + LDP.BeamSignZ(:)*d*tand(LDP.AngleVertical);  % [m] height relative to hub
S.shear             = ((H+z)/(H+LDP.Z_Lidar)).^LDP.ShearExponent;

S.n_step            = 0;
S.PreviousBeamID    = -1;                                   % force update on first call
S.u_held            = NaN(nb,ng);                           % last valid u per beam and gate
S.U_input_last      = LDP.U_init;
S.U_adv             = LDP.U_init;                           % advection speed (spatial mean)
S.U_LPF             = LDP.U_init;                           % speed for the low-pass corner frequency
% ring buffer per beam and gate of valid measurements: arrival time and value
S.buf_ta            = zeros(nb,ng,N);
S.buf_u             = zeros(nb,ng,N);
S.n_written         = zeros(nb,ng);                         % number of stored measurements
S.n_arrived         = zeros(nb,ng,2);                       % number of arrived measurements per query time
S.ta_last           = -Inf(nb,ng);                          % last arrival time (monotonic)
S.LPF_InputLast     = [];                                   % state of the output low-pass
S.LPF_OutputLast    = [];
end


function y = TustinLowpass(y_last,x,x_last,wdt)
% first order low-pass (Tustin), wdt = omega_c * DT
y = ((2-wdt)*y_last + wdt*(x+x_last))/(2+wdt);
end


function f = InductionShape(xi)
% induction zone shape: U(x)/U_inf = 1 - a*f(x/R), f(0) = 1
f = 1 - xi./sqrt(1+xi.^2);
end


function u_gate = MeanOverBeams(u,U_init)
% mean over the available (non-NaN) beams per gate; U_init for gates without any value (start-up)
ok          = ~isnan(u);
n           = sum(ok,1);
u(~ok)      = 0;
u_gate      = sum(u,1)./max(n,1);
u_gate(n==0)= U_init;
end
