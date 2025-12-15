close all;
clc; clear;

% FMCW chirp (complex baseband) example
fc = 77e9;      % carrier (physical), used for parameter calc only
B  = 2e9;       % bandwidth (Hz)
Tc = 40e-6;     % chirp duration (s)

% NOTE: for real full-band simulation you'd need fs >= 2*B (>=4 GHz).
% For visualization/demo we pick a lower fs and interpret signal as baseband envelope.
fs = 4e9;                % sampling rate for baseband simulation (demo)
dt = 1/fs;
t = 0:dt:Tc-dt;           % time vector for one chirp
S = B / Tc;               % chirp slope (Hz/s)

% Choose baseband start frequency f0 = 0 (sweeps 0 -> B). Complex baseband chirp:
phi = 2*pi*( 0.*t + 0.5 * S .* t.^2 );   % instantaneous phase (rad)
s_bb = exp(1j*phi);                      % complex baseband chirp

% Instantaneous frequency (Hz) of baseband chirp: fint(t) = S*t + f0
f_inst = S .* t;                         

% plot time-domain real part and instantaneous frequency
figure;
subplot(2,1,1);
plot(t*1e6, real(s_bb));
xlabel('Time (μs)'); ylabel('Re\{s_{bb}\}'); title('Complex baseband chirp (real part)');
grid on;

subplot(2,1,2);
plot(t*1e6, f_inst/1e3);
xlabel('Time (μs)'); ylabel('Inst. freq (kHz)'); title('Instantaneous frequency (baseband)');
grid on;

% Spectrogram (f-t) for visualization
figure;
win_len = 128;                         % 窗长度（样点），短窗提升时间分辨
window = hamming(win_len);
noverlap = round(0.75*win_len);
nfft = 2048;
spectrogram(s_bb, window, noverlap, nfft, fs, 'centered');
title('Spectrogram of complex baseband chirp');
colormap jet;

R = 0.10;
tau = 2*R/3e8;
fB = S * tau;
disp(['beat freq f_B = ', num2str(fB), ' Hz']);
