close all; clc; clear;

% ----------------------------
% Radar & signal parameters
% ----------------------------
c = 3e8;
fc = 77e9;
B  = 2e9;          % bandwidth (Hz)
Tc = 40e-6;        % chirp duration (s)
fs = 4e9;          % baseband sampling rate (Hz)
S  = B / Tc;       % slope (Hz/s)
f0 = 0;            % baseband start frequency
SNR_dB = 30;       % target SNR for noise injection (dB)

% Targets: two point targets with ranges & velocities
R_targets = [10, 14];       % meters
v_targets = [2, -3];        % m/s (positive away, negative towards)
sigma_reflect = [0.8, 0.6]; % reflectivities

% Burst (slow-time) parameters
Nchirp = 64;                % number of chirps in a burst (Doppler FFT size)
T_idle = 20e-6;             % inter-chirp idle time (s); PRI = Tc + T_idle
PRI = Tc + T_idle;          % pulse repetition interval
lambda = c / fc;

% FFT sizes
N_range_fft   = 2^12;       % range FFT size (>= fast-time samples)
N_doppler_fft = Nchirp;     % Doppler FFT size

% ----------------------------
% Derived timing vectors
% ----------------------------
dt = 1/fs;
Ns = round(Tc * fs);                % samples per chirp
t_fast = (0:Ns-1) * dt;             % fast-time within a chirp
t_slow = (0:Nchirp-1) * PRI;        % slow-time across chirps

% Baseband TX chirp (complex LFM)
phi_tx = 2*pi*( f0.*t_fast + 0.5*S.*t_fast.^2 );
s_tx = exp(1j*phi_tx);              % size: [1 x Ns]

% ----------------------------
% Simulate received signal over a burst
% Model: sum over targets, each delayed and Doppler shifted across slow-time
% ----------------------------
s_rx_burst = zeros(Nchirp, Ns);     % [slow x fast]

for m = 1:Nchirp
    % For each chirp, build the RX signal in fast-time
    s_rx = zeros(1, Ns);

    for k = 1:numel(R_targets)
        Rk = R_targets(k);
        vk = v_targets(k);
        % Range update with radial velocity during slow-time (monostatic)
        R_m = Rk + vk * t_slow(m);

        % Round-trip delay
        tau = 2 * R_m / c;
        delay_samples = tau * fs;

        % Amplitude scaling by 1/R^2 with reflectivity
        alpha_k = sigma_reflect(k) * (1 / max(R_m, 1e-3)^2);

        % Fractional delay via linear interpolation
        n = 0:Ns-1;
        idx = n - delay_samples;  % desired index
        % Linear interpolation with bounds
        i0 = floor(idx);
        frac = idx - i0;

        valid = (i0 >= 1) & (i0+1 <= Ns);
        val = zeros(size(idx));
        % Use zeros outside bounds to model no return before window
        i0c = i0(valid);
        fracc = frac(valid);
        val(valid) = (1-fracc).*s_tx(i0c) + fracc.*s_tx(i0c+1);

        s_rx = s_rx + alpha_k * val;
    end

    % Add receiver noise to this chirp
    rx_signal_power = mean(abs(s_rx).^2);
    if rx_signal_power <= 0
        rx_signal_power = 1e-12;
    end
    SNR_linear = 10^(SNR_dB/10);
    noise_power_rx = rx_signal_power / SNR_linear;
    noise_rx = sqrt(noise_power_rx/2) * (randn(size(s_rx)) + 1j*randn(size(s_rx)));

    s_rx_burst(m,:) = s_rx + noise_rx;
end

% ----------------------------
% Dechirp (mix) and windowing
% ----------------------------
% Beat signal per chirp: s_tx .* conj(s_rx)
beat_burst = s_tx(ones(Nchirp,1),:) .* conj(s_rx_burst);

% Windowing in fast-time (range) and slow-time (Doppler)
w_range = hann(Ns).';
w_doppler = hann(Nchirp);
beat_win = (w_doppler * w_range) .* beat_burst;  % separable 2D window

% ----------------------------
% Range FFT (fast-time)
% ----------------------------
% Fast-time sizes
Ns = round(Tc * fs);             % should be 160000
N_range_fft = 2^18;              % 262144 >= Ns; good resolution (df ~ 15.26 kHz)

% Range FFT (use Ns samples, zero-pad up to N_range_fft)
RFFT = fft(beat_win, N_range_fft, 2);    % [slow x rangeFFT]
N_half = floor(N_range_fft/2);
RFFT_pos = RFFT(:, 1:N_half);

% Frequency and range axes
freq_range = (0:N_half-1) * (fs / N_range_fft);
R_bins = (c * freq_range) / (2 * S);

% Optional: limit display to a practical range window
max_display_R = 50;    % meters
range_mask = R_bins <= max_display_R;
RFFT_pos = RFFT_pos(:, range_mask);
R_bins = R_bins(range_mask);


% ----------------------------
% Doppler FFT (slow-time)
% ----------------------------
RDM = fftshift(fft(RFFT_pos, N_doppler_fft, 1), 1);    % [doppler x range]
% Convert to power map
RDM_pow = abs(RDM).^2;

% Doppler axis (Hz) ~ PRF-based; for FMCW dechirped Doppler sign consistent
PRF = 1/PRI;
fd_axis = linspace(-PRF/2, PRF/2, N_doppler_fft);      % Doppler freq (slow-time)
v_axis = (lambda/2) * fd_axis;                         % radial velocity (m/s)

% ----------------------------
% 2D CA-CFAR (range–Doppler)
% ----------------------------
% CFAR parameters
Pfa = 1e-5;             % desired false alarm rate
Ng_r = 3;               % guard cells (range) on EACH side
Ng_d = 2;               % guard cells (doppler) on EACH side
Nr_ref = 8;            % reference cells (range) per side
Nd_ref = 4;             % reference cells (doppler) per side

% Total reference cells
Nref = (2*Nr_ref + 2*Nd_ref + 4*Nr_ref*Nd_ref) - (2*Ng_r + 2*Ng_d + 4*Ng_r*Ng_d);
% Easier: build mask explicitly, then count. We'll do explicit mask.

[Nd, Nr] = size(RDM_pow);
CFAR_mask = false(2*(Nd_ref+Ng_d)+1, 2*(Nr_ref+Ng_r)+1);  % window center included
center_d = Nd_ref+Ng_d+1;
center_r = Nr_ref+Ng_r+1;
% Reference region: exclude guard band around center
for di = 1:size(CFAR_mask,1)
    for ri = 1:size(CFAR_mask,2)
        dd = abs(di - center_d);
        rr = abs(ri - center_r);
        if (dd == 0 && rr == 0)
            CFAR_mask(di,ri) = false; % CUT
        elseif (dd <= Ng_d) && (rr <= Ng_r)
            CFAR_mask(di,ri) = false; % guard cells
        else
            CFAR_mask(di,ri) = true;  % reference cells
        end
    end
end
Nref = nnz(CFAR_mask);

% Alpha from Pfa & Nref (exponential noise assumption)
alpha = Nref * (Pfa^(-1/Nref) - 1);

% CFAR sliding window
detections = false(Nd, Nr);
threshold_map = zeros(Nd, Nr);
noise_est_map = zeros(Nd, Nr);

half_win_d = Nd_ref + Ng_d;
half_win_r = Nr_ref + Ng_r;

for d = 1:Nd
    d_min = max(1, d - half_win_d);
    d_max = min(Nd, d + half_win_d);
    for r = 1:Nr
        r_min = max(1, r - half_win_r);
        r_max = min(Nr, r + half_win_r);

        % Extract local window
        W = RDM_pow(d_min:d_max, r_min:r_max);
        % Build mask aligned to window size
        md = size(W,1);
        mr = size(W,2);
        % Center index within window
        d_c = min(center_d, md);
        r_c = min(center_r, mr);

        % Mask resize/pad (handle edges)
        mask_local = false(md, mr);
        % Compute offset ranges in original mask coordinates
        d0 = center_d - (d - d_min);
        r0 = center_r - (r - r_min);
        for di = 1:md
            for ri = 1:mr
                mi = d0 - center_d + di;
                mj = r0 - center_r + ri;
                if mi >= 1 && mi <= size(CFAR_mask,1) && mj >= 1 && mj <= size(CFAR_mask,2)
                    mask_local(di,ri) = CFAR_mask(mi,mj);
                else
                    mask_local(di,ri) = false;
                end
            end
        end

        % Reference noise estimate (mean of reference cells)
        ref_vals = W(mask_local);
        if isempty(ref_vals)
            noise_est = 0;
        else
            noise_est = mean(ref_vals);
        end
        thr = alpha * noise_est;

        threshold_map(d,r) = thr;
        noise_est_map(d,r) = noise_est;

        CUT = RDM_pow(d,r);
        detections(d,r) = (CUT > thr);
    end
end

% ----------------------------
% Peak extraction & labeling
% ----------------------------
% Optional: cluster detections to peak list
BW = bwlabel(detections, 4);
stats = regionprops(BW, RDM_pow, 'MaxIntensity', 'PixelIdxList', 'Centroid');
peak_list = [];
for i = 1:numel(stats)
    [~, imax] = max(RDM_pow(stats(i).PixelIdxList));
    idx = stats(i).PixelIdxList(imax);
    [d_idx, r_idx] = ind2sub([Nd, Nr], idx);
    peak_list = [peak_list; r_idx, d_idx, RDM_pow(d_idx, r_idx)]; %#ok<AGROW>
end
% Sort peaks by power
if ~isempty(peak_list)
    [~, ord] = sort(peak_list(:,3), 'descend');
    peak_list = peak_list(ord,:);
end

% Convert peak indices to physical quantities
ranges_detected = R_bins(peak_list(:,1));
vels_detected   = v_axis(peak_list(:,2));

% ----------------------------
% Display results
% ----------------------------
fprintf('CFAR: Pfa = %.1e, Nref = %d, alpha = %.3f\n', Pfa, Nref, alpha);

% Show top detections (up to 10)
Nshow = min(10, size(peak_list,1));
for i = 1:Nshow
    fprintf('Detection #%d: Range = %.3f m, Velocity = %.3f m/s, Power = %.3e\n', ...
        i, ranges_detected(i), vels_detected(i), peak_list(i,3));
end

% ----------------------------
% Plots
% ----------------------------
figure('Name','Range-Doppler map and CFAR','NumberTitle','off','Position',[200 200 1000 800]);


% 假设 ranges_detected, vels_detected 已经得到
% 只取前两个最强目标
Nshow = min(2, numel(ranges_detected));

subplot(2,2,1); hold on;
imagesc(R_bins, v_axis, 10*log10(RDM_pow + eps));
axis xy; colormap jet; colorbar;
xlabel('Range (m)'); ylabel('Velocity (m/s)');
title('Range-Doppler power (dB)');
for i = 1:Nshow
    plot(ranges_detected(i), vels_detected(i), 'm^', 'MarkerSize',3,'LineWidth',1.5);
    text(ranges_detected(i)+0.2, vels_detected(i)+(i-1)*1.5, ...
        sprintf('(%.2f m, %.2f m/s)', ranges_detected(i), vels_detected(i)), ...
        'Color','w','FontSize',10,'FontWeight','bold');
end
hold off;

subplot(2,2,2); hold on;
imagesc(R_bins, v_axis, 10*log10(threshold_map + eps));
% 自动获取最大值并压缩动态范围（例如显示主瓣附近 40 dB）
caxis([max(10*log10(RDM_pow(:)))-40, max(10*log10(RDM_pow(:)))]);


axis xy; colormap jet; colorbar;
xlabel('Range (m)'); ylabel('Velocity (m/s)');
title('CFAR threshold (dB)');

for i = 1:Nshow
    plot(ranges_detected(i), vels_detected(i), 'm^', 'MarkerSize',3,'LineWidth',1.5);
    text(ranges_detected(i)+0.2, vels_detected(i)+(i-1)*1.5, ...
        sprintf('(%.2f m, %.2f m/s)', ranges_detected(i), vels_detected(i)), ...
        'Color','w','FontSize',10,'FontWeight','bold');
end
hold off;

subplot(2,2,3); hold on;
imagesc(R_bins, v_axis, detections);
axis xy; colormap gray; colorbar;
xlabel('Range (m)'); ylabel('Velocity (m/s)');
title('CFAR detections (binary)');

for i = 1:Nshow
    plot(ranges_detected(i), vels_detected(i), 'm^', 'MarkerSize',3,'LineWidth',1.5);
    text(ranges_detected(i)+0.2, vels_detected(i)+(i-1)*1.5, ...
        sprintf('(%.2f m, %.2f m/s)', ranges_detected(i), vels_detected(i)), ...
        'Color','w','FontSize',10,'FontWeight','bold');
end
hold off;

subplot(2,2,4); hold on;
imagesc(R_bins, v_axis, 10*log10(RDM_pow + eps));
axis xy; colormap jet; colorbar;
xlabel('Range (m)'); ylabel('Velocity (m/s)');
title('Detections overlaid on RDM');
caxis([max(10*log10(RDM_pow(:)))-40, max(10*log10(RDM_pow(:)))]); % 动态范围压缩

% 只显示前两个最强目标

grid on; 

for i = 1:Nshow
    plot(ranges_detected(i), vels_detected(i), 'm^', 'MarkerSize',3,'LineWidth',1.5);
    text(ranges_detected(i)+0.2, vels_detected(i)+(i-1)*1.5, ...
        sprintf('(%.2f m, %.2f m/s)', ranges_detected(i), vels_detected(i)), ...
        'Color','w','FontSize',10,'FontWeight','bold');
end
hold off;




% ----------------------------
% Additional info
% ----------------------------
df_range = fs / N_range_fft;
dr = c*df_range/(2*S);
fd_res = PRF / N_doppler_fft;
dv = (lambda/2)*fd_res;
fprintf('Range FFT bin: df = %.3f Hz, dr ≈ %.5f m\n', df_range, dr);
fprintf('Doppler FFT bin: fd_res = %.3f Hz, dv ≈ %.5f m/s\n', fd_res, dv);
