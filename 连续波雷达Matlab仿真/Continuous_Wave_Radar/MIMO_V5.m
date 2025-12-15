% MIMO_V4_statEst.m
% Automotive Medium Range Radar (MRR) - SNR sweep, group-by-target vertical comparison
% 改为统计估计：加权质心与二阶矩（标准差）代替最大峰值搜索

clc; clear; close all;

c0   = 3e8; % Light Speed
% Requirements / constants
delta_R = 0.3; % m
delta_V = 1; % m/s
delta_angle = 8; % degrees
R_max = 160; % m
Vel_max = 200; % km/h
RCS_car = 10*log10(10); % dBsqm

% Frequency / waveform / antenna
Fc = 78e9;
lambda = c0 / Fc;

Na = round(360/pi/delta_angle); % antenna elements
delta_dis = lambda/2;

Gt = 15; Gr = 15; % dBi

% Receiver / Tx settings
F = 15; % Noise figure dB
Pt = 12; % Tx power dBm
L = 2;  % system loss dB

% Sweep / PRF / CPI
fd_max = 2*Vel_max*1e3/3600 / lambda;
T = 1/(2*fd_max);
PRF = 1/T;
CPI = round(PRF/(2*delta_V/lambda));
CPI = 2^(round(log2(CPI)));

% Range check
R_max_new = c0*T/2;
if R_max_new < R_max, error('Unusable: R_max_new < R_max'); end

te = 290; k_constant = 1.38e-23;

% Waveform params
B = c0/(2*delta_R);
Kr = B/T;
fs = 4*Kr*R_max/c0 + 1/T;
RF_fs = B*3;

t  = (0 : 1/RF_fs : T-1/RF_fs);
TX_RF   = exp(1i*pi*Kr*t.^2).*exp(1i*2*pi*Fc*t);
TX_Ref  = conj(TX_RF);

fs = RF_fs/round(RF_fs/fs);
N_Fast  = round(T*fs);

dletaR = fs/N_Fast*3e8/2/Kr;
idR = dletaR*[0:N_Fast/2-1];
idV = lambda/2*(-PRF/2:PRF/CPI:PRF/2-PRF/CPI);

% Lowpass design (kept from original)
Fpass = 0.9*fs/2; Fstop = 1.1*Fpass;
Dpass = 0.0057501127785; Dstop = 0.0001; dens = 20;
[Nfir, Fo, Ao, W] = firpmord([Fpass, Fstop]/(fs/2), [1 0], [Dpass, Dstop]);
b  = firpm(Nfir, Fo, Ao, W, {dens});
Hd = dfilt.dffir(b);

% True targets
target_rcs = [8,10,15];
target_range = [36, 50, 60];
target_velocity = [5, -10 ,15];
target_theta = [-35, 0, 60];

% baseline noise amplitude (power -> voltage)
noise_amp = k_constant*te*10^(F/10)*B;
noise_amp = sqrt(noise_amp);

LNA_ADC_Gain = 48; % dB

% SNR sweep array (dB)
SNR_dB_array = [-20,-10,0,10,20,40];
% SNR_dB_array = [0.1,0,  10, 20,40];
nSNR = length(SNR_dB_array);
nTgt = length(target_range);

% Results storage: Results(target, sidx, :) = [range, range_std, vel, vel_std, angle, angle_std]
Results = nan(nTgt, nSNR, 6);

% Parameters for local windows used in centroid estimation (in bins)
r_win_half = 2; % range half-width (bins) for RD local window
v_win_half = 2; % velocity half-width (bins)
ang_win_half = 6; % angle half-width (bins) for angle-range slice

% Loop over SNR values
for sidx = 1:nSNR
    SNR_dB = SNR_dB_array(sidx);
    noise_amp_scaled = noise_amp * 10^(-SNR_dB/20);
    
    % Range-Doppler Raw Data Simulation
    Rawdata(CPI, N_Fast) = 0;
    for k = 1:CPI
        echo = TX_RF*0;
        for tn = 1:nTgt
            range = target_range(tn) + target_velocity(tn)*T*(k-1);
            echo = echo + target_rcs(tn)*exp(1i*pi*Kr*(t-2*range/c0).^2).*exp(1i*2*pi*Fc*(t-2*range/c0)) / (range^4);
        end
        Gain = Pt-30 + Gt + Gr + 20*log10(lambda) + LNA_ADC_Gain - 30*log10(4*pi) - L;
        Gain = sqrt(10^(Gain/10));
        echo = echo * Gain;
        echo = echo + noise_amp_scaled.*(randn(1,length(echo)) + 1i*randn(1,length(echo)));
        Mixer_Output = conj(echo .* TX_Ref);
        Mixer_Output = decimate(Mixer_Output, RF_fs/fs);
        RX_Base  = filter(Hd, Mixer_Output);
        Rawdata(k,:) = RX_Base;
    end
    
    % Range-Doppler transform
    x = Rawdata(1,:);
    N_sig = length(x);
    raw_data = Rawdata.';
    raw_data = fftshift(fft(raw_data, [], 1), 1); % Range FFT
    raw_data = fftshift(fft(raw_data, [], 2), 2); % Doppler FFT
    raw_data = raw_data(1 + N_sig/2 : N_sig, :);
    % use linear power (not dB) for statistical estimation
    rd_power = abs(raw_data).^2;
    rd_map = 20*log10(abs(raw_data) + eps) + 30; % keep for plotting if needed
    
    % Estimate range & velocity for each true target using local weighted centroid
    est_ranges = zeros(1,nTgt);
    est_ranges_std = zeros(1,nTgt);
    est_vels = zeros(1,nTgt);
    est_vels_std = zeros(1,nTgt);
    for tn = 1:nTgt
        true_vel = target_velocity(tn);
        true_rng = target_range(tn);
        % find grid indices closest to true values
        [~, vidx0] = min(abs(idV - true_vel));
        [~, ridx0] = min(abs(idR - true_rng));
        % define local window
        r_window = max(1,ridx0-r_win_half):min(size(rd_power,1),ridx0+r_win_half);
        v_window = max(1,vidx0-v_win_half):min(size(rd_power,2),vidx0+v_win_half);
        subP = rd_power(r_window, v_window);
        % coordinates mesh
        [Vr, Rr] = meshgrid(idV(v_window), idR(r_window));
        weights = subP;
        Wsum = sum(weights(:)) + eps;
        % weighted centroid (range, vel)
        rng_centroid = sum(weights(:) .* Rr(:)) / Wsum;
        vel_centroid = sum(weights(:) .* Vr(:)) / Wsum;
        % weighted variance (second central moment) -> std
        rng_var = sum(weights(:) .* (Rr(:)-rng_centroid).^2) / Wsum;
        vel_var = sum(weights(:) .* (Vr(:)-vel_centroid).^2) / Wsum;
        est_ranges(tn) = rng_centroid;
        est_ranges_std(tn) = sqrt(rng_var);
        est_vels(tn) = vel_centroid;
        est_vels_std(tn) = sqrt(vel_var);
    end
    
    % Azimuth processing (MIMO / angle-range)
    MIMO_data(Na:N_Fast) = 0;
    for k = 1:Na
        echo = TX_RF*0;
        for tn = 1:nTgt
            range = target_range(tn);
            echo = echo + target_rcs(tn)*exp(1i*pi*Kr*(t-2*range/c0).^2).*exp(1i*2*pi*Fc*(t-2*range/c0)) ...
                * exp(-1i*2*pi*(k-1)*delta_dis*sind(target_theta(tn))/lambda) / (range^4);
        end
        Gain = 30 + Pt-30 + Gt + Gr + 20*log10(lambda) + LNA_ADC_Gain - 30*log10(4*pi) - L;
        Gain = sqrt(10^(Gain/10));
        echo = echo * Gain;
        echo = echo + noise_amp_scaled.*(randn(1,length(echo)) + 1i*randn(1,length(echo)));
        Mixer_Output = conj(echo .* TX_Ref);
        Mixer_Output = decimate(Mixer_Output, RF_fs/fs);
        RX_Base  = filter(Hd, Mixer_Output);
        MIMO_data(k,:) = RX_Base;
    end
    rafData = fftshift(fft(MIMO_data, [], 2), 2);
    rafData = rafData(:, 1 + N_sig/2 : N_sig);
    Ridx = dletaR*[0:N_Fast/2-1];
    Na_fft = 256; az0 = linspace(-1,1,Na_fft); azz = asind(az0);
    data = fftshift(fft(rafData, Na_fft, 1), 1);
    % keep linear amplitude for stats
    data_lin = abs(data);
    tdata = 20*log10(data_lin/max(data_lin(:)) + eps); % for plotting
    
    % Estimate angle for each estimated range using weighted centroid over angle bins
    est_angles = zeros(1,nTgt);
    est_angles_std = zeros(1,nTgt);
    for tn = 1:nTgt
        % find nearest range bin in Ridx to the estimated range
        [~, ridx_ang] = min(abs(Ridx - est_ranges(tn)));
        colIdx = min(max(1, ridx_ang), size(data_lin,2));
        az_slice = data_lin(:, colIdx); % amplitude over angle bins
        % define local angular window around peak for stability
        [~, peak_idx] = max(az_slice);
        a_win = max(1, peak_idx - ang_win_half) : min(length(azz), peak_idx + ang_win_half);
        az_bins = azz(a_win);
        weights = az_slice(a_win).^2; % use power-like weights for sharper localization
        Wsum = sum(weights) + eps;
        % angular centroid (degrees)
        ang_centroid = sum(weights .* az_bins') / Wsum;
        % angular variance
        ang_var = sum(weights .* (az_bins' - ang_centroid).^2) / Wsum;
        est_angles(tn) = ang_centroid;
        est_angles_std(tn) = sqrt(ang_var);
    end
    
    % Save into Results array (range, range_std, vel, vel_std, angle, angle_std)
    for tn = 1:nTgt
        Results(tn, sidx, 1) = est_ranges(tn);
        Results(tn, sidx, 2) = est_ranges_std(tn);
        Results(tn, sidx, 3) = est_vels(tn);
        Results(tn, sidx, 4) = est_vels_std(tn);
        Results(tn, sidx, 5) = est_angles(tn);
        Results(tn, sidx, 6) = est_angles_std(tn);
    end
    
    % Save last-run maps for later plotting
    if sidx == nSNR
        rd_map_last = 20*log10(rd_power + eps);
        tdata_last = tdata;
        Ridx_last = Ridx;
        est_ranges_last = est_ranges;
        est_ranges_std_last = est_ranges_std;
        est_vels_last = est_vels;
        est_vels_std_last = est_vels_std;
        est_angles_last = est_angles;
        est_angles_std_last = est_angles_std;
        azz_last = azz;
    end
end

% Display a single combined table grouped by target (each target block: rows = SNRs)
fprintf('\nGrouped-by-target summary (每个目标下按 SNR 竖向排列便于比较):\n');
for tn = 1:nTgt
    fprintf('\nTarget %d (true R=%.1f m, V=%.1f m/s, Ang=%.1f deg)\n', tn, target_range(tn), target_velocity(tn), target_theta(tn));
    fprintf('-----------------------------------------------------------------------------------------\n');
    fprintf('%6s | %12s | %10s | %12s | %10s | %10s | %12s\n ', 'SNR(dB)', 'Range(m)', 'R_std(m)', 'Velocity(m/s)', 'V_std(m/s)', 'Angle(deg)','Angle_std(deg)');
    fprintf('-----------------------------------------------------------------------------------------\n');
    for sidx = 1:nSNR
        fprintf('%6.1f | %12.2f | %10.2f | %12.2f | %10.2f | %10.2f | %10.2f\n', ...
            SNR_dB_array(sidx), Results(tn,sidx,1), Results(tn,sidx,2), ...
                Results(tn,sidx,3), Results(tn,sidx,4), Results(tn,sidx,5), Results(tn,sidx,6));

    end
    fprintf('-----------------------------------------------------------------------------------------\n');
end

% Plot only for the last SNR: Range-Doppler power (linear->dB) with centroid markers
fontsz = 16;
figure('Name','Range-Doppler (last SNR)','NumberTitle','off');
surf(idV, idR, rd_map_last, 'EdgeColor', 'none');
xlabel('Velocity (m/s)'); ylabel('Range (m)'); zlabel('Intensity (dB)');
colormap(jet); shading interp; view(-44,41);
xlim([-40 40]); ylim([0 80]);
caxis([max(rd_map_last(:))-60 max(rd_map_last(:))]); % dynamic scale around peaks
h = colorbar; title(h,'dB'); a = h.Position; set(h,'Position',[a(1)+0.08 a(2)+0.23 0.01 0.5]);
set(gcf,'color',[1 1 1]); set(gca,'FontSize',fontsz); set(findall(gcf,'type','text'),'FontSize',fontsz);
hold on;
for tn = 1:nTgt
    % plot centroid and uncertainty ellipse projection (std as error bars)
    [~, vidx] = min(abs(idV - est_vels_last(tn)));
    [~, ridx] = min(abs(idR - est_ranges_last(tn)));
    map_val = interp2(idV, idR, rd_map_last, est_vels_last(tn), est_ranges_last(tn));
    plot3(est_vels_last(tn), est_ranges_last(tn), map_val, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
    % annotation with std
    ann_txt = sprintf('Vel = %.2f ± %.2f m/s\nRange = %.2f ± %.2f m', ...
    est_vels_last(tn), est_vels_std_last(tn), ...
    est_ranges_last(tn), est_ranges_std_last(tn));

    text(est_vels_last(tn)+1, est_ranges_last(tn)+2, map_val, ann_txt, 'FontSize',9, 'Color','r','FontWeight','bold');
end

% Angle-Range (last SNR) with statistical angle markers
Xr = Ridx_last' * cosd(azz_last);
Yr = Ridx_last' * sind(azz_last);
figure('Name','Angle-Range (last SNR)','NumberTitle','off');
pcolor(Yr', Xr', tdata_last); shading interp; colormap(jet);
xlim([-R_max R_max]); ylim([0 R_max]); h2 = colorbar; title(h2,'dB');
xlabel('X (m)'); ylabel('Y (m)'); a = h2.Position; set(h2,'Position',[a(1)+0.08 a(2)+0.05 0.01 0.6]);
set(gcf,'color',[1 1 1]); set(gca,'FontSize',fontsz); set(findall(gcf,'type','text'),'FontSize',fontsz);
hold on;
for tn = 1:nTgt
    ang = est_angles_last(tn);
    ang_std = est_angles_std_last(tn);
    rg = est_ranges_last(tn);
    xp = rg * sind(ang); yp = rg * cosd(ang);
    plot(xp, yp, 'ro', 'MarkerSize',8, 'LineWidth',1.5);
    % plot small arc or error bars to show angular uncertainty (approx)
    nums = 21;
    angs = linspace(ang-ang_std, ang+ang_std, nums);
    arc_x = rg * sind(angs);
    arc_y = rg * cosd(angs);
    plot(arc_x, arc_y, 'r--', 'LineWidth',1);
    txt = sprintf('Ang=%.1f\\pm%.1f° R=%.1f m', ang, ang_std, rg);
    text(xp+2, yp+2, txt, 'FontSize',10, 'Color','r', 'FontWeight','bold');
end
