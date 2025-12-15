clear; close all; clc;

c = 3e8;
fc = 77e9; lambda = c/fc;
Tc = 40e-6;
fs = 4e9;
PRI = 60e-6;
num_chirps = 128;

% 三个带宽
B_list = [1e9, 2e9, 3e9];

% 把目标间距设为一个能观察到差异的值（你也可以逐步减小看分界）
R_center = 10;   
deltaR = 0.3;     % 0.3 m 比 1GHz 的理论 0.15m 稍大，便于可视化
R1 = R_center - deltaR/2;
R2 = R_center + deltaR/2;
v1 = 0; v2 = 0;
sigma1 = 1; sigma2 = 1;

SNR_dB_rx = 50; % 提高 SNR
SNR_lin = 10^(SNR_dB_rx/10);

% FFT 设置
Nfft_range = 65536;   % 很大零填充以增强谱图细节（视觉上更容易分辨）
Nfft_doppler = 256;
win_fast = hann(round(Tc*fs));
win_slow = hann(num_chirps);

% 显示范围缩到目标附近
R_display_min = R_center - 1;
R_display_max = R_center + 1;

figure('Position',[100 100 1400 600]);
for ib = 1:length(B_list)
    B = B_list(ib);
    S = B / Tc;
    % 生成 tx chirp
    t_fast = (0:1/fs: Tc-1/fs);
    s_tx = exp(1j*2*pi*(0.5*S.*t_fast.^2));
    N_fast = numel(t_fast);

    % 生成 BEAT_mat（仅用单帧多chirp对齐相同回波）
    BEAT_mat = zeros(N_fast, num_chirps);
    for m = 1:num_chirps
        % 两个静止目标
        tau1 = 2*R1 / c; tau2 = 2*R2 / c;
        alpha1 = sigma1 / (R1^2); alpha2 = sigma2 / (R2^2);
        t_samples = (0:N_fast-1)/fs;
        s_rx1 = alpha1 * interp1(t_samples, s_tx, t_samples - tau1, 'linear', 0);
        s_rx2 = alpha2 * interp1(t_samples, s_tx, t_samples - tau2, 'linear', 0);
        s_rx = s_rx1 + s_rx2;
        % 加噪（按回波功率）
        rx_power = mean(abs(s_rx).^2);
        if rx_power==0, rx_power=1e-12; end
        noise_rx = sqrt(rx_power/SNR_lin/2) * (randn(size(s_rx)) + 1j*randn(size(s_rx)));
        s_rx_noisy = s_rx + noise_rx;
        s_tx_noisy = s_tx;
        beat = s_tx_noisy .* conj(s_rx_noisy);
        beat = beat - mean(beat); % 去直流，避免 v=0 垂直泄漏
        BEAT_mat(:,m) = beat(:);
    end

    % 2D FFT -> Range-Doppler
    BEAT_win = (BEAT_mat .* repmat(win_fast(:),1,num_chirps)) .* repmat(win_slow(:).', N_fast, 1);
    R_fft = fft(BEAT_win, Nfft_range, 1);
    RD_map = fftshift(fft(R_fft, Nfft_doppler, 2), 2);
    RD_dB = 20*log10(abs(RD_map)+eps);

    % Range axis
    df_range = fs / Nfft_range;
    f_axis = (0:Nfft_range-1) * df_range;
    R_axis = (c * f_axis) / (2 * S);
    half_idx = 1:floor(Nfft_range/2);
    R_axis_use = R_axis(half_idx);
    RD_use = RD_dB(half_idx, :);

    % 找显示索引
    idx_r_min = find(R_axis_use >= R_display_min, 1, 'first');
    idx_r_max = find(R_axis_use <= R_display_max, 1, 'last');

    % 画 Range-Doppler 局部热图
    subplot(2, length(B_list), ib);
    v_axis = (-Nfft_doppler/2 : Nfft_doppler/2-1) * (1/(PRI*Nfft_doppler)); 
    v_axis = v_axis * lambda / 2;
    imagesc(v_axis, R_axis_use(idx_r_min:idx_r_max), RD_use(idx_r_min:idx_r_max, :));
    axis xy; xlabel('Velocity (m/s)'); ylabel('Range (m)');
    title(sprintf('B=%.0f MHz, theory ΔR=%.3f m', B/1e6, c/(2*B)));
    caxis(max(RD_use(:)) + [-40 0]); colormap jet; colorbar;

    % 提取零多普勒（中列）的 range profile 并画出来（归一化）
    subplot(2, length(B_list), length(B_list)+ib);
    dop_col = round(Nfft_doppler/2); % zero doppler column
    range_profile = RD_use(:, dop_col);
    % 只看兴趣区并归一化
    rp = range_profile(idx_r_min:idx_r_max);
    Rp_axis = R_axis_use(idx_r_min:idx_r_max);
    rp_norm = rp - max(rp); % 0 为峰
    plot(Rp_axis, rp_norm, '-k','LineWidth',1.2); hold on;
    xlabel('Range (m)'); ylabel('Relative mag (dB)');
    title(sprintf('Range profile (zero doppler) B=%.0f MHz', B/1e6));
    ylim([-60 5]); xlim([R_display_min R_display_max]); grid on;

    % 峰值检测与二次插值（parabolic）以求亚箱峰位置
    [pks, locs] = findpeaks(rp_norm, 'SortStr','ascend'); % note rp_norm is negative
    % convert to usual positive peaks for convenience
    [pksp, locsp] = findpeaks(-rp_norm, 'SortStr','descend');
    if numel(locsp) >= 2
        % take two largest peaks
        locs_sel = locsp(1:min(2,numel(locsp)));
        cols = ['r','m'];
        for k = 1:length(locs_sel)
            idxp = locs_sel(k);
            % parabolic interpolation around peak idxp
            if idxp>1 && idxp < length(rp_norm)
                y1 = -rp_norm(idxp-1); y2 = -rp_norm(idxp); y3 = -rp_norm(idxp+1);
                p = (y1 - y3) / (2*(y1 - 2*y2 + y3)); % parabola offset (bins)
                peak_bin = idxp + p;
                peak_range = Rp_axis(1) + (peak_bin-1)*(Rp_axis(2)-Rp_axis(1));
                peak_db = - (y2 - (y1 - y3)^2/(8*(y1 - 2*y2 + y3))); % approx peak dB
            else
                peak_range = Rp_axis(idxp);
                peak_db = -rp_norm(idxp);
            end
            plot(peak_range, -peak_db, 'o', 'MarkerEdgeColor', cols(k), 'MarkerFaceColor', cols(k));
            text(peak_range, -peak_db+2, sprintf('R=%.4fm', peak_range), 'Color', cols(k));
        end
    else
        % 若检测不到两个峰，尝试显示最大峰
        [~, imx] = max(-rp_norm);
        plot(Rp_axis(imx), rp_norm(imx), 'ro');
    end
    hold off;
end
