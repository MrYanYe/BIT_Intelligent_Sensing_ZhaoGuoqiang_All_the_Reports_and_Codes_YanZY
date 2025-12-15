close all;
clc; clear;

% ----------------------------
% 参数（可按需修改）
% ----------------------------
c = 3e8;
fc = 77e9;        % 载波（仅用于参数推导/注释）
B  = 2e9;         % 带宽 (Hz)
Tc = 40e-6;       % 脉冲/扫频时长 (s)
fs = 4e9;         % 基带采样率（为了在谱图中看见斜线）
S  = B / Tc;      % 斜率 (Hz/s)
f0 = 0;           % 基带起始频率
SNR_dB = 30;      % 目标信噪比，用于加噪声（dB）
% 目标参数
R_target = 10;    % 目标距离 10 m
sigma_reflect = 0.8; % 假设目标“反射率”（0~1）给定
% ----------------------------

dt = 1/fs;
t = 0:dt:Tc-dt;                  % 时间向量
phi = 2*pi*( f0.*t + 0.5 * S .* t.^2 );   % 即时相位 (rad)
s_bb = exp(1j*phi);                      % 复基带线性调频信号
f_inst = S .* t + f0;                    % 即时频率 (Hz)

% 加噪声（发射信号用于显示/对比）
signal_power = mean(abs(s_bb).^2);
SNR_linear = 10^(SNR_dB/10);
noise_power = signal_power / SNR_linear;
noise = sqrt(noise_power/2) * (randn(size(s_bb)) + 1j*randn(size(s_bb)));
s_bb_noisy = s_bb + noise;

% 计算并显示理想的 beat 频率示例（用于参考）
tau = 2*R_target/c;
fB_theory = S * tau;
disp(['theoretical beat freq f_B = ', num2str(fB_theory), ' Hz']);

% ----------------------------
% 模拟目标回波（时延 τ，衰减 α）
% 说明：
% - 对于远场点目标，回波幅度与距离平方成反比，
%   可用一个比例 α = sigma_reflect * (1/R^2) 的形式表示相对幅度。
% - 这里将发射复基带信号延迟 tau 后按 α 缩放并加入噪声得到回波。
% - 延迟可能是非整数采样点，使用线性插值实现亚采样延迟（fractional delay）。
% ----------------------------

% 时延、衰减计算
tau = 2*R_target / c;                 % round-trip delay
delay_samples = tau * fs;             % 可为非整数
% 简单工程设定：alpha = reflectivity * (1/R^2)，再乘常数 K 保证幅度量级合理
% 这里把 K 设为 1（若需要，可按系统雷达方程引入天线增益、波长等）
alpha = sigma_reflect * (1 / (R_target^2));

% 实现亚采样时延（线性插值）
n = 0:length(t)-1;
t_shifted = n - delay_samples;        % 想要的样本索引（可能为小数）
s_rx = zeros(size(s_bb));
% 线性插值边界处理：延迟后超出信号起点的部分为 0（即没回波）
for k = 1:length(n)
    idx = t_shifted(k);
    if idx >= 1 && idx <= length(n)
        i0 = floor(idx);
        frac = idx - i0;
        if i0 < 1
            val = 0;
        elseif i0+1 > length(s_bb)
            val = s_bb(end);
        else
            val = (1-frac)*s_bb(i0) + frac*s_bb(i0+1);
        end
        s_rx(k) = alpha * val;
    else
        s_rx(k) = 0;
    end
end

% 在回波上加入噪声（以与发射同的 SNR 标准，或者单独设置）
% 这里我们让接收噪声与之前同级（可调）
rx_signal_power = mean(abs(s_rx).^2);
if rx_signal_power == 0
    % 若 alpha 太小导致能量数值近零，给一个非常小的噪声基准
    rx_signal_power = 1e-12;
end
noise_power_rx = rx_signal_power / SNR_linear;
noise_rx = sqrt(noise_power_rx/2) * (randn(size(s_rx)) + 1j*randn(size(s_rx)));
s_rx_noisy = s_rx + noise_rx;

% ----------------------------
% 发射与回波混频（齐相，基带混频）
%  对于复基带信号，常用做法是乘以接收信号的共轭：
%    beat(t) = s_tx .* conj(s_rx)
%  对于 FMCW，混频后会得到接近余弦的恒定频率 fB（随延时）
% ----------------------------
beat = s_bb_noisy .* conj(s_rx_noisy);

% 去直流与窗函数，准备 FFT 估计频率
% 这里示例使用单次线性调频的整个脉冲进行 FFT（相当于把 fast-time 看作一个短时信号）
Nfft = 2^16;
w = hann(length(beat)).';
beat_windowed = beat .* w;

% 计算 FFT 并找峰值
BEAT_FFT = fftshift(fft(beat_windowed, Nfft));
freq_axis = (-Nfft/2 : Nfft/2-1) * (fs / Nfft);
PSD = 20*log10(abs(BEAT_FFT) + eps);

% 找到峰值频率（实部或幅度峰）
[~, peak_idx] = max(abs(BEAT_FFT));
fB_est = freq_axis(peak_idx);   % 估计的基带频率（Hz）
% 若使用双边频谱，fB_est 可能为正或负；取绝对值
fB_est = abs(fB_est);

% 将估计的 beat 频率换算为距离
R_est = (c * fB_est) / (2 * S);

% 输出估计
fprintf('Estimated beat freq f_B_est = %.3f Hz\n', fB_est);
fprintf('Estimated range R_est = %.3f m\n', R_est);

% ----------------------------
% 绘图一：时域（优化展示） -- 保留原来的图
% ----------------------------
figure('Name','Time domain overview and zooms','NumberTitle','off','Position',[100 100 900 700]);

% 下采样因子用于展示（仅用于绘图）
ds = 200;                     % 下采样因子（图中点稀疏化），按需增减
t_ds = t(1:ds:end);
re_ds = real(s_bb_noisy(1:ds:end));

subplot(3,1,1);
plot(t_ds*1e6, re_ds, '-k');
xlabel('Time (μs)'); ylabel('Re\{s_{bb}\}');
title('Complex baseband chirp (real part) - downsampled for visibility');
grid on;

% 包络（Hilbert）用于展示幅度变化
env = abs(hilbert(real(s_bb_noisy)));
subplot(3,1,2);
plot(t*1e6, env, 'b');
xlabel('Time (μs)'); ylabel('Envelope amplitude');
title('Envelope of real part (Hilbert)');
grid on;

% 放大两个小段以查看局部波形
subplot(3,1,3);
zoom_len_us = 0.2;                        % 放大段长度 microseconds
zoom_len = max(10, round(zoom_len_us*1e-6*fs));   % 对应样点数，防止过小
center1 = round(5e-6*fs);                % 放大中心1（5 μs）
center2 = round(20e-6*fs);               % 放大中心2（20 μs）

seg1_idx = max(1,center1-round(zoom_len/2)) : min(length(t), center1+round(zoom_len/2));
seg2_idx = max(1,center2-round(zoom_len/2)) : min(length(t), center2+round(zoom_len/2));

plot(t(seg1_idx)*1e6, real(s_bb_noisy(seg1_idx)), '-r'); hold on;
plot(t(seg2_idx)*1e6, real(s_bb_noisy(seg2_idx)), '-g');
xlabel('Time (μs)'); ylabel('Re\{s_{bb}\}');
title('Zoomed segments of real part (two segments)');
legend('Segment at 5 μs','Segment at 20 μs');
grid on;
hold off;

% ----------------------------
% 绘图二：即时频率（原始）
% ----------------------------
figure('Name','Instantaneous frequency','NumberTitle','off','Position',[150 150 700 300]);
plot(t*1e6, f_inst/1e6, 'LineWidth',1.2);    % 以 MHz 显示
xlabel('Time (μs)'); ylabel('Inst. freq (MHz)');
title('Instantaneous frequency (baseband)');
grid on;

% ----------------------------
% 绘图三：Spectrogram 原图（保留）
% ----------------------------
figure('Name','Spectrogram','NumberTitle','off','Position',[200 200 900 600]);
win_len = 512;                            % 增大窗长度可改善频率分辨率
window = hamming(win_len);
noverlap = round(0.75*win_len);
nfft = 4096;                              % 增大 nfft 提升频率插值细节
spectrogram(s_bb_noisy, window, noverlap, nfft, fs, 'centered');
title('Spectrogram of complex baseband chirp (with noise)');
colormap jet;
caxis([-140 -80]);                        % 与原色条一致的 dB/Hz 范围（可调）
c = colorbar;
set(get(c,'Label'),'String','Power (dB/Hz)');

% ----------------------------
% 新增绘图：回波与混频结果与 FFT 结果
% ----------------------------
figure('Name','Echo, mixed beat and FFT','NumberTitle','off','Position',[250 250 1000 800]);

subplot(4,1,1);
plot(t*1e6, real(s_rx_noisy));
xlabel('Time (μs)'); ylabel('Re\{s_{rx}\}');
title(['Received echo (real part). Delay \tau = ', num2str(tau*1e6, '%.3f'), ' μs, alpha = ', num2str(alpha)]);
grid on;

subplot(4,1,2);
plot(t*1e6, real(beat));
xlabel('Time (μs)'); ylabel('Re\{beat\}');
title('Mixed signal (s_{tx} .* conj(s_{rx})) - time domain');
grid on;

subplot(4,1,3);
% 短时傅里叶或单次 FFT 的幅度（线性）
time_zoom = 1e-6 * (0:round(1e-6*fs)-1); % small axis dummy
plot(t(1:round(1e-6*fs))*1e6, real(beat(1:round(1e-6*fs))));
xlabel('Time (μs)'); ylabel('Re\{beat\}');
title('Beat signal zoom (first 1 μs) to show approximately sinusoidal beat');
grid on;

subplot(4,1,4);
plot(freq_axis/1e3, PSD, 'k'); hold on;
plot(fB_est/1e3, PSD(peak_idx), 'ro', 'MarkerFaceColor','r');
% 格式化坐标文本并放在红点右上方
annot_text = sprintf('  (%.2f kHz, %.2f dB)', fB_est/1e3, PSD(peak_idx));
text(fB_est/1e3, PSD(peak_idx), annot_text, 'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',10,'Color','r');

% 标注峰值
plot(fB_est/1e3, PSD(peak_idx), 'ro', 'MarkerFaceColor','r');
xlabel('Frequency (kHz)'); ylabel('Magnitude (dB)');
title(['FFT of beat (Nfft=', num2str(Nfft), ')   Estimated f_B = ', num2str(fB_est,'%.2f'), ' Hz   R_{est} = ', num2str(R_est,'%.3f'), ' m']);
xlim([0 fs/16/1e3]); % 只看正频谱
grid on;
hold off;

% ----------------------------
% 附加：计算并显示谱图参数信息
% ----------------------------
df = fs / nfft;
dt_win = (win_len - noverlap) / fs;
fprintf('STFT frequency bin width df = %.3f Hz\n', df);
fprintf('STFT time hop dt = %.3f us\n', dt_win*1e6);
fprintf('Plotted SNR (dB) target = %.1f dB\n', SNR_dB);
