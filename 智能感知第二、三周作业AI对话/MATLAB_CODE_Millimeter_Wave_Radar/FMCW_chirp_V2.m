close all;
clc; clear;

% ----------------------------
% 参数（可按需修改）
% ----------------------------
fc = 77e9;        % 载波（仅用于参数推导）
B  = 2e9;         % 带宽 (Hz)
Tc = 40e-6;       % 脉冲/扫频时长 (s)
fs = 4e9;         % 基带采样率（为了在谱图中看见斜线）
S  = B / Tc;      % 斜率 (Hz/s)
f0 = 0;           % 基带起始频率
SNR_dB = 30;      % 目标信噪比，用于加噪声（dB）
% ----------------------------

dt = 1/fs;
t = 0:dt:Tc-dt;                  % 时间向量
phi = 2*pi*( f0.*t + 0.5 * S .* t.^2 );   % 即时相位 (rad)
s_bb = exp(1j*phi);                      % 复基带线性调频信号
f_inst = S .* t + f0;                    % 即时频率 (Hz)

% 加噪声（复高斯噪声，调整为期望 SNR）
signal_power = mean(abs(s_bb).^2);
SNR_linear = 10^(SNR_dB/10);
noise_power = signal_power / SNR_linear;
noise = sqrt(noise_power/2) * (randn(size(s_bb)) + 1j*randn(size(s_bb)));
s_bb_noisy = s_bb + noise;

% 计算并显示理想的 beat 频率示例
R = 0.10;
tau = 2*R/3e8;
fB = S * tau;
disp(['beat freq f_B = ', num2str(fB), ' Hz']);

% ----------------------------
% 绘图一：时域（优化展示）
%  - 下采样展示整体实部
%  - 两个放大片段显示细节
%  - 绘制幅度包络
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
% 绘图三：Spectrogram
%  - 设置窗长、重叠与 nfft，兼顾时间/频率分辨
%  - 使用中心化频率显示正负基带
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

% 兼容式设置 colorbar 标签
c = colorbar;
set(get(c,'Label'),'String','Power (dB/Hz)');

% ----------------------------
% 附加：计算并显示谱图参数信息
% ----------------------------
df = fs / nfft;
dt_win = (win_len - noverlap) / fs;
fprintf('STFT frequency bin width df = %.3f Hz\n', df);
fprintf('STFT time hop dt = %.3f us\n', dt_win*1e6);
fprintf('Plotted SNR (dB) target = %.1f dB\n', SNR_dB);
