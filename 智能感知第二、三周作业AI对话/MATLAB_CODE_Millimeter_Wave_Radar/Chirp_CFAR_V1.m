% -------------- 在你原始脚本基础上扩展：生成多次chirp并做2D CA-CFAR --------------
close all; clc; clear;

% ---------- 基本雷达/信号参数（可与原脚本一致） ----------
c = 3e8;
fc = 77e9;
B  = 2e9;
Tc = 40e-6;
fs = 4e9;
S  = B / Tc;
f0 = 0;
SNR_dB = 30;

dt = 1/fs;
t = 0:dt:Tc-dt;
phi = 2*pi*( f0.*t + 0.5 * S .* t.^2 );
s_bb = exp(1j*phi);

% 目标参数（两个目标示例）
R_targets = [10, 25];         % meter，两个目标
sigma_reflect = [0.8, 0.5];   % 反射率
v_targets = [0, 0];           % 径向速度 (m/s)，这里先设静止。若有多普勒可改动。

% 多帧/慢时间设置（构建 R-v 矩阵所需）
Nc = 64;                      % chirps per CPI（慢时间采样数），可调
PRI = Tc + 200e-6;            % PRI（脉间隔），仅用于多普勒相位累积，可大于 Tc
fs_slow = 1/PRI;
% 产生每个 chirp 的回波（简单模型：各目标回波按时延延迟并随慢时间累积多普勒相位）
SNR_linear = 10^(SNR_dB/10);

% 预分配 slow-time 矩阵（每行一个chirp）
Nfast = length(t);
rx_chirps = zeros(Nc, Nfast);    % 复数接收（基带）
tx_chirps = repmat(s_bb, Nc, 1); % 发射恒定chirp（可加入相位扰动）

% 先计算每个目标的时延样本与衰减
taus = 2*R_targets / c;
delay_samples = taus * fs;
alphas = sigma_reflect .* (1 ./ (R_targets.^2));

% 生成每个慢时间chirp的接收信号（叠加目标回波）
for m = 1:Nc
    rx = zeros(1, Nfast);
    % 对每个目标累加带有慢时间多普勒相位的延迟回波
    for k = 1:length(R_targets)
        tau = taus(k);
        ds = delay_samples(k);
        % 线性插值实现亚采样延迟（与原脚本类似）
        t_shifted_idx = (0:Nfast-1) - ds;
        val = zeros(1, Nfast);
        for n = 1:Nfast
            idx = t_shifted_idx(n);
            if idx >= 1 && idx <= Nfast
                i0 = floor(idx);
                frac = idx - i0;
                if i0 < 1
                    v = 0;
                elseif i0+1 > Nfast
                    v = s_bb(end);
                else
                    v = (1-frac)*s_bb(i0) + frac*s_bb(i0+1);
                end
            else
                v = 0;
            end
            val(n) = v;
        end
        % slow-time 多普勒相位：exp(j*2*pi*2*v_target*m*PRI/lambda) （若 v=0 则为1）
        lambda = c / fc;
        doppler_phase = exp(1j * 2*pi * (2*v_targets(k))/lambda * (m-1) * PRI);
        rx = rx + alphas(k) * val * doppler_phase;
    end
    % 在每个chirp上加入接收噪声（让每个chirp SNR大致相同）
    rx_power = mean(abs(rx).^2);
    if rx_power == 0
        rx_power = 1e-12;
    end
    noise_power = rx_power / SNR_linear;
    noise = sqrt(noise_power/2) * (randn(size(rx)) + 1j*randn(size(rx)));
    rx_chirps(m,:) = rx + noise;
end

% 发射信号（用于混频）——这里用每帧相同的s_bb
tx_rep = repmat(s_bb, Nc, 1);

% 混频（逐chirp做：beat = tx .* conj(rx)）
beat_chirps = tx_rep .* conj(rx_chirps);

% 对每个chirp做快速时间 FFT（range FFT）
Nfft_range = 2^12;   % range FFT points，可调整以改变距离分辨
range_fft = fft(beat_chirps, Nfft_range, 2);
% 取前半谱（正频谱对应正距离）
range_fft_pos = range_fft(:, 1:Nfft_range/2);

% 生成 R-v 矩阵（慢时间沿行，快时间FFT为列）
Rv = range_fft_pos;   % size: Nc x Nrange_bins
% 也可以取幅度平方（功率谱）用于CFAR
Rv_pow = abs(Rv).^2;

% 画出 R-v 热图（幅度）
figure('Name','Range-Doppler (slow-time vs range)','NumberTitle','off');
imagesc(1:size(Rv_pow,2), 1:Nc, 10*log10(Rv_pow)); axis xy;
xlabel('Range bin'); ylabel('Slow time (chirp index)');
title('R-v matrix (dB)');

% -------------------------
% 2D CA-CFAR 实现（滑动窗口）——通用实现（cell averaging across 2D neighborhood）
% -------------------------
% CFAR 参数（可调）
Pfa = 1e-6;           % 目标虚警率
guard_range = 2;      % range 方向 guard cells each side
guard_doppler = 1;    % doppler/slow-time 方向 guard cells each side
train_range = 8;      % range 方向训练单元 each side
train_doppler = 4;    % doppler 方向训练单元 each side

% 训练单元总数
N_train = (2*train_range+2*guard_range+1 - (2*guard_range+1)) * (2*train_doppler+2*guard_doppler+1 - (2*guard_doppler+1));
% 更直接计算：训练区域总格点数（exclude CUT and guard）
% 计算训练单元数方法更稳妥：
win_range = 2*(train_range+guard_range) + 1;
win_dop   = 2*(train_doppler+guard_doppler) + 1;
guard_w_range = 2*guard_range + 1;
guard_w_dop   = 2*guard_doppler + 1;
N_train = win_range * win_dop - guard_w_range * guard_w_dop;

% 阈值缩放因子 α（假设训练细胞平方和/均值符合指数分布推导）
alpha = N_train * (Pfa^(-1/N_train) - 1);

% 结果存储
[Cfar_mask, threshold_map] = deal(zeros(size(Rv_pow)));

[nRows, nCols] = size(Rv_pow);

% 为了效率，预先计算二维滑动和可以用 conv2 进行训练区和 guard 区加权和
% 这里采用直接循环实现（可读性更好），若数据很大可改成 integral image/conv2优化

for r = 1:nRows
    for c = 1:nCols
        % 窗口边界
        r1 = r - (train_doppler + guard_doppler);
        r2 = r + (train_doppler + guard_doppler);
        c1 = c - (train_range + guard_range);
        c2 = c + (train_range + guard_range);
        % guard inner boundary
        rg1 = r - guard_doppler;
        rg2 = r + guard_doppler;
        cg1 = c - guard_range;
        cg2 = c + guard_range;
        % clip to matrix
        r1c = max(1, r1); r2c = min(nRows, r2);
        c1c = max(1, c1); c2c = min(nCols, c2);
        rg1c = max(1, rg1); rg2c = min(nRows, rg2);
        cg1c = max(1, cg1); cg2c = min(nCols, cg2);
        % extract window
        window = Rv_pow(r1c:r2c, c1c:c2c);
        % mask out guard + CUT
        % compute indices of guard relative to window
        rr = (r1c:r2c);
        cc = (c1c:c2c);
        [RR,CC] = ndgrid(rr,cc);
        guard_mask = (RR >= rg1c & RR <= rg2c & CC >= cg1c & CC <= cg2c);
        % training cells are window where guard_mask == false
        train_cells = window(~guard_mask);
        % if not enough training cells (edge), skip or use available
        if isempty(train_cells)
            noise_level = eps;
        else
            noise_level = mean(train_cells(:));  % cell-averaging
        end
        thresh = alpha * noise_level;
        threshold_map(r,c) = thresh;
        if Rv_pow(r,c) > thresh
            Cfar_mask(r,c) = 1;
        else
            Cfar_mask(r,c) = 0;
        end
    end
end

% 显示 CFAR 检测结果（覆盖在 RV 图上）
figure('Name','CA-CFAR detections on R-v','NumberTitle','off');
imagesc(1:nCols, 1:nRows, 10*log10(Rv_pow)); axis xy; colormap jet; hold on;
% 找到检测点并绘制红圈
[det_r, det_c] = find(Cfar_mask==1);
plot(det_c, det_r, 'wo', 'MarkerSize',8, 'LineWidth',1.5);
xlabel('Range bin'); ylabel('Slow time (chirp index)');
title(sprintf('2D CA-CFAR detections (Pfa=%.1e, N_{train}=%d)', Pfa, N_train));
hold off;

% 为便于判断，把 Range dimension 做平移求和（将慢时间上的检测投影到range）
range_detection_count = sum(Cfar_mask,1); % 对慢时间求和
range_bins = 1:nCols;
figure('Name','Range detection projection','NumberTitle','off');
plot(range_bins, range_detection_count, '-k', 'LineWidth',1.4); grid on;
xlabel('Range bin'); ylabel('Number of detections across slow-time');
title('Projection of CFAR detections onto range axis');

% 打印参数与提示
fprintf('CFAR parameters: train_range=%d, train_doppler=%d, guard_range=%d, guard_doppler=%d\n', ...
    train_range, train_doppler, guard_range, guard_doppler);
fprintf('Total training cells N_train = %d, alpha = %.3e (for Pfa=%.1e)\n', N_train, alpha, Pfa);

% 将检测的 range bin 转换为距离（若需）
range_res = c / (2 * S) * (fs / Nfft_range); % 每bin距离（近似）
det_ranges_m = (det_c-1) * range_res;
if ~isempty(det_ranges_m)
    fprintf('Detected ranges (m) sample: %s\n', mat2str(unique(round(det_ranges_m,3))));
else
    fprintf('No detections found.\n');
end

% ----------------------------------------------------------------------------
% 说明补充：
% - 若要检测速度（doppler）峰值，可对每个range bin在慢时间上做FFT得到真实多普勒轴并用二维CFAR在频移-距阵上检测。
% - 若边缘窗口训练单元较少，alpha解析式仍然适用但实际Pfa会偏离；常用做法是在边界处跳过检测或用镜像/周期填充训练区。
% - 若噪声统计不满足指数假设（例如存在强杂波/非高斯噪声），可采用秩次或中值CFAR变种（OS-CFAR）以增强鲁棒性。
% ----------------------------------------------------------------------------
