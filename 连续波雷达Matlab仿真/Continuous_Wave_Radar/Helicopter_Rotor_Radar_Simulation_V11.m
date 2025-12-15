%% 直升机旋翼微动效应仿真与雷达信号分析 —— V2 为主体，第一幅图替换为 V8 的 Range-Doppler 动画（其余图保持不变）
clear; clc; close all;

%% =========================================================================
% 1. 参数初始化
%% =========================================================================
params = struct();
params.rotor_num = 4;
params.main_rotor_radius = 7.315;
params.main_rotor_rpm = 258;
params.tail_rotor_radius = 1.2;
params.tail_rotor_rpm = 1200;
params.flap_angle_range = [-2, 2];
params.lag_angle_range = [-1.5, 1.5];
params.twist_angle_range = [-10, 14];

params.radar_freq = 10e9;
params.lambda = 3e8 / params.radar_freq;
params.fs = 1e4;           % 保持 V2 的采样率
params.snr_dB = 15;
params.alpha_atten = 0.1;

params.sim_time = 5;       % 保持 V2 的仿真时长
% 注意：若仿真时间太长，绘图和内存消耗较大；可适当缩短以测试
params.t = linspace(0, params.sim_time, round(params.sim_time * params.fs));
params.omega_main = params.main_rotor_rpm * 2*pi/60;

params.flight_mode = 'hover';
params.pitch_angle = 0;
params.roll_angle = 0;

% TX/visualization parameters for Range-Doppler animation (借用 V8 风格)
tx = struct();
tx.prf = 100;
tx.pulse_len = 1/tx.prf;
tx.B = 20e6;
tx.f0 = params.radar_freq - tx.B/2;

%% =========================================================================
% 2. 辅助函数（保留 V2 的并增加 V8 中的部分用于动画）
%% =========================================================================

function lag_displacement = sim_lag_motion(t, params)
    % 针对向量 t 返回位移（与 V2 保持相同接口）
    m = 0.5; k = 800; c = 15;
    wn = sqrt(k/m);
    zeta = c/(2*sqrt(m*k));
    x0 = 0.05; v0 = 0;
    if zeta < 1
        wd = wn * sqrt(max(0, 1 - zeta^2));
        wd = max(wd, eps);
        lag_displacement = exp(-zeta*wn.*t) .* (x0.*cos(wd.*t) + (v0 + zeta*wn*x0)./wd.*sin(wd.*t));
    else
        lambda1 = (-c + sqrt(max(0, c^2 - 4*m*k)))/(2*m);
        lambda2 = (-c - sqrt(max(0, c^2 - 4*m*k)))/(2*m);
        denom = (lambda1 - lambda2);
        if abs(denom) < eps
            A = 0; B = 0;
        else
            A = (v0 - lambda2*x0)/denom;
            B = (lambda1*x0 - v0)/denom;
        end
        lag_displacement = A*exp(lambda1.*t) + B*exp(lambda2.*t);
    end
end

function lag_displacement = sim_lag_motion_scalar(t)
    % 标量 t 版本（用于 V8 风格的逐点计算）
    m = 0.5; k = 800; c = 15;
    wn = sqrt(k/m);
    zeta = c/(2*sqrt(m*k));
    x0 = 0.05; v0 = 0;
    if zeta < 1
        wd = wn * sqrt(max(0, 1 - zeta^2));
        wd = max(wd, eps);
        lag_displacement = exp(-zeta*wn*t) .* (x0.*cos(wd*t) + (v0 + zeta*wn*x0)./wd.*sin(wd*t));
    else
        lambda1 = (-c + sqrt(max(0, c^2 - 4*m*k)))/(2*m);
        lambda2 = (-c - sqrt(max(0, c^2 - 4*m*k)))/(2*m);
        denom = (lambda1 - lambda2);
        if abs(denom) < eps
            A = 0; B = 0;
        else
            A = (v0 - lambda2*x0)/denom;
            B = (lambda1*x0 - v0)/denom;
        end
        lag_displacement = A*exp(lambda1*t) + B*exp(lambda2*t);
    end
end

function doppler_history = compute_doppler(velocities, params)
    ur = [0, 0, 1]; % 雷达视线方向
    v_r = velocities * ur';
    doppler_history = (2 * v_r) / params.lambda;
end

function doppler_history = compute_doppler_simple(velocities, lambda)
    ur = [0, 0, 1];
    v_r = velocities * ur';
    doppler_history = (2 * v_r) / lambda;
end

function [echo_signal, echo_noisy] = generate_radar_echo(doppler_history, params)
    N = size(doppler_history, 1);
    T = length(params.t);
    rcs_values = 0.1 + 0.9*rand(N, 1);
    echo_signal = zeros(1, T);
    if size(doppler_history,2) ~= T
        error('doppler_history 尺寸与 params.t 长度不匹配');
    end
    for i = 1:N
        signal_i = rcs_values(i) .* exp(1j * 2*pi * (doppler_history(i, :) .* params.t));
        echo_signal = echo_signal + signal_i;
    end
    slant_range = 1000 + 50*sin(params.omega_main*params.t);
    L_a = exp(-params.alpha_atten * slant_range/1000);
    echo_atten = echo_signal .* L_a;
    sigma_n = std(real(echo_atten));
    if sigma_n == 0
        sigma_n = 1e-12;
    end
    sigma_n = sigma_n / sqrt(10^(params.snr_dB/10));
    noise = sigma_n * (randn(size(echo_atten)) + 1j*randn(size(echo_atten)));
    echo_noisy = echo_atten + noise;
    echo_signal = echo_atten;
end

function [echo_atten, echo_noisy] = generate_radar_echo_simple(doppler_history, params)
    N = size(doppler_history, 1);
    T = length(params.t);
    rcs_values = 0.2 + 0.8*rand(N, 1);
    echo_signal = zeros(1, T);
    if size(doppler_history,2) ~= T
        error('doppler_history length mismatch');
    end
    for i = 1:N
        phi0 = 2*pi*rand;
        signal_i = rcs_values(i) .* exp(1j * (2*pi * (doppler_history(i, :) .* params.t) + phi0));
        echo_signal = echo_signal + signal_i;
    end
    slant_range = 1000 + 50*sin(params.omega_main*params.t) - 5 * (params.t/params.sim_time);
    L_a = exp(-params.alpha_atten * slant_range/1000);
    echo_atten = echo_signal .* L_a;
    sigma_n = std(real(echo_atten));
    if sigma_n == 0
        sigma_n = 1e-12;
    end
    sigma_n = sigma_n / sqrt(10^(params.snr_dB/10));
    noise = sigma_n * (randn(size(echo_atten)) + 1j*randn(size(echo_atten)));
    echo_noisy = echo_atten + noise;
end

function lfm_sig = generate_lfm(t, f0, B, T)
    mu = B / T;
    lfm_sig = exp(1j * 2*pi * (f0.*t + 0.5*mu.*t.^2));
end

% （V2 中的 animate_rotor 被替换，不再在此处调用；Range-Doppler 动画将替代）
% 保留 animate_rotor 函数以防后续需要（不调用也不会影响）
function animate_rotor(positions, params)
    fig = figure('Name', '旋翼运动轨迹动画', 'NumberTitle', 'off');
    ax = axes('Parent', fig);
    hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
    xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
    title(ax, '直升机主旋翼运动轨迹（红色为桨叶散射点）');
    R = params.main_rotor_radius;
    theta = linspace(0, 2*pi, 200);
    xc = R*cos(theta); yc = R*sin(theta); zc = zeros(size(theta));
    plot3(ax, xc, yc, zc, 'k--', 'LineWidth', 0.8);
    hScatter = scatter3(ax, NaN, NaN, NaN, 36, 'r', 'filled');
    axis(ax, [-R*1.2, R*1.2, -R*1.2, R*1.2, -R*0.6, R*0.6]);
    view(ax, 3);
    hText = text(ax, -R, -R, R*0.55, '', 'FontSize', 10, 'Color', 'k');
    Tlen = length(params.t);
    step = max(1, floor(Tlen/2000));
    for t_idx = 1:step:Tlen
        x = squeeze(positions(:, 1, t_idx));
        y = squeeze(positions(:, 2, t_idx));
        z = squeeze(positions(:, 3, t_idx));
        valid = isfinite(x) & isfinite(y) & isfinite(z);
        if any(valid)
            set(hScatter, 'XData', x(valid), 'YData', y(valid), 'ZData', z(valid));
        else
            set(hScatter, 'XData', [], 'YData', [], 'ZData', []);
        end
        set(hText, 'String', sprintf('t = %.3f s', params.t(t_idx)));
        drawnow limitrate;
        pause(0.01);
        if ~isvalid(fig)
            break;
        end
    end
    hold(ax, 'off');
end

%% =========================================================================
% 3. 旋翼微动建模（与 V2 保持一致，只是点数改为适中）
%% =========================================================================
fprintf('正在进行旋翼微动建模...\n');

scatter_per_rotor = 5;           % 采用 V2 的 5 个点/桨叶采样（与原 V2 保持）
scatter_num = params.rotor_num * scatter_per_rotor;
positions = zeros(scatter_num, 3, length(params.t));
velocities = zeros(scatter_num, 3, length(params.t));

for t_idx = 1:length(params.t)
    t = params.t(t_idx);
    omega = params.omega_main;
    psi = omega * t;
    for rotor_idx = 1:params.rotor_num
        psi_rotor = psi + (rotor_idx-1)*2*pi/params.rotor_num;
        for point_idx = 1:scatter_per_rotor
            idx = (rotor_idx-1)*scatter_per_rotor + point_idx;
            r = params.main_rotor_radius * (0.2 + 0.8*(point_idx-1)/4);
            beta0 = deg2rad(mean(params.flap_angle_range));
            betac = deg2rad((params.flap_angle_range(2)-params.flap_angle_range(1))/4);
            betas = betac;
            beta = beta0 + betac*cos(psi_rotor) + betas*sin(psi_rotor);
            z_flap = r * beta;
            lag_dis = sim_lag_motion_scalar(t);
            x_lag = lag_dis * cos(psi_rotor);
            theta_twist = deg2rad(5) * sin(2*psi_rotor);
            z_twist = r * theta_twist * 0.1;
            x = (r + x_lag) * cos(psi_rotor);
            y = (r + x_lag) * sin(psi_rotor);
            z = z_flap + z_twist;
            positions(idx, :, t_idx) = [x, y, z];
            if t_idx > 1
                dt = params.t(t_idx) - params.t(t_idx-1);
                velocities(idx, :, t_idx) = (positions(idx, :, t_idx) - positions(idx, :, t_idx-1))/dt;
            else
                velocities(idx, :, t_idx) = [-omega*r*sin(psi_rotor), omega*r*cos(psi_rotor), 0];
            end
        end
    end
end

% 替换：不再调用简单的 animate_rotor（V2 原来第一幅图），而使用下面的 Range-Doppler 动画（V8 风格）
% animate_rotor(positions, params);

%% =========================================================================
% 4. 雷达回波生成与多普勒分析（与 V2 保持一致）
%% =========================================================================
fprintf('正在生成雷达回波与多普勒信号...\n');

doppler_history = zeros(scatter_num, length(params.t));
for t_idx = 1:length(params.t)
    vel_t = squeeze(velocities(:, :, t_idx));
    doppler_history(:, t_idx) = compute_doppler(vel_t, params);
end

[echo_clean, echo_noisy] = generate_radar_echo(doppler_history, params);

figure('Name', '雷达回波信号', 'NumberTitle', 'off');
subplot(2,1,1);
plot(params.t, real(echo_clean));
title('无噪声雷达回波（实部）');
xlabel('时间 (s)'); ylabel('幅度'); grid on;

subplot(2,1,2);
plot(params.t, real(echo_noisy));
title(['含噪声雷达回波（SNR=', num2str(params.snr_dB), ' dB，实部）']);
xlabel('时间 (s)'); ylabel('幅度'); grid on;

figure('Name', '多普勒频移历程', 'NumberTitle', 'off');
mean_dop = mean(doppler_history, 1);
plot(params.t, mean_dop);
title('旋翼散射点平均多普勒频移');
xlabel('时间 (s)'); ylabel('多普勒频移 (Hz)'); grid on;
fmax_tmp = max(abs(mean_dop));
if fmax_tmp == 0
    fmax_tmp = 1;
end
ylim([-1.2*fmax_tmp, 1.2*fmax_tmp]);

%% =========================================================================
% 5. Range-Doppler 动画（替代原 V2 的第一幅轨迹图；依据 V8 的实现做必要适配）
%    保持其它 V2 图、分析与特征提取不变
%    注意：这里将 Range-Doppler 的实现封装为函数，真正运行放在脚本末尾以在最后显示
%% =========================================================================
fprintf('已准备 Range-Doppler 动画函数，实际播放将放到脚本末尾显示...\n');

% 将 Range-Doppler 动画实现封装成函数，稍后在脚本末尾调用
function run_range_doppler_animation(echo_noisy, positions, params, tx)
    % 准备参数
    frame_duration = 1/tx.prf;
    samples_per_frame = max(4, round(frame_duration * params.fs));    % 每帧样本数
    frame_hop = samples_per_frame;
    num_frames = floor(length(params.t) / frame_hop);

    range_grid = linspace(900, 1100, 80);
    nr = length(range_grid);
    nfft_rd = 256;
    f_dop = linspace(-params.fs/2, params.fs/2, nfft_rd);

    gif_filename = 'rotor_rd_updateInPlace.gif';
    if exist(gif_filename,'file'), delete(gif_filename); end

    cmap = hot(256);
    rotor_colors = lines(params.rotor_num);

    % Figure and axes (persistent-object approach)
    h_fig = figure('Name','Range-Doppler Animated','NumberTitle','off','Color',[0.12 0.12 0.12], 'Renderer','opengl');
    set(h_fig, 'Position', [100 100 1200 650]);

    % main axes (left)
    main_left = 0.05; main_width = 0.62; main_bottom = 0.06; main_height = 0.88;
    ax_main = axes('Parent', h_fig, 'Position',[main_left main_bottom main_width main_height], 'Color',[0.12 0.12 0.12]);
    hImg = imagesc(f_dop, range_grid, zeros(nr, length(f_dop)), 'Parent', ax_main);
    axis(ax_main,'xy');
    colormap(ax_main, cmap);
    c = colorbar('EastOutside');
    c.Color = [1 1 1];
    c.Label.String = 'Relative power (dB)';
    set(ax_main, 'XColor', [1 1 1], 'YColor', [1 1 1], 'Color',[0.12 0.12 0.12]);
    xlabel(ax_main,'Doppler (Hz)','Color',[1 1 1]);
    ylabel(ax_main,'Range (m)','Color',[1 1 1]);
    title_handle = title(ax_main, 'Range-Doppler','Color',[1 1 1]);
    caxis([-50 0]);

    % prepare polar axes on right (2x2)
    polar_left = main_left + main_width + 0.03;
    polar_width = 0.30; polar_height = 0.42; polar_gap = 0.03;
    polarHandles = gobjects(params.rotor_num,1);
    bladeHandles = cell(params.rotor_num,1);
    ptsHandles = gobjects(params.rotor_num,1);
    scatter_per_rotor = size(positions,1) / params.rotor_num;
    scatter_rotor_idx = repelem((1:params.rotor_num)', scatter_per_rotor);

    for rotor_idx = 1:params.rotor_num
        col = mod(rotor_idx-1,2);
        row = floor((rotor_idx-1)/2);
        left = polar_left + col*(polar_width/2 + 0.02);
        bottom = main_bottom + main_height - (row+1)*(polar_height) - row*polar_gap;
        polarHandles(rotor_idx) = polaraxes('Parent', h_fig, 'Position', [left bottom polar_width/2 polar_height], 'Color',[0.05 0.05 0.05]);
        hold(polarHandles(rotor_idx), 'on');
        blade_count_vis = params.rotor_num; % 用旋翼数作为可视叶片数（视觉效果）
        bh = gobjects(blade_count_vis,1);
        for b = 1:blade_count_vis
            bh(b) = polarplot(polarHandles(rotor_idx), [0 0], [0 1], 'Color',[0 0.6 1], 'LineWidth', 2);
        end
        bladeHandles{rotor_idx} = bh;
        ptsHandles(rotor_idx) = polarplot(polarHandles(rotor_idx), NaN, NaN, '.', 'Color',[1 0.85 0], 'MarkerSize', 8);
        rlim(polarHandles(rotor_idx), [0 params.main_rotor_radius*1.2]);
        polarHandles(rotor_idx).ThetaZeroLocation = 'right';
        polarHandles(rotor_idx).ThetaDir = 'counterclockwise';
        polarHandles(rotor_idx).ThetaColor = [1 1 1];
        polarHandles(rotor_idx).RColor = [1 1 1];
        title(polarHandles(rotor_idx), sprintf('Rotor %d', rotor_idx), 'Color', rotor_colors(rotor_idx,:));
        hold(polarHandles(rotor_idx), 'off');
    end

    drawnow;

    % Animation loop (update-in-place)
    fprintf('Animating (update-in-place) Range-Doppler ...\n');
    first_frame_written = false;
    dB_min = -50;

    for fr = 1:num_frames
        idx_start = (fr-1)*frame_hop + 1;
        idx_end = min(idx_start + samples_per_frame - 1, length(params.t));
        if idx_end - idx_start + 1 < 4, continue; end
        t_frame = params.t(idx_start:idx_end);

        % compute rd_mat for this frame
        rd_mat = zeros(nr, nfft_rd);
        slant_range_frame = 1000 + 50*sin(params.omega_main * mean(t_frame)) - 5*(mean(t_frame)/params.sim_time);
        for ir = 1:nr
            R = range_grid(ir);
            range_gain = exp(-((R - slant_range_frame)/10).^2);
            % 使用 echo_noisy 的这个时间段并用 range_gain 作为幅度加权
            sig = echo_noisy(idx_start:idx_end) * range_gain;
            win = hann(length(sig)).';
            sigw = sig .* win;
            S = fftshift(abs(fft(sigw, nfft_rd)));
            rd_mat(ir, :) = S;
        end
        rd_db = 20*log10(rd_mat + eps);
        rd_db = rd_db - max(rd_db(:));
        rd_db = max(rd_db, dB_min);

        % update main image in-place
        set(hImg, 'CData', rd_db);
        set(title_handle, 'String', sprintf('Range-Doppler (Frame: %d / %d    Time: %.3f s)', fr, num_frames, params.t(idx_start)));

        % update polar axes data for each rotor without recreating axes
        time_center = mean(t_frame);
        mid_idx = round((idx_start + idx_end)/2);
        for rotor_idx = 1:params.rotor_num
            psi_rotor_center = params.omega_main * time_center + (rotor_idx-1)*2*pi/params.rotor_num;
            idxs = find(scatter_rotor_idx == rotor_idx);
            pts_xy = squeeze(positions(idxs, 1:2, mid_idx));
            if isempty(pts_xy)
                thetas = []; rs = [];
            else
                thetas = atan2(pts_xy(:,2), pts_xy(:,1));
                rs = sqrt(pts_xy(:,1).^2 + pts_xy(:,2).^2);
            end

            % update blades
            bh = bladeHandles{rotor_idx};
            blade_count_vis = numel(bh);
            for b = 1:blade_count_vis
                ang = psi_rotor_center + (b-1)*2*pi/blade_count_vis;
                set(bh(b), 'ThetaData', [ang ang], 'RData', [0 params.main_rotor_radius*1.05]);
            end
            % update scatter points
            set(ptsHandles(rotor_idx), 'ThetaData', thetas, 'RData', rs);

            % update polar title with angle
            set(get(polarHandles(rotor_idx),'Title'), 'String', sprintf('Rotor %d Angle: %.1f deg', rotor_idx, mod(rad2deg(psi_rotor_center),360)), 'Color', rotor_colors(rotor_idx,:));
        end

        drawnow limitrate;

        % capture frame and write to GIF
        frame = getframe(h_fig);
        im = frame2im(frame);
        [Amap, map_used] = rgb2ind(im, 256);
        if ~first_frame_written
            imwrite(Amap, map_used, gif_filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.05);
            first_frame_written = true;
        else
            imwrite(Amap, map_used, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
        end
    end

    fprintf('Range-Doppler 动画完成并保存为 %s\n', gif_filename);
end

%% =========================================================================
% 6. 时频分析（STFT + CWT）  —— 保持 V2 原样
%% =========================================================================
fprintf('正在进行时频分析...\n');
doppler_signal = mean(doppler_history, 1);

% STFT
window = hamming(256);
noverlap = 128;
nfft = 512;
[s_stft, f_stft, t_stft] = spectrogram(doppler_signal, window, noverlap, nfft, params.fs);

figure('Name', '时频分析结果', 'NumberTitle', 'off');
subplot(2,1,1);
imagesc(t_stft, f_stft, 20*log10(abs(s_stft)+eps));
axis xy;
colormap('jet'); colorbar;
title('微多普勒 STFT 谱图（dB）');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
ylim([0, min(max(f_stft), params.fs/2)]);

% CWT
[cwt_coeff, f_cwt] = cwt(doppler_signal, 'amor', params.fs);
pos_idx = f_cwt > 0;
f_cwt_pos = f_cwt(pos_idx);
cwt_coeff_pos = abs(cwt_coeff(pos_idx, :));

subplot(2,1,2);
Nt = length(params.t);
Ncwt = size(cwt_coeff_pos, 2);
if Ncwt ~= Nt
    t_cwt = linspace(params.t(1), params.t(end), Ncwt);
    [Tq, Fq] = meshgrid(t_cwt, f_cwt_pos);
    [Tt, Ft] = meshgrid(params.t, f_cwt_pos);
    cwt_interp = interp2(Tq, Fq, 20*log10(cwt_coeff_pos+eps), Tt, Ft);
else
    cwt_interp = 20*log10(cwt_coeff_pos+eps);
end
pcolor(params.t, f_cwt_pos, cwt_interp);
shading flat;
colormap('jet'); colorbar;
title('微多普勒 CWT 时频图（dB）');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
ylim([0, min(max(f_stft), params.fs/2)]);

%% =========================================================================
% 7. 特征提取与目标分类 —— 保持 V2 原样
%% =========================================================================
fprintf('正在进行特征提取与目标分类...\n');

[Pxx, f_period] = periodogram(doppler_signal, [], [], params.fs);
[~, idx_max] = max(Pxx);
fund_freq = f_period(idx_max);

% 带宽计算（优先使用 powerbw，否则用近似）
try
    bw = powerbw(Pxx, f_period);
catch
    cum = cumsum(Pxx);
    total = cum(end);
    low_idx = find(cum >= 0.25*total, 1);
    high_idx = find(cum >= 0.75*total, 1);
    if isempty(low_idx) || isempty(high_idx)
        bw = 0;
    else
        bw = f_period(high_idx) - f_period(low_idx);
    end
end

harmonic_energy = 0;
total_energy = sum(Pxx) + eps;
for n = 2:5
    harm_freq = n * fund_freq;
    harm_idx = find(f_period >= harm_freq-1 & f_period <= harm_freq+1);
    if ~isempty(harm_idx)
        harmonic_energy = harmonic_energy + sum(Pxx(harm_idx));
    end
end
harmonic_ratio = harmonic_energy / total_energy;

fprintf('\n=== 微多普勒特征提取结果 ===\n');
fprintf('1. 微动基频：%.6f Hz\n', fund_freq);
fprintf('2. 多普勒带宽：%.6f Hz\n', bw);
fprintf('3. 谐波能量占比：%.4f%%\n', harmonic_ratio*100);

% SVM 分类：使用 templateSVM 指定核函数，再传给 fitcecoc
rng(1);
heli_feat = [fund_freq + 0.1*randn(50,1), bw + 5*randn(50,1), harmonic_ratio+0.05*randn(50,1)];
drone_feat = [fund_freq*2 + 0.2*randn(50,1), bw*0.5 + 2*randn(50,1), harmonic_ratio*0.5+0.03*randn(50,1)];
X_train = [heli_feat; drone_feat];
Y_train = [repmat("Helicopter", 50, 1); repmat("Drone", 50, 1)];

% 使用 templateSVM 指定 KernelFunction
t = templateSVM('KernelFunction', 'rbf', 'Standardize', true);
mdl = fitcecoc(X_train, Y_train, 'Learners', t);

test_feat = [fund_freq, bw, harmonic_ratio];
pred_label = predict(mdl, test_feat);

figure('Name', '目标分类结果', 'NumberTitle', 'off');
gscatter(X_train(:,1), X_train(:,2), Y_train, 'rb', 'ox', 8); hold on;
plot(test_feat(1), test_feat(2), 'gs', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
title(['SVM 分类结果：测试样本预测为【', char(pred_label), '】']);
xlabel('微动基频 (Hz)'); ylabel('多普勒带宽 (Hz)');
legend('直升机训练集', '无人机训练集', '当前测试样本');
grid on; hold off;

fprintf('\n=== 目标分类结果 ===\n');
fprintf('当前仿真目标预测类别：%s\n', char(pred_label));



%% =========================================================================
% 8. 稳健参数估计 —— 替换原来的 Section 8
%% =========================================================================
fprintf('\n=== 参数估计结果 ===\n');

% 打印真实参数
fprintf('真实主旋翼转速: %.2f RPM\n', params.main_rotor_rpm);
fprintf('真实主旋翼半径: %.2f m\n', params.main_rotor_radius);
fprintf('真实桨叶数: %d\n', params.rotor_num);
fprintf('真实尾桨转速: %.2f RPM\n', params.tail_rotor_rpm);
fprintf('----------------------------------------\n');

% 1. 主旋翼转速 (RPM) —— 用自相关法估计基频
sig = doppler_signal - mean(doppler_signal);
[acf, lags] = xcorr(sig, 'coeff');
lags = lags(lags>0);
acf = acf(lags>0);
[~, loc] = max(acf(1:round(end/10)));   % 只看前 1/10 区间，避免伪峰
fund_freq_est = params.fs / lags(loc);
est_rpm = fund_freq_est * 60;
fprintf('估计主旋翼转速: %.2f RPM\n', est_rpm);

% 2. 主旋翼半径 —— 从 STFT 中找最大多普勒频移
[~, f_stft, t_stft, ps] = spectrogram(doppler_signal, hamming(512), 256, 1024, params.fs);
ps_db = 20*log10(abs(ps)+eps);
[~, idx_max] = max(ps_db(:));
[f_idx, ~] = ind2sub(size(ps_db), idx_max);
fmax_est = f_stft(f_idx);
R_est = (fmax_est * params.lambda) / (2 * 2*pi*fund_freq_est);
fprintf('估计主旋翼半径: %.2f m\n', R_est);

% 3. 桨叶数 —— 在功率谱中找基频谐波
[Pxx, f_period] = periodogram(doppler_signal, [], [], params.fs);
[pks, locs] = findpeaks(Pxx, 'MinPeakHeight', max(Pxx)*0.2);
harm_freqs = f_period(locs);
harm_diffs = diff(harm_freqs);
blade_count_est = round(mean(harm_diffs) / fund_freq_est);
fprintf('估计桨叶数: %d\n', blade_count_est);

% 4. 尾桨转速 —— 在高频段单独搜索
tail_band = f_period > 100;   % 100 Hz 以上
[~, tail_idx] = max(Pxx(tail_band));
if ~isempty(tail_idx)
    tail_freq = f_period(tail_band);
    tail_freq = tail_freq(tail_idx);
    tail_rpm_est = tail_freq * 60;
    fprintf('估计尾桨转速: %.2f RPM\n', tail_rpm_est);
else
    fprintf('估计尾桨转速: 未检测到明显峰值\n');
end



%% =========================================================================




fprintf('脚本运行完成！\n');

%% =========================================================================
% 最后：在所有分析和绘图完成后，再显示并播放 Range-Doppler 动画（将图放到最后显示）
% 注意：其余部分保持不变，Range-Doppler 动画在此处才真正运行并显示
%% =========================================================================
run_range_doppler_animation(echo_noisy, positions, params, tx);
