% RadarAnimate_FMCW_sector_improved.m
% 改进说明：
% - 对 dB 映射做自适应压缩和 gamma 校正以避免亮点饱和占比
% - 对 RD/RA 图像分别做二维高斯平滑与孤立峰抑制提高视觉连贯性
% - 保持原有数据排列兼容和多天线波束形成逻辑
clear; close all; clc;

%% ========== 用户配置（请根据实验实际值修改） ==========
inputFolder = 'save_data\FMCW\DTargets_FMCW';
% inputFolder = 'save_data\FMCW\STarget_FMCW';    % 数据文件夹
% inputFolder = 'save_data\FMCW\BG_FMCW';    % 数据文件夹

filePattern = 'declutterIQ_frame*.mat';

% FMCW 物理参数（请填写准确值）
fc    = 100e9;
B     = 100e6;
Tchirp= 60e-6;
fs    = 1e6;
c     = 3e8;
k     = B / Tchirp;
lambda = c / fc;

% 处理与显示参数
numRangeFFT    = 512;
numDopplerFFT  = 128;
numAngleFFT    = 256;
maxRangeDisplay = 20;
velocityAxisLimit = 10;
angleExtentDeg = 60;
colormapChoice = parula;
dbRange = [-100 -75 ];      % 扩大上限使亮点不被过早截断（可调）
pausePerFrame = 0.03;

% 波束形成选项
doBeamform = true;
spatialWindowType = 'taylor';
taylorSLL = -40;

% 清晰度/平滑/对比度控制（用于改善亮点与细节）
gammaRD = 0.9;            % Range-Doppler gamma 校正（<1 提升暗部）
gammaRA = 0.9;            % Range-Angle gamma 校正
gaussSigmaRD = 1.0;       % RD 平滑高斯核标准差（像素）
gaussSigmaRA = 1.2;       % RA 平滑高斯核标准差（像素）
peakSuppressFactor = 0.6; % 峰值抑制系数（0-1，越小抑制越强）
localContrastClip = 0.02; % 局部对比度裁剪（0~0.1）

%% ========== 读取文件列表并排序 ==========
files = dir(fullfile(inputFolder, filePattern));
if isempty(files)
    error('未找到匹配的文件：%s/%s', inputFolder, filePattern);
end
[~, idx] = sort({files.name});
files = files(idx);

%% ========== 预计算频率/距离/速度/角度轴 ==========
numRangeBins = numRangeFFT/2;
freqBins = (0:numRangeBins-1) * (fs / numRangeFFT);
ranges = (c .* freqBins) ./ (2 * k);

maxRangeIdx = find(ranges <= maxRangeDisplay, 1, 'last');
if isempty(maxRangeIdx)
    maxRangeIdx = numRangeBins;
end
rangesDisplay = ranges(1:maxRangeIdx);

Fs_d = 1 / Tchirp;
fdBins = linspace(-Fs_d/2, Fs_d/2, numDopplerFFT);
velAxis = fdBins * (lambda / 2);

angleAxisDeg = linspace(-angleExtentDeg, angleExtentDeg, numAngleFFT);
angleAxisRad = deg2rad(angleAxisDeg);

%% ========== Cartesian 网格为右侧扇形插值准备 ==========
xLim = 15;
xVec = linspace(-xLim, xLim, 300);
yVec = linspace(0, maxRangeDisplay, 300);
[Xgrid, Ygrid] = meshgrid(xVec, yVec);
Rgrid = sqrt(Xgrid.^2 + Ygrid.^2);
ThetaGrid = atan2(Xgrid, Ygrid);
ThetaGridDeg = rad2deg(ThetaGrid);
ThetaMask = (ThetaGridDeg >= -angleExtentDeg) & (ThetaGridDeg <= angleExtentDeg) & (Rgrid <= maxRangeDisplay);

%% ========== 初始化图窗与占位图像 ==========
figure('Name','Range-Doppler and Range-Angle Animation (Improved)','Color','w','Position',[200 150 1200 520]);
colormap(colormapChoice);

hAx1 = subplot(1,2,1);
hImgRD = imagesc(velAxis, rangesDisplay, ones(numel(rangesDisplay), numel(velAxis)) * dbRange(1));
axis xy;
xlabel('Velocity (m/s)'); ylabel('Range (m)');
title('Range - Doppler Map');
xlim([-velocityAxisLimit velocityAxisLimit]); ylim([0 maxRangeDisplay]);
cbar1 = colorbar; cbar1.Label.String = 'Power (dB)';
caxis(dbRange);

hAx2 = subplot(1,2,2);
hImgRA = imagesc(xVec, yVec, ones(numel(yVec), numel(xVec)) * dbRange(1));
axis xy; axis equal;
xlabel('x (m)'); ylabel('y (m)');
title('Range - Angle Map (sector)');
xlim([-xLim xLim]); ylim([0 maxRangeDisplay]);
cbar2 = colorbar; cbar2.Label.String = 'Power (dB)';
caxis(dbRange);

drawnow;

%% ========== 生成高斯核工具函数句柄 ==========
% 生成二维高斯核（可用于 imfilter）
function h = gauss2d(sigma)
    sz = max(3,ceil(6*sigma));
    if mod(sz,2)==0, sz = sz+1; end
    [X,Y] = meshgrid(linspace(-floor(sz/2),floor(sz/2),sz));
    h = exp(-(X.^2+Y.^2)/(2*sigma^2));
    h = h / sum(h(:));
end
gaussKernelRD = gauss2d(gaussSigmaRD);
gaussKernelRA = gauss2d(gaussSigmaRA);

%% ========== 主循环：逐文件读取并更新显示 ==========
for k = 1:numel(files)
    fname = fullfile(files(k).folder, files(k).name);
    S = load(fname);
    varnames = fieldnames(S);
    iqVar = '';
    typical = {'declutterIQ','IQ','iq','data','iqData','frameData','rawIQ'};
    for vn = 1:numel(varnames)
        if any(strcmpi(varnames{vn}, typical))
            iqVar = varnames{vn};
            break;
        end
    end
    if isempty(iqVar)
        for vn = 1:numel(varnames)
            if isnumeric(S.(varnames{vn}))
                iqVar = varnames{vn};
                break;
            end
        end
    end
    if isempty(iqVar)
        warning('文件 %s 未找到数字数组，跳过', files(k).name);
        continue;
    end
    iq = S.(iqVar);

    %% ====== 兼容常见的数据排列 ======
    sz = size(iq);
    nd = ndims(iq);
    if nd == 2
        fastLen = sz(1);
        numChirps = sz(2);
        numAntennas = 1;
        iq = reshape(iq, fastLen, numChirps, 1);
    elseif nd == 3
        d1 = sz(1); d2 = sz(2); d3 = sz(3);
        if d2 <= 64 && d3 > 64
            iq = permute(iq, [1 3 2]);
        elseif d3 <= 64 && d2 > 64
            % keep
        else
            if d3 <= 64 && d2 > 64
                % keep
            elseif d2 <= 64 && d3 <= 64 && d2 ~= d3
                if d2 < d3
                    iq = permute(iq, [1 3 2]);
                else
                    % keep
                end
            else
                iq = permute(iq, [1 3 2]);
            end
        end
        [fastLen, numChirps, numAntennas] = size(iq);
    else
        warning('IQ 维度不支持：%s，跳过', files(k).name);
        continue;
    end
    [fastLen, numChirps, numAntennas] = size(iq);

    %% ====== Range FFT（沿 fast-time），仅取正频率部分） ======
    rngFFT = fft(iq, numRangeFFT, 1);
    rngFFT = rngFFT(1:numRangeFFT/2, :, :);
    rngFFT = rngFFT(1:maxRangeIdx, :, :);
    R = size(rngFFT,1);

    %% ====== Doppler processing ======
    rdData = zeros(R, numDopplerFFT);
    win_chirp = hann(numChirps);
    for r = 1:R
        slowMat = squeeze(rngFFT(r,:,:));
        if numAntennas == 1
            slowVec = slowMat;
        else
            if size(slowMat,2) == numAntennas
                slowVec = sum(slowMat,2);
            else
                slowVec = squeeze(mean(slowMat,2));
            end
        end
        slowVec = slowVec(:);
        if length(slowVec) ~= numChirps
            slowVec = slowVec(1:numChirps);
        end
        dop = fftshift(fft(slowVec .* win_chirp, numDopplerFFT));
        rdData(r, :) = abs(dop).';
    end

    % 转为功率 dB
    P_rd = (rdData).^2;
    db_rd = 10 * log10(max(P_rd, 1e-12));
    db_rd(db_rd < dbRange(1)) = dbRange(1);
    db_rd(db_rd > dbRange(2)) = dbRange(2);

    %% ====== 对 RD 做对比度压缩与平滑处理 ======
    % 归一化到 [0,1]
    db_rd_norm = (db_rd - dbRange(1)) / (dbRange(2) - dbRange(1));
    db_rd_norm = max(0, min(1, db_rd_norm));
    % gamma 校正提升细节
    db_rd_gamma = db_rd_norm .^ gammaRD;
    % 局部对比度裁剪（简单版：线性剪裁高亮区域）
    db_rd_clipped = db_rd_gamma;
    % 峰值抑制：减少局部孤立大值的影响
    localMax = imdilate(db_rd_clipped, strel('disk',2));
    suppressMask = db_rd_clipped > (localMax * (1 - peakSuppressFactor) + db_rd_clipped * peakSuppressFactor);
    db_rd_clipped(suppressMask) = db_rd_clipped(suppressMask) * 0.95 + localMax(suppressMask) * 0.05;
    % 空间平滑（高斯）
    db_rd_smooth = imfilter(db_rd_clipped, gaussKernelRD, 'replicate');
    % 反映回 dB 量纲
    db_rd_final = db_rd_smooth * (dbRange(2) - dbRange(1)) + dbRange(1);

    %% ====== Angle processing ======
    if numAntennas > 1 && doBeamform
        raData = zeros(R, numAngleFFT);
        switch lower(spatialWindowType)
            case 'taylor'
                w_ant = taylorwin(numAntennas, 4, taylorSLL).';
            case 'hann'
                w_ant = hann(numAntennas).';
            case 'none'
                w_ant = ones(1, numAntennas);
            otherwise
                w_ant = hann(numAntennas).';
        end
        for r = 1:R
            snapshot = squeeze(rngFFT(r,1,:));
            if size(snapshot,1) ~= numAntennas
                snapshot = snapshot(:);
            end
            if length(snapshot) ~= numAntennas
                tmp = squeeze(rngFFT(r,:,:));
                if size(tmp,2) == numAntennas
                    snapshot = mean(tmp,1).';
                else
                    snapshot = zeros(numAntennas,1);
                end
            end
            snapshot_win = snapshot(:).' .* w_ant;
            angSpec = fftshift(fft(snapshot_win, numAngleFFT));
            raData(r, :) = abs(angSpec);
        end
        P_ra = (raData).^2;
        db_ra = 10 * log10(max(P_ra, 1e-12));
        db_ra(db_ra < dbRange(1)) = dbRange(1);
        db_ra(db_ra > dbRange(2)) = dbRange(2);
    else
        if numAntennas > 1 && ~doBeamform
            raData = zeros(R, numAngleFFT);
            for r = 1:R
                snapshot = squeeze(rngFFT(r,1,:));
                snapshot = snapshot(:);
                tmp = zeros(1, numAngleFFT);
                tmp(1:min(length(snapshot),numAngleFFT)) = snapshot(1:min(length(snapshot),numAngleFFT)).';
                angSpec = fftshift(fft(tmp));
                raData(r,:) = abs(angSpec);
            end
            P_ra = (raData).^2;
            db_ra = 10 * log10(max(P_ra, 1e-12));
            db_ra(db_ra < dbRange(1)) = dbRange(1);
            db_ra(db_ra > dbRange(2)) = dbRange(2);
        else
            db_ra = ones(R, numAngleFFT) * dbRange(1);
        end
    end

    %% ====== 对 RA 做压缩与平滑处理（在 (range,angle) 域） ======
    db_ra_norm = (db_ra - dbRange(1)) / (dbRange(2) - dbRange(1));
    db_ra_norm = max(0, min(1, db_ra_norm));
    db_ra_gamma = db_ra_norm .^ gammaRA;
    % 峰值抑制（按行处理以抑制孤立角度突变）
    for rr = 1:size(db_ra_gamma,1)
        row = db_ra_gamma(rr,:);
        localMx = imdilate(row, strel('line',5,0));
        mask = row > (localMx * (1 - peakSuppressFactor) + row * peakSuppressFactor);
        row(mask) = row(mask)*0.9 + localMx(mask)*0.1;
        db_ra_gamma(rr,:) = row;
    end
    % 空间（range-angle）平滑
    db_ra_smoothed = imfilter(db_ra_gamma, gaussKernelRA, 'replicate');
    db_ra_final = db_ra_smoothed * (dbRange(2) - dbRange(1)) + dbRange(1);

    %% ====== 更新左图：Range-Doppler ==========
    set(hImgRD, 'XData', velAxis, 'YData', rangesDisplay(1:R), 'CData', db_rd_final);
    title(hAx1, sprintf('Range - Doppler Map (Frame %d, up to %d m)', k, maxRangeDisplay));

    %% ====== 将 (range,angle) 的 db_ra 映射到 Cartesian 并更新右图 ==========
    [AngleGridDeg_source, RangeGrid_source] = meshgrid(angleAxisDeg, rangesDisplay(1:R));
    srcX = RangeGrid_source(:) .* sind(AngleGridDeg_source(:));
    srcY = RangeGrid_source(:) .* cosd(AngleGridDeg_source(:));
    srcV = db_ra_final(:);

    tgtX = Xgrid(:);
    tgtY = Ygrid(:);
    validMask = ThetaMask(:);

    % scatteredInterpolant 插值（对有效区域）
    F = scatteredInterpolant(srcX, srcY, srcV, 'linear', 'none');
    tgtV_all = nan(size(tgtX));
    if any(validMask)
        tgtV_all(validMask) = F(tgtX(validMask), tgtY(validMask));
    end
    tgtV_all_filled = tgtV_all;
    tgtV_all_filled(isnan(tgtV_all_filled)) = dbRange(1);

    db_ra_cart = reshape(tgtV_all_filled, size(Xgrid));
    % 对 Cartesian 结果再做小幅平滑去掉插值伪影
    db_ra_cart_smooth = imfilter((db_ra_cart - dbRange(1)) / (dbRange(2)-dbRange(1)), gaussKernelRA, 'replicate');
    db_ra_cart_final = db_ra_cart_smooth * (dbRange(2)-dbRange(1)) + dbRange(1);

    set(hImgRA, 'XData', xVec, 'YData', yVec, 'CData', db_ra_cart_final);
    title(hAx2, sprintf('Range - Angle Map (Frame %d, up to %d m)', k, maxRangeDisplay));

    % 绘制扇形边界（清晰但不过分显眼）
    hold(hAx2, 'on');
    th1 = deg2rad(-angleExtentDeg); th2 = deg2rad(angleExtentDeg);
    rEdge = linspace(0, maxRangeDisplay, 200);
    xEdge1 = rEdge .* sin(th1); yEdge1 = rEdge .* cos(th1);
    xEdge2 = rEdge .* sin(th2); yEdge2 = rEdge .* cos(th2);
    old1 = findobj(hAx2,'Tag','sectorEdge1'); old2 = findobj(hAx2,'Tag','sectorEdge2');
    if ~isempty(old1), delete(old1); end
    if ~isempty(old2), delete(old2); end
    plot(hAx2, xEdge1, yEdge1, 'w-', 'LineWidth', 0.8, 'Tag', 'sectorEdge1');
    plot(hAx2, xEdge2, yEdge2, 'w-', 'LineWidth', 0.8, 'Tag', 'sectorEdge2');
    hold(hAx2, 'off');

    drawnow;
    pause(pausePerFrame);
end

disp('播放完成');
