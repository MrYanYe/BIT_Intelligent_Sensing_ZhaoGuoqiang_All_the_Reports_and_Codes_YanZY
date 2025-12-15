function helperslexSaveRadarIQDataToFileAcquiredViaDemoradParam
% This function is only in support of slexSaveRadarIQDataToFileAcquiredViaDemorad. 

%   Copyright 2019 The MathWorks, Inc.

% Radar parameters
paramDemorad.startFrequency = 24e9;
paramDemorad.stopFrequency = 24.25e9;
paramDemorad.centerFrequency = 24.125e9; 
paramDemorad.rampTime = 264e-6;
paramDemorad.pulseRepetitionInterval = 280e-6;
bandwidth = paramDemorad.stopFrequency - paramDemorad.startFrequency;
paramDemorad.sweepSlope = bandwidth/paramDemorad.rampTime;

% Hardware configuration
paramDemorad.transmitPower = 100;
paramDemorad.acquisitionTime = 20;
paramDemorad.numChirps = 32;
paramDemorad.numSamples = 256;

% Processing parameters
lambda = physconst('lightspeed')/paramDemorad.centerFrequency;
maxSpeed = lambda/(2*paramDemorad.pulseRepetitionInterval);
paramDemorad.radialSpeedLim = maxSpeed * [-0.5, 0.5];
paramDemorad.numPulses = 30;
paramDemorad.dopplerNFFT = 256;
paramDemorad.rangeNFFT = 256;
paramDemorad.spatialTaper = ones(paramDemorad.numSamples*paramDemorad.numPulses,1)*taylorwin(4,4,-40).';

% Visualization parameters
paramDemorad.noiseFloor = -10;
maxRange = 1e6*physconst('lightspeed') / ...
  (2*paramDemorad.sweepSlope);
maxRangeLims = linspace(0,maxRange,paramDemorad.rangeNFFT);
paramDemorad.rangeLim = [0 20]; % desired range limits
[~,paramDemorad.rangeIdx] = min(abs(maxRangeLims-20));

% Constants
paramDemorad.numElements = 4; 
paramDemorad.receiveElementSpacing = 0.0062;
paramDemorad.sampleRate = 1000000;

% Metadata
metadata.rampTime = paramDemorad.rampTime;
metadata.pulseRepititionInterval = paramDemorad.pulseRepetitionInterval;
metadata.startFrequency = paramDemorad.startFrequency;
metadata.stopFrequency = paramDemorad.stopFrequency;
metadata.sweepSlope = paramDemorad.sweepSlope;

paramDemorad.metadata = metadata;

% Set in base workspace
assignin('base','paramDemorad',paramDemorad);