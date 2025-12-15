% »•–±¥¶¿Ì
Fs = 60e6;
Tw = 20e-6;
Bw = 20e6;
k = Bw/Tw;
t = 1/Fs:1/Fs:Tw;
fc = 10e9;
R =150;
c = 3e8;

Tx = exp(1j*2*pi*(fc+0.5*k*t).*t);
Rx = zeros(1,length(Tx));

tau = 2*R/c;

numDelayPoint = round(Fs*tau);

Rx(numDelayPoint+1:end) = Tx(1:(length(Tx)-numDelayPoint));
figure(1);
subplot(2,1,1)
plot(t,real(Tx))
title('Time Domain')

subplot(2,1,2)
plot(t,real(Rx))
title('Time Domain')

%%
afterMix = Rx.*conj(Tx);

deltaFreq = Fs/length(Tx);
xaxis = (-Fs/2:deltaFreq:Fs/2-deltaFreq)*c/2/k;
plot(xaxis,abs(fftshift(fft(afterMix))))