% close all; clear all; clc;
% path = 'D:\BCI_EEG\MelvinTestBenchData\edf1\MelvinEditted.csv';
% path = 'D:\BCI_EEG\BCIData\Eddo\EddoEditted.csv';
% path = 'D:\BCI_EEG\BCIData\Ammar\AmmarEditted.csv';
% path = 'D:\BCI_EEG\BCIData\Clement\ClementEditted.csv';
% path = 'D:\BCI_EEG\BCIData\Chris\ChrisEditted.csv';
% path = 'D:\BCI_EEG\BCIData\Yulia\YuliaEditted.csv';
path = 'F:\BCI_EEG\BCIData\LiCui\Licui0Editted.csv';
% path = 'D:\BCI_EEG\BCIData\LiCui\Licui1Editted.csv';
% path = 'D:\BCI_EEG\BCIData\LiCui\Licuinew0Editted.csv';
% path = 'D:\BCI_EEG\BCIData\LiCui\Licuinew1Editted.csv';
% path = 'F:\BCI_Data\zero\exp0\exp0editted.csv';

H = csvread(path);
raw = H(:,3:16);

% % doing the hanning windowing
fftlength = 256;
hanning = [1:fftlength]';
hanning_in = 2*pi()*(hanning - (fftlength+1)/2)/(fftlength+1);
hanning = (sin(hanning_in)./hanning_in).^2;
hanning = repmat(hanning,1,14);

f = [128/fftlength:128/fftlength:128];

thetaIndex = find(f>=4 & f<8);
alphaIndex = find(f>=8 & f<12);
lowBetaIndex = find(f>=12 & f<16);
highBetaIndex = find(f>=16 & f<25);
gammaIndex = find(f>=25 & f<40); 
totIndex = find(f>=6 & f<40);
outdata = [];

med = median(raw,2);
raw = raw - repmat(med,1,14);

% %denoise
for j=2:size(raw,1)
    del = raw(j,:)-raw(j-1,:);
    del = min(del,ones(1,14)*15);
    del = max(del,-ones(1,14)*15);
    raw(j,:) = raw(j-1,:)+del;
end

% % another filter - HPF
a = 0.0078125;
b = 0.9921875;

preVal = zeros(1,14);
eeg.filt = zeros(size(raw));

for j=1:size(raw,1)
    preVal = a*raw(j,:)+b*preVal;
    eeg.filt(j,:) = raw(j,:)-preVal;
end

eeg.theta      = [];
eeg.alpha      = [];
eeg.lowBeta    = [];
eeg.highBeta   = [];
eeg.gamma      = [];
eeg.tot        = [];
eeg.totmed     = [];

% % get the band signals
for k = fftlength:32:size(eeg.filt,1)
    spectrum         = fft(eeg.filt(k-fftlength+1:k,:) .* hanning);
    spectrum         = sqrt(spectrum .* conj(spectrum));
    eeg.theta        = [eeg.theta; k sum(spectrum(thetaIndex,:))];
    eeg.alpha        = [eeg.alpha; k sum(spectrum(alphaIndex,:))];
    eeg.lowBeta      = [eeg.lowBeta; k sum(spectrum(lowBetaIndex,:))];
    eeg.highBeta     = [eeg.highBeta; k sum(spectrum(highBetaIndex,:))];
    eeg.gamma        = [eeg.gamma; k sum(spectrum(gammaIndex,:))];
    eeg.tot          = [eeg.tot; k sum(spectrum(totIndex,:))];
end

% % set the time and initialize the channel loop value
timeRange = 1:237;
channel = 2;

% plot the bands for 14 channels
while channel<=15
    figure;
    plot(eeg.theta(timeRange,1),eeg.theta(timeRange,channel),'b');
    hold on
    plot(eeg.alpha(timeRange,1),eeg.alpha(timeRange,channel),'g');
    hold on
    plot(eeg.lowBeta(timeRange,1),eeg.lowBeta(timeRange,channel),'c');
    hold on
    plot(eeg.highBeta(timeRange,1),eeg.highBeta(timeRange,channel),'m');
    hold on
    plot(eeg.gamma(timeRange,1),eeg.gamma(timeRange,channel),'y');
    like = int2str(channel);
    title('Channel');
    legend('theta','alpha','lowBeta','highBeta','gamma');
    xlabel('Sample Number (Start of FFT; 32 sample steps)');
    ylabel('Amplitude (squared?)');
    ylim([0 3500]);;
    grid on;
    channel=channel+1;
end


% Calculate the bandpower (one value per band per channel)
for channel=2:15
    p_theta(channel) = bandpower(eeg.theta(timeRange,channel));
    p_alpha(channel) = bandpower(eeg.alpha(timeRange,channel));
    p_lowBeta(channel) = bandpower(eeg.lowBeta(timeRange,channel));
    p_highBeta(channel) = bandpower(eeg.highBeta(timeRange,channel));
    p_gamma(channel) = bandpower(eeg.gamma(timeRange,channel));
end

% plot the powerbands per band
figure;
subplot(5,1,1);bar(1:14,p_theta(2:15));title('Theta');grid on;
subplot(5,1,2);bar(1:14,p_alpha(2:15));title('Alpha');grid on;
subplot(5,1,3);bar(1:14,p_lowBeta(2:15));title('lowBeta');grid on;
subplot(5,1,4);bar(1:14,p_highBeta(2:15));title('highBeta');grid on;
subplot(5,1,5);bar(1:14,p_gamma(2:15));title('Gamma');grid on;

% % check the maximum value for low Beta and high Beta bands
% [max_value_low,max_index_low] = max(p_lowBeta(2:15));
% [max_value_high,max_index_high] = max(p_highBeta(2:15));

% % doing the cross power spectral density
% [pxy,F] = cpsd(eeg.theta(timeRange,3),eeg.alpha(timeRange,3),[],[],[],128);
% figure;
% plot(F,angle(pxy));
% [acor,lag] = xcorr(eeg.lowBeta(timeRange,3),eeg.highBeta(timeRange,3));

% % Trying the magnitude-squared coherence estimate
% for k=2:15
%     [cxy,fc] = mscohere(eeg.alpha(timeRange,k),eeg.lowBeta(timeRange,k));
%     figure;
%     plot(fc/pi,cxy);title('Alpha-lowBeta Coherence 0');grid on;
% end

% % plot the raw Beta waves for comparison
% figure;
% plot(eeg.lowBeta(timeRange,1),eeg.lowBeta(timeRange,3));hold on;
% plot(eeg.highBeta(timeRange,1),eeg.highBeta(timeRange,3));legend('lowBeta','highBeta');grid on;

% Applyting PCA
[pca_ceoff,pca_score, pca_latent] = pca(eeg.lowBeta(timeRange,9));
figure;
plot(eeg.lowBeta(timeRange,1),eeg.lowBeta(timeRange,9)); hold on;
plot(eeg.lowBeta(timeRange,1),pca_score);legend('raw','pca score');
data = eeg.lowBeta(timeRange,9);

% Write into text file for training and testing data
file = fopen('F:\\BCI_EEG\\BCIData\\LiCui\\train0.txt','wt');
fprintf(file,'%f\n',data(1:200));
fclose(file);
file = fopen('F:\\BCI_EEG\\BCIData\\LiCui\\test0.txt','wt');
fprintf(file,'%f\n',data(201:237));
fclose(file);

%Applying ELM
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM('F:\\BCI_EEG\\BCIData\\LiCui\\train0.txt','F:\\BCI_EEG\\BCIData\\LiCui\\test0.txt',1,100000,'sin');



























