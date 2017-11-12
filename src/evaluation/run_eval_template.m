function [p, s, a] = run_eval_template(nim, gtdir, esdir, sampledirs, gtsuffix, essuffix)

imlist = cell(1, nim);
count = 0;
psnrAll = zeros(nim,1);
ssimAll = zeros(nim,1);
absAll = zeros(nim,1);

for i = 1:length(imlist)
  imname = imlist{i};
  imgt = im2double(imread(fullfile(gtdir, sampledirs{i}, gtsuffix)));
  imes = im2double(imread(fullfile(esdir, sampledirs{i}, essuffix)));
  [h,w,c] = size(imgt);
  if c == 1
    tmp = zeros(h,w,3);
    tmp(:,:,1) = imgt;
    tmp(:,:,2) = imgt;
    tmp(:,:,3) = imgt;
    imgt = tmp;
  end
  [hs,ws,cs] = size(imes);
  if cs == 1
    tmps = zeros(hs,ws,3);
    tmps(:,:,1) = imes;
    tmps(:,:,2) = imes;
    tmps(:,:,3) = imes;
    imes = tmps;
  end
  imgtv = reshape(imgt, [h*w,3]);
  imesv = reshape(imes, [h*w,3]);
  psnrAll(i) = psnr(imes, imgt);
  ssimAll(i) = ssim(imes, imgt);
  absAll(i) = mean2(abs(imes - imgt));
  fprintf('%d/%d\n',i,nim);
end

%%
p = mean(psnrAll);
s = mean(ssimAll);
a = mean(absAll);
fprintf('Mean PSRN: %f\n', p);
fprintf('Mean SSIM: %f\n', s);
fprintf('Mean Abs: %f\n', a);
