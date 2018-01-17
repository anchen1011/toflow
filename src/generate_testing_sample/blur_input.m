function blur_input(data_path, output_path)

scale_factor    = 4;

%% read input images
filenames   = dir(fullfile(data_path, '*.png'));
num_imgs    = length(filenames);
img_list    = cell(num_imgs, 1);
for iimg = 1 : num_imgs
  img_list{iimg} = im2double(imread(fullfile(data_path, filenames(iimg).name)));
  img_list{iimg} = imresize(img_list{iimg}, 1 / scale_factor);
end

for iimg = 1 : num_imgs
	img_list{iimg} = imresize(img_list{iimg}, scale_factor);
end

if ~exist(output_path, 'dir')
  mkdir(output_path);
end
for iimg = 1 : num_imgs
	imwrite(img_list{iimg}, [output_path '/im' sprintf('%04d',iimg) '.png'])
end

end
