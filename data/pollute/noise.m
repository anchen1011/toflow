function noise(data_path)

filenames   = dir(fullfile(data_path, '*.png'));
num_imgs    = length(filenames);
img_list    = cell(num_imgs, 1);
for iimg = 1 : num_imgs
    img_list{iimg} = im2uint8(imread(fullfile(data_path, filenames(iimg).name)));
	img_list{iimg} = addnoise(img_list{iimg});
	imwrite(img_list{iimg}, [data_path '/noisy/im' sprintf('%04d',iimg) '.png']);
end

end
