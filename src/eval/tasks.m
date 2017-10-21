function [n, tasklist] = tasks(output_root)

tasklist = cell(4, 7);
n = 0;

n = n+1;
tasklist{n,1} = 'denoise';
tasklist{n,2} = 7824;
tasklist{n,3} = '../../data/vimeo_sep';
tasklist{n,4} = fullfile(output_root, 'denoise');
tasklist{n,5} = 'sep';
tasklist{n,6} = 'im4.png';
tasklist{n,7} = '0004.png';

n = n+1;
tasklist{n,1} = 'deblock';
tasklist{n,2} = 7824;
tasklist{n,3} = '../../data/vimeo_sep';
tasklist{n,4} = fullfile(output_root, 'deblock');
tasklist{n,5} = 'sep';
tasklist{n,6} = 'im4.png';
tasklist{n,7} = '0001.png';

n = n+1;
tasklist{n,1} = 'sr';
tasklist{n,2} = 7824;
tasklist{n,3} = '../../data/vimeo_sep';
tasklist{n,4} = fullfile(output_root, 'sr');
tasklist{n,5} = 'sep';
tasklist{n,6} = 'im4.png';
tasklist{n,7} = '0001.png';

n = n+1;
tasklist{n,1} = 'interp';
tasklist{n,2} = 3782;
tasklist{n,3} = '../../data/vimeo_tri';
tasklist{n,4} = fullfile(output_root, 'interp');
tasklist{n,5} = 'tri';
tasklist{n,6} = 'im2.png';
tasklist{n,7} = '0001.png';

n
end
