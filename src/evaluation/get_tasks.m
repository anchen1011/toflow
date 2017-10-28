function [n, tasklist] = get_tasks(output_root)

tasklist = cell(4, 7);
n = 0;


% Comment/Uncomment the whole block to disable/enable a evaluation task
n = n+1;
tasklist{n,1} = 'denoise';                          % Task name
tasklist{n,2} = 7824;                               % Number of samples
tasklist{n,3} = '../../data/vimeo_sep';             % Target root
tasklist{n,4} = fullfile(output_root, 'denoise');   % Output root
tasklist{n,5} = 'sep';                              % Subpaths class (corresponding to the suffix after get_path)
tasklist{n,6} = 'im4.png';                          % The name of target file under the target path
tasklist{n,7} = 'out.png';                          % The name of output file under the target path


n = n+1;
tasklist{n,1} = 'deblock';
tasklist{n,2} = 7824;
tasklist{n,3} = '../../data/vimeo_sep';
tasklist{n,4} = fullfile(output_root, 'deblock');
tasklist{n,5} = 'sep';
tasklist{n,6} = 'im4.png';
tasklist{n,7} = 'out.png';


n = n+1;
tasklist{n,1} = 'sr';
tasklist{n,2} = 7824;
tasklist{n,3} = '../../data/vimeo_sep';
tasklist{n,4} = fullfile(output_root, 'sr');
tasklist{n,5} = 'sep';
tasklist{n,6} = 'im4.png';
tasklist{n,7} = 'out.png';


n = n+1;
tasklist{n,1} = 'interp';
tasklist{n,2} = 3782;
tasklist{n,3} = '../../data/vimeo_tri';
tasklist{n,4} = fullfile(output_root, 'interp');
tasklist{n,5} = 'tri';
tasklist{n,6} = 'im2.png';
tasklist{n,7} = 'out.png';

n
end
