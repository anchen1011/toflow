function tasklist = get_task(output_dir, target_dir, task)

tasklist = cell(1, 7);

tasklist{1,1} = task;                               % Task name
tasklist{1,2} = 7824;                               % Number of samples
tasklist{1,3} = target_dir;                         % Target dir
tasklist{1,4} = output_dir;                         % Output dir
tasklist{1,5} = 'sep';                              % Subpaths class (corresponding to the suffix after get_path)
tasklist{1,6} = 'im4.png';                          % The name of target file under the target path
tasklist{1,7} = 'out.png';                          % The name of output file under the target path
if strcmp(task, 'interp')
  tasklist{1,2} = 3782;
  tasklist{1,5} = 'tri';
  tasklist{1,6} = 'im2.png';
end

end
