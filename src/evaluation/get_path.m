function v = get_path(arg)

if strcmp(arg, 'tri')
	v = get_path_tri();
end

if strcmp(arg, 'sep')
	v = get_path_sep();
end

end
