function project_root = set_path()
%SET_PATH Add the complete OR-ANOVA repository to the MATLAB path.

core_folder = fileparts(mfilename('fullpath'));
project_root = fileparts(core_folder);
addpath(genpath(project_root));

fprintf('Project path is ready.\n');
end
