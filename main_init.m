%%% MAIN FUNCTION %%%

%% Initialization
clear all; % Important to run clear all before to not save other variables mistakenly
% Run Initialization function
[ship_masks,cameraParams,T_ortho,T_align,imageView,panoView,pixel_loc,blending_masks] = initialization();

% Save Initialization outputs
path_output = uigetdir(pwd,'Select directory to save outputs of initialization function...');
save(fullfile(path_output,'initialization_outputs_new99.mat'));

%% MakePanorama
% Load initialization outputs
% [path_output,file_output] = uigetfile(pwd,'Select outputs of initialization function...');
% load(fullfile(file_output,path_output));
% % %   
% % % % Set path for input images -> single Frame
% path_inputimages_single = uigetdir('Select directory for input imageset...');
% % Load input images
% imageSets = imageSet(fullfile(path_inputimages_single));
% % 
% %%%% OR %%%%
% 
% % Set path for input images -> multiple Frames
% path_inputimages_multi = uigetdir(pwd,'Select directory for input multiple imagesets...');
% % Load input images
% imageSets = imageSet(path_inputimages_multi,'recursive');
%  
% % Run MakePanorama function
[IceConST,IceConKM,IceFloeDis] = makePanorama(imageSets,ship_masks,0,cameraParams,T_ortho,imageView,T_align,panoView,blending_masks,pixel_loc);
%  
