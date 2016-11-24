function bestCaseOrientation = Optimization()

% Load initialization outputs
[path_output,file_output] = uigetfile('Select outputs of initialization function...');
load(fullfile(file_output,path_output));

% Set path for input images -> single Frame
path_inputimages_single = uigetdir('Select directory for input imageset...');
% Load input images
imageSets = imageSet(fullfile(path_inputimages_single));

% Read Images
for j=1:6
    k = -1*(-7+j); % Oden images are clockwise, not counter-clockwise
    I{j} = read(imageSets, k);
end

% Masking out ship
I{3} = masking_ship(I{3},ship_masks{3},'ShowMessages','on','ShowImages','off');
I{4} = masking_ship(I{4},ship_masks{4},'ShowMessages','on','ShowImages','off');

% Undistort Images
I{3} = image_undistortion(I{3},cameraParams,'ShowMessages','on');
I{4} = image_undistortion(I{4},cameraParams,'ShowMessages','on');

% Resize and make BW to improve Speed
Io{3} = imresize(rgb2gray(I{3}),1); 
Io{4} = imresize(rgb2gray(I{4}),1);

% Define Camera Orientations
camOrientation{3} = [+45 0 0];
camOrientation{4} = [+45 0 0];

% Define Intervals and steps to try out for optimization
n_N = -2.5; % Start value of orientation (I set these values because of personal intension and visual inspection)
n_r = 1; % The Range around the start value -> Intervall = n_N-n_r:n_N+n_r
n_i = 2; % In how many values should the intervall be divided (number of loops)
n_s = (n_r*2)/(n_i+1);

m_N = 2.5;
m_r = 1;
m_i = 2;
m_s = (m_r*2)/(m_i+1);

k_N = 0;
k_r = 1;
k_i = 2;
k_s = (k_r*2)/(k_i+1);

l_N = 0;
l_r = 1;
l_i = 2;
l_s = (l_r*2)/(l_i+1);

o_N = 0;
o_r = 3;
o_i = 2;
o_s = (o_r*2)/(o_i+1);

p_N = 0;
p_r = 3;
p_i = 2;
p_s = (p_r*2)/(p_i+1);

counter = 1;
for n=1:n_i
    for m=1:m_i
        for k=1:k_i
            for l=1:l_i
                for o=1:o_i
                    for p=1:p_i
                        fprintf('Progressing: %i | %i | %i | %i | %i | %i | \n',n, m, k, l, o, p);
                        tic
                        % Set actual case orientation
                        caseOrientation{1} = [(o_N-o_r)+o*(o_s) (n_N-n_r)+n*(n_s) (k_N-k_r)+k*(k_s)];
                        caseOrientation{2} = [(p_N-p_r)+p*(p_s) (m_N-m_r)+m*(m_s) (l_N-l_r)+l*(l_s)];
                        
                        % Calculate Rectification
                        [T_ortho{3},imageView{3}] = T_ortho_calc(I{3},camOrientation{3},caseOrientation{1},cameraParams);
                        [T_ortho{4},imageView{4}] = T_ortho_calc(I{4},camOrientation{4},caseOrientation{2},cameraParams);

                        % Rectification
                        I{3} = imwarp(Io{3},T_ortho{3},'linear','OutputView',imageView{3},'Fill',0,'SmoothEdges',true);
                        I{4} = imwarp(Io{4},T_ortho{4},'linear','OutputView',imageView{4},'Fill',0,'SmoothEdges',true);
                        
                        % Calculate T-align
                        T_align{3} = affine2d(eye(3,3));
                        [T_align{4},~,~] = estimate_translation(I{3},I{4},'ShowMessages','off','ShowImages','off');

                        % Invert T-align
                        T_align{4}.T = T_align{3}.T*T_align{4}.T;
                        T_align{4}=invert(T_align{4});
                        
                        % Calculate Size of Panorama
                        [status,panoView] = calculate_panosize(T_align,I,'ShowMessages','off');
                        if status == 0
                            break;
                        end
                        % Project Images on Panorama
                        I{3} = place_on_pano(I{3},T_align{3},panoView,'ShowMessages','off','ShowImages','off');
                        I{4} = place_on_pano(I{4},T_align{4},panoView,'ShowMessages','off','ShowImages','off');
                        
                        % Mask only overlapping area
                        mask = I{3}.*I{4};
                        mask(mask>0) = 1;

                        % Mask only overlapping area
                        Iover1 = I{3}.*mask;
                        Iover2 = I{4}.*mask;
                        
                        % Calculate Pixel Difference and divide to number
                        % of pixel in overlapping area
                        diff(counter) = (sum(sum(abs(Iover1 - Iover2))))/(sum(sum(mask)));
                        bestOrientation{1}(counter,:) = [(o_N-o_r)+o*(o_s) (n_N-n_r)+n*(n_s) (k_N-k_r)+k*(k_s)];
                        bestOrientation{2}(counter,:) = [(p_N-p_r)+p*(p_s) (m_N-m_r)+m*(m_s) (l_N-l_r)+l*(l_s)];
                        disp('Parameters updated!');

                        toc
                        counter = counter+1;
                    end
                    
                end
                
            end
            
        end
        
    end
    
end

%Find minima in difference:
[~,idx] = min(diff);

bestCaseOrientation{1} = bestOrientation{1}(idx,:);
bestCaseOrientation{2} = bestOrientation{2}(idx,:);


end

%% SUBFUNCTIONS

function Ims = masking_ship(I,Im,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
% I: Image to be masked | Format uint8 and rgb | 0..255 Pixel Values
% Im: Mask | Format logical and rgb
%
% Optional Input Arguments
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 'ShowImages' | 'on' or 'off' | default is 'off' | Displays Images
% 
% Outputs
% Ims: masked Image | Format uint8 and rgb | 0..255 Pixel Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = inputParser;
defaultShowImages = 'off';
expectedShowImages = {'on','off'};
defaultShowMessages = 'off';
expectedShowMessages = {'on','off'};
classes = {'uint8'};
classes2 = {'logical'};
attributes = {'size',[NaN,NaN,3]};
addRequired(p,'I',@(x)validateattributes(x,classes,attributes));
addRequired(p,'Im',@(x)validateattributes(x,classes2,attributes));
addParameter(p,'ShowImages',defaultShowImages,...
                 @(x) any(validatestring(x,expectedShowImages)));
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,I,Im,varargin{:})


Ims = uint8(double(I).*Im); % Apply Masking

if strcmp(p.Results.ShowImages,'on')
    f=figure; imshowpair(I,Im);
    title('Green: Mask Image | Violett: Remaining Image');
    pause();
    close(f);
end

if strcmp(p.Results.ShowMessages,'on')
    fprintf('Image masked!\n');
end

end

function Iud = image_undistortion(I,cameraParams,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
% I: Image to be undistorted | Format uint8 and rgb | 0..255 Pixel Values
% cameraParams: Camera Parameters | cameraParameters Object
% 
% Optional Input Arguments
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 'ShowImages' | 'on' or 'off' | default is 'off' | Displays Images
% 
% Outputs
% Iud: undistorted Image | Format uint8 and rgb | 0..255 Pixel Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = inputParser;
defaultShowImages = 'off';
expectedShowImages = {'on','off'};
defaultShowMessages = 'off';
expectedShowMessages = {'on','off'};
ClassName = 'cameraParameters';
classes = {'uint8'};
attributes = {'size',[NaN,NaN,3]};
addRequired(p,'I',@(x)validateattributes(x,classes,attributes));
addRequired(p,'cameraParams',@(x)isa(x,ClassName));
addParameter(p,'ShowImages',defaultShowImages,...
                 @(x) any(validatestring(x,expectedShowImages)));
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,I,cameraParams,varargin{:})

% Calculate Image Size
width = size(I,1);
heigth = size(I,2);

% Correct for Radial and Tangential Distortion
Iud = undistortImage(I, cameraParams,'OutputView','Full');

% Resize Image like DistortImage
Iud = imresize(Iud,[width heigth]);

if strcmp(p.Results.ShowImages,'on')
    f=figure; imshowpair(I,Iud,'montage');
    title('Left side: Original Image | Right side: undistorted Image');
    pause();
    close(f);
end

if strcmp(p.Results.ShowMessages,'on')
    fprintf('Image undistorted!\n');
end

end

function [T_align,PixelLoc1,PixelLoc2] = estimate_translation(I1,I2,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Inputs
% I1: Image 1 | rgb uint8 0..255
% I2: Image 2 | rgb uint8 0..255
%
% Optional Input Arguments (Name value pairs)
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 'ShowImages' | 'on' or 'off' | default is 'off' | Displays Images
% 
% Outputs
% T: 1x1 affine 2d Object
% PixelLoc1: Locations [xi yi] of Feature Points 1 | numeric Nx2 Array
% PixelLoc2: Locations [xi yi] of Feature Points 2 | numeric Nx2 Array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = inputParser;
defaultShowImages = 'off';
expectedShowImages = {'on','off'};
defaultShowMessages = 'off';
expectedShowMessages = {'on','off'};
classes = {'uint8'};
attributes = {'size',[NaN,NaN,NaN]};
addRequired(p,'I1',@(x)validateattributes(x,classes,attributes));
addRequired(p,'I2',@(x)validateattributes(x,classes,attributes));
addParameter(p,'ShowImages',defaultShowImages,...
                 @(x) any(validatestring(x,expectedShowImages)));
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,I1,I2,varargin{:})

% Grayscale Images
% I1 = rgb2gray(I1);
% I2 = rgb2gray(I2);

%% Detect Features off Image 1
points1 = detectSURFFeatures(I1,'NumOctaves',1,'NumScaleLevels',4,'MetricThreshold',100);
[features1, points1] = extractFeatures(I1, points1);

% Erase Outlier Features
mask1 = I1;
mask1(mask1>0) = 1;
maskidx1=find(mask1>0);
pointsidx1 = sub2ind(size(I1),round(points1.Location(:,2)),round(points1.Location(:,1)));
validPoints1=ismember(pointsidx1,maskidx1);
notValidPointsidx1 = find(validPoints1==0);
validpoints1 = points1.Location;
validpoints1(notValidPointsidx1,:)=[];
features1(notValidPointsidx1,:) = [];


if strcmp(p.Results.ShowImages,'on')
    f1=figure; imshow(I1);
    title('Detected Features on Image1');
    hold on;
    points1.plot
    hold off;
    pause(0.5)
    close(f1);
end


%% Detect Features off Image 2
points2 = detectSURFFeatures(I2,'NumOctaves',1,'NumScaleLevels',4,'MetricThreshold',100);
[features2, points2] = extractFeatures(I2, points2);

% Erase Outlier Features
mask2 = I2;
mask2(mask2>0) = 1;
maskidx2=find(mask2>0);
pointsidx2 = sub2ind(size(I2),round(points2.Location(:,2)),round(points2.Location(:,1)));
validPoints2=ismember(pointsidx2,maskidx2);
notValidPointsidx2 = find(validPoints2==0);
validpoints2 = points2.Location;
validpoints2(notValidPointsidx2,:)=[];
features2(notValidPointsidx2,:) = [];


if strcmp(p.Results.ShowImages,'on')
    f2 = figure; imshow(I2);
    title('Detected Features on Image2');
    hold on;
    points2.plot
    hold off;
    pause(0.5)
    close(f2);
end

%% Find corresponding Points between two Images
indexPairs = matchFeatures(features1, features2,'Method','Approximate','MatchThreshold',50,'MaxRatio',0.5,'Unique',true,'Metric','SSD');
matchedPoints1 = validpoints1(indexPairs(:,1), :);
matchedPoints2 = validpoints2(indexPairs(:,2), :);

if strcmp(p.Results.ShowImages,'on')
    f3 = figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
    title('Matching Features');
    legend('matchedPts1','matchedPts2');
    pause(0.5)
    close(f3);
end

%% Estimate geometric Transform for I2
[T_align, matchedPoints1, matchedPoints2,status] = estimateGeometricTransform(matchedPoints1, matchedPoints2,...
    'similarity', 'Confidence', 75, 'MaxNumTrials', 1000, 'MaxDistance',25); %1.5 for right side

if status == 1
    fprintf('WARNING: No Matches found!\n');
    return;
end

%% Alternative Estimate Translation
% x = mean(matchedPoints2(:,1)-matchedPoints1(:,1));
% y = mean(matchedPoints2(:,2)-matchedPoints1(:,2));
% 
% Tl = [ 1 0 x; 0 1 y; 0 0 1];
% T = projective2d(Tl');

%% Extract Locations of Features
PixelLoc1 = round(matchedPoints1);
PixelLoc2 = round(matchedPoints2);

% %% Calculate Average Pixel Intensity
% linidx1=sub2ind(size(I1),round(matchedPoints1(:,2)),round(matchedPoints1(:,1)));
% linidx2=sub2ind(size(I2),round(matchedPoints2(:,2)),round(matchedPoints2(:,1)));
% PixelIntense1 = double(I1(linidx1));
% PixelIntense2 = double(I2(linidx2));

%% Show Images and Messages
if strcmp(p.Results.ShowImages,'on')
    f4 = figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage');
    title('Matching Features after Geometric Transform Estimation');
    legend('matchedPts1','matchedPts2');
end
if strcmp(p.Results.ShowMessages,'on')
    fprintf('Feature Matching of Images complete!\n');
end

%%%TRY
% sf=sign(T.T(1,1))*sqrt(T.T(1,1)^2+T.T(2,1)^2);
% T.T(3,1) = T.T(3,1) * sf;
% T.T(3,2) = T.T(3,2) * sf;
% T.T(1:2,1:2) = eye(2,2);

% T.T(1,2) = 0;
% T.T(2,1) = 0;

end

function [status,panoramaView] = calculate_panosize(T,I,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Inputs
% T: 1xN Cell which contains 1x1 affine 2d Objects | N = Number of Tforms
% I: 1xN cell containing Images | rgb uint8 0..255
%
% Optional Input Arguments (Name value pairs)
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 
% Outputs
% panoramaView: Imref2d Object 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


p = inputParser;
defaultShowMessages = 'off';
expectedShowMessages = {'on','off'};
%classes = {'uint8'};
%attributes = {'size',[NaN,NaN,3]};
addRequired(p,'T');
addRequired(p,'I');
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,T,I,varargin{:})

% Define Number of Transformations
counter = [1 2 3 4 5 6].*~cellfun('isempty',T);
counter(counter==0)=[];

% Pre Allocate Variables
xlim = zeros(length(counter),2);
ylim = zeros(length(counter),2);

% Find output limits for projected Images
for i=counter
	[xlim(i,:),ylim(i,:)] = outputLimits(T{i}, [1 size(I{i},2)], [1 size(I{i},1)]);
end

% Find the minimum and maximum of output limits
xMin = min(xlim(:));
xMax = max(xlim(:));

yMin = min(ylim(:));
yMax = max(ylim(:));

sf = round((xMax - xMin))/size(I{1},2);

% Width and height of preprocessed Images
width  = round((xMax - xMin)/sf);
height = round((yMax - yMin)/sf);


% Create a 2-D spatial reference object defining the size of the preprocessed Images
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

if strcmp(p.Results.ShowMessages,'on')
    fprintf('Panorama Size Calculation complete!\n');
end

status = 1;
if height > 10000 || width > 10000
    fprintf('WARNING: Panorama Size %i x %i too big!\n',height,width);
    status = 0;
    return;
end


end

function Ipano = place_on_pano(I,T,panoramaView,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Inputs
% I: Image | rgb uint8 0..255
% T: 1x1 affine 2d Objects
% panoramaView: Imref2d Object 
%
% Optional Input Arguments (Name value pairs)
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 'ShowImages' | 'on' or 'off' | default is 'off' | Displays Images
% 
% Outputs
% Ipano: Projected Image onto Panorama |rgb uint8 0..255
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = inputParser;
defaultShowImages = 'off';
expectedShowImages = {'on','off'};
defaultShowMessages = 'off';
expectedShowMessages = {'on','off'};
classes = {'uint8'};
attributes = {'size',[NaN,NaN,NaN]};
addRequired(p,'I',@(x)validateattributes(x,classes,attributes));
addRequired(p,'T');
addRequired(p,'panoramaView');
addParameter(p,'ShowImages',defaultShowImages,...
                 @(x) any(validatestring(x,expectedShowImages)));
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,I,T,panoramaView,varargin{:})


% Place Images on final Preprocessed Images
Ipano = imwarp(I,T,'OutputView',panoramaView);

if strcmp(p.Results.ShowMessages,'on')
    fprintf('Image placed on Panorama!\n');
end
if strcmp(p.Results.ShowImages,'on')
    figure; imshow(Ipano);
end

end

function [T_ortho,imageView] = T_ortho_calc(I,camOrientation,caseOrientation,cameraParams,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
% I: Image to be projected | Format uint8 and rgb | 0..255 Pixel Values
% camOrientation: 3x1 Matrix containing X,Y,Z Angle for 
%                    CamOrientation | double
% caseOrientation: 3x1 Matrix containing X,Y,Z Angle for Case | double
% K: 3x3 Camera Matrix | double 
% 
% Optional Input Arguments
% 'ScaleFactor' | double | Image Size will be scaled down >1 for small
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 'ShowImages' | 'on' or 'off' | default is 'off' | Displays Images
% 
% Outputs
% Ip: projected Image | Format uint8 and rgb | 0..255 Pixel Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = inputParser;
defaultShowImages = 'off';
expectedShowImages = {'on','off'};
defaultShowMessages = 'off';
expectedShowMessages = {'on','off'};
defaultScaleFactor = 1;
classes = {'uint8'};
classes2 = {'double'};
attributes = {'size',[NaN,NaN,NaN]};
attributes2 = {'size',[1,3]};
addRequired(p,'I',@(x)validateattributes(x,classes,attributes));
addRequired(p,'cameraOrientation',@(x)validateattributes(x,classes2,attributes2));
addRequired(p,'caseOrientation',@(x)validateattributes(x,classes2,attributes2));
addRequired(p,'cameraParams');
addParameter(p,'ScaleFactor',defaultScaleFactor,@isnumeric);
addParameter(p,'ShowImages',defaultShowImages,...
                 @(x) any(validatestring(x,expectedShowImages)));
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,I,camOrientation,caseOrientation,cameraParams,varargin{:})
    rotx = caseOrientation(1);
    roty = caseOrientation(2);
    rotz = caseOrientation(3);

    Rx = [1 0           0           ;...
          0 cosd(rotx)  -sind(rotx) ;...
          0 sind(rotx)  cosd(rotx)] ;

    Ry = [cosd(roty)    0   sind(roty)  ;...
          0             1   0           ;...
          -sind(roty)   0   cosd(roty)] ;

    Rz = [cosd(rotz)    -sind(rotz)     0   ;...
          sind(rotz)    cosd(rotz)      0   ;...
          0             0              1]   ;

    % Set up Final Rotation Matrix
    Rc = Rz * Ry * Rx;

    % Read Camera Orientation Angles
    rotx = camOrientation(1);
    roty = camOrientation(2);
    rotz = camOrientation(3);

    % Construct Rotation Matrices
    Rx = [1 0           0           ;...
          0 cosd(rotx)  -sind(rotx) ;...
          0 sind(rotx)  cosd(rotx)] ;

    Ry = [cosd(roty)    0   sind(roty)  ;...
          0             1   0           ;...
          -sind(roty)   0   cosd(roty)] ;

    Rz = [cosd(rotz)    -sind(rotz)     0   ;...
          sind(rotz)    cosd(rotz)      0   ;...
          0             0              1]   ;

    % Set up Final Rotation Matrix
    R = Rz * Ry * Rx;

    R = Rc*R;

    % Build Homography
    H = cameraParams.IntrinsicMatrix' * R / cameraParams.IntrinsicMatrix';

    % Make Projective2d
    T_ortho = projective2d(H');

    % Calculate limits for projected Images
    [xlim,ylim] = outputLimits(T_ortho, [1 size(I,2)], [1 size(I,1)]);

    sf=round((max(xlim) - min(xlim)))/size(I,2);
    
    
    % Width and height of projected Images
    width  = round((max(xlim) - min(xlim))/sf);
    height = round((max(ylim) - min(ylim))/sf);

    % Calculate imref2d object
    imageView = imref2d([height width], xlim, ylim);

    if strcmp(p.Results.ShowMessages,'on')
        fprintf('Calculation of T_ortho complete!\n');
    end
end