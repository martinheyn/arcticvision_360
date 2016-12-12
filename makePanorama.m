function [Panorama,IceConST,IceConKM,IceFloeSize] = makePanorama(imageSet,ship_masks,cameraParams,T_ortho,imageView,T_align,panoView,blending_masks,pixel_loc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MakePanorama Function
% Inputs: 
% - imageSet:        1x1 ImageSet containing all input imageset of frame(s)
% - ship_masks:      Binary images | rgb logical
% - cameraParams:    Camera Parameter Object
% - CamOrientation:  Cell Array with 1x3 double vectors
% - CaseOrientation: Cell Array with 1x3 double vectors
% - T-ortho:         Cell Array with 3x3 projective 2D Objects
% - T-align:         Cell Array with 3x3 affine 2D Objects
% - imageView:       Cell Array of imref2d Objects
% - panoView:        Imref2d Object
% - pixel_loc:       Cell Array with 1x2 double vectors
% - blending_masks:  Binary images | rgb logical
%
% Outputs: 
% -panorama:         Depending on imageSet ->
%                    - ImageSequence
%                    - Single Panoramic Image 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check if single or multiple frames
if length(imageSet) == 1
    singlemultiple = 1;
end
if length(imageSet) > 1
    singlemultiple = 0;
end

switch singlemultiple
    case 1 % Single Frame Panorama
        [ImageFile1,ImagePath1] = uiputfile('*.jpg'); % Get Directory for save Panorama file
        
        ImagePath2 = ImagePath1;
        ImagePath3 = ImagePath2;
        ImagePath4 = ImagePath3;
        ImagePath5 = ImagePath4;
        
        ImageFile2 = strcat('IceConMapSimpThres',ImageFile1);
        ImageFile3 = strcat('IceConMapKMeans',ImageFile1);
        ImageFile4 = strcat('IcefloeDistMap',ImageFile1);
        ImageFile5 = strcat('IcefloeDistEdge',ImageFile1);
        
%         [ImageFile2,ImagePath2] = uiputfile(ImagePath1,'*.jpg'); % Get Directory for save IceConMap file Simple Thres
%         [ImageFile3,ImagePath3] = uiputfile('*.jpg'); % Get Directory for save IceConMap file KMeans
%         [ImageFile4,ImagePath4] = uiputfile('*.jpg'); % Get Directory for save IcefloeDistMap file Map
%         [ImageFile5,ImagePath5] = uiputfile('*.jpg'); % Get Directory for save IcefloeDistMap file Edges
        h = waitbar(0,'Reading Images...','Name','Creating Panoramic Image (This can take several minutes!)'); %open waitbar
        
        % Preallocate Variables to improve Speed
        I = cell(1,6);

        % Read Images
        for j=1:6
			k = -1*(-7+j);
            I{j} = read(imageSet, k);
        end
        waitbar(0.1,h,sprintf('Masking out Ship...'))
        
        % Masking out ship
        for j=1:6
            I{j} = masking_ship(I{j},ship_masks{j},'ShowMessages','on','ShowImages','off');
        end
        waitbar(0.2,h,sprintf('Undistort Images...'))
        
        % Undistort Images
        for j=1:6
            I{j} = image_undistortion(I{j},cameraParams,'ShowMessages','on','ShowImages','off');
        end
        waitbar(0.3,h,sprintf('Rectify Images...'))
        
        % Rectify Images
        for j=1:6
            I{j} = imwarp(I{j},T_ortho{j},'linear','OutputView',imageView{j},'Fill',0,'SmoothEdges',true);
        end
        waitbar(0.5,h,sprintf('Colorcorrect Images...'))
        
        % Colorcorrect Images % Image 1 is Reference
        I{1} = rgb2gray(I{1});
        I{2} = colorcorrection(I{1},I{2},pixel_loc{1}{1},pixel_loc{1}{2},'ShowMessages','on','ShowImages','off');
        I{3} = colorcorrection(I{2},I{3},pixel_loc{2}{2},pixel_loc{2}{3},'ShowMessages','on','ShowImages','off');
        I{4} = colorcorrection(I{3},I{4},pixel_loc{3}{3},pixel_loc{3}{4},'ShowMessages','on','ShowImages','off');
        I{5} = colorcorrection(I{4},I{5},pixel_loc{4}{4},pixel_loc{4}{5},'ShowMessages','on','ShowImages','off');
        I{6} = colorcorrection(I{5},I{6},pixel_loc{5}{5},pixel_loc{5}{6},'ShowMessages','on','ShowImages','off');
        waitbar(0.6,h,sprintf('Placing Images on Panorama...'))
        
        % Place Images on Panorama
        for j=1:6
            I{j} = place_on_pano(I{j},T_align{j},panoView,'ShowMessages','on','ShowImages','off');
        end
        waitbar(0.7,h,sprintf('Blending Images...'))
        
        % Blending
        I{1} = image_stitching(I{1},I{2},blending_masks{1},'ShowMessages','on','ShowImages','off');
        I{2} = image_stitching(I{1},I{3},blending_masks{2},'ShowMessages','on','ShowImages','off');
        I{3} = image_stitching(I{2},I{4},blending_masks{3},'ShowMessages','on','ShowImages','off');
        I{4} = image_stitching(I{3},I{5},blending_masks{4},'ShowMessages','on','ShowImages','off');
        I{5} = image_stitching(I{4},I{6},blending_masks{5},'ShowMessages','on','ShowImages','off');
        waitbar(0.8,h,sprintf('Ice Concentration Analysis...'))
        
        % Function Output
        PrePanorama = I{5};
        
        % Crop Image and make Space for IceAnalysis Results
        cols=max(PrePanorama,[],1);
        colsmin = find(cols,1,'first');
        colsmax = find(cols,1,'last');
        rows=max(PrePanorama,[],2);
        rowsmin = find(rows,1,'first');
        rowsmax = find(rows,1,'last');
        rect = [colsmin rowsmin colsmax-colsmin rowsmax-rowsmin];
        PrePanorama =  imcrop(PrePanorama,rect);
        PrePanorama = [PrePanorama; zeros(400,size(PrePanorama,2))];
    
        % Detect Ice Concentration Simple Thres
        [Imap1,PercentageST] = ice_detection(PrePanorama,'SimpleThreshold',1,90,130,4,'ShowMessages','on','ShowImages','off');
        % Insert into Image
        Panorama = rgb2gray(insertText(PrePanorama,[1 size(PrePanorama,1)-300],sprintf('Ice-Water Concentration(SimpleThreshold): Ice: %2.2f%% Dark Ice: %2.2f%% Water: %2.2f%%',PercentageST(1),PercentageST(2),PercentageST(3)),'FontSize',48,'TextColor','black','BoxColor','white'));
        % Detect Ice Concentration Kmeans
        [Imap2,PercentageKM] = ice_detection(PrePanorama,'KmeansAlgorithm',1,90,130,4,'ShowMessages','on','ShowImages','off');
        % Insert into Image
        Panorama = rgb2gray(insertText(Panorama,[1 size(PrePanorama,1)-200],sprintf('Ice-Water Concentration(Kmeans): Ice: %2.2f%% Dark Ice: %2.2f%% Water: %2.2f%%',PercentageKM(1),PercentageKM(2),PercentageKM(3)),'FontSize',48,'TextColor','black','BoxColor','white'));
        waitbar(0.9,h,sprintf('Ice Floe Distribution Analysis...'))
        
        % Detect Ice Floe Distribution Map
        [Imap3,PercentageFloeDis] = ice_detection(PrePanorama,'FloeDistAlgorithm',4,90,130,4,'ShowMessages','on','ShowImages','off');
        % Insert into Image
        Panorama = rgb2gray(insertText(Panorama,[1 size(PrePanorama,1)-100],sprintf('Floe Size Distribution: Small: %2.2f%% Medium: %2.2f%% Large: %2.2f%%',PercentageFloeDis(1),PercentageFloeDis(2),PercentageFloeDis(3)),'FontSize',48,'TextColor','black','BoxColor','white'));
        % Detect Ice Floe Distribution Edges
        [Imap4,~] = ice_detection(PrePanorama,'Edges',4,90,110,2,'ShowMessages','on','ShowImages','off');
        % Insert into Image

        imwrite(Panorama,fullfile(ImagePath1,ImageFile1));
        imwrite(uint8(Imap1),fullfile(ImagePath2,ImageFile2));
        imwrite(uint8(Imap2),fullfile(ImagePath3,ImageFile3));
        imwrite(uint8(Imap3),fullfile(ImagePath4,ImageFile4));
        imwrite(uint8(Imap4),fullfile(ImagePath5,ImageFile5));
        
        waitbar(1,h,sprintf('Processing Succeded!'))
        IceConST = [PercentageST(1),PercentageST(2),PercentageST(3)];
        IceConKM = [PercentageKM(1),PercentageKM(2),PercentageKM(3)];
        IceFloeSize = [PercentageFloeDis(1),PercentageFloeDis(2),PercentageFloeDis(3)];
        
        
        
    case 0 % Multiple Frame Panorama
        [VideoFile1,VideoPath1] = uiputfile(); % Get Directory for save Videofile Panorama
        VideoFile1 = strrep(VideoFile1,'.rpt','');
        
        VideoPath2 = VideoPath1;
        VideoPath3 = VideoPath2;
        VideoPath4 = VideoPath3;
        VideoPath5 = VideoPath4;
        
        VideoFile2 = strcat('IceConMapSimpThres',VideoFile1);
        VideoFile3 = strcat('IceConMapKMeans',VideoFile1);
        VideoFile4 = strcat('IcefloeDistMap',VideoFile1);
        VideoFile5 = strcat('IcefloeDistEdge',VideoFile1);
        
        moviefile1 = VideoWriter(strcat(VideoPath1,'\',VideoFile1),'MPEG-4');
        moviefile1.FrameRate = 5;
        open(moviefile1);
        moviefile2 = VideoWriter(strcat(VideoPath2,'\',VideoFile2));
        moviefile2.FrameRate = 5;
        open(moviefile2);
        moviefile3 = VideoWriter(strcat(VideoPath3,'\',VideoFile3));
        moviefile3.FrameRate = 5;
        open(moviefile3);
        moviefile4 = VideoWriter(strcat(VideoPath4,'\',VideoFile4));
        moviefile4.FrameRate = 5;
        open(moviefile4);
        moviefile5 = VideoWriter(strcat(VideoPath5,'\',VideoFile5));
        moviefile5.FrameRate = 5;
        open(moviefile5);
%         
        timedone=0;
        h = waitbar(0,'Reading Images...','Name','Creating Panoramic Video (This can take several minutes!)'); %open waitbar
        set = 0;
        est_time = NaN;
        for i=1:length(imageSet)
            tic
            % Preallocate Variables to improve Speed
            I = cell(1,6);

            % Read Images
            for j=1:6
			    k = -1*(-7+j);
                %k = j;
                I{j} = read(imageSet(set+i), k);
            end
            waitbar(0.1/length(imageSet)*i,h,sprintf('Processing Frame %i of %i | Remaining time: %2.2f min\n Masking out Ship...',i,length(imageSet),est_time/60))
            
            % disable annoying warning messages
            [~, MSGID] = lastwarn();
            warning('off',MSGID);
            
            % Masking out ship
            for j=1:6
                I{j} = masking_ship(I{j},ship_masks{j},'ShowMessages','on','ShowImages','off');
            end
            waitbar(0.2/length(imageSet)*i,h,sprintf('Processing Frame %i of %i | Remaining time: %2.2f min\n Undistort Images...',i,length(imageSet),est_time/60))

            % Undistort Images
            for j=1:6
                I{j} = image_undistortion(I{j},cameraParams,'ShowMessages','on','ShowImages','off');
            end
            waitbar(0.3/length(imageSet)*i,h,sprintf('Processing Frame %i of %i | Remaining time: %2.2f min\n Rectify Images...',i,length(imageSet),est_time/60))

            % Rectify Images
            for j=1:6
                I{j} = imwarp(I{j},T_ortho{j},'linear','OutputView',imageView{j},'Fill',0,'SmoothEdges',true);
            end
            waitbar(0.5/length(imageSet)*i,h,sprintf('Processing Frame %i of %i | Remaining time: %2.2f min\n Colorcorrect Images...',i,length(imageSet),est_time/60))

            % Colorcorrect Images % Image 1 is Reference
            I{1} = rgb2gray(I{1});
            I{2} = colorcorrection(I{1},I{2},pixel_loc{1}{1},pixel_loc{1}{2},'ShowMessages','on','ShowImages','off');
            I{3} = colorcorrection(I{2},I{3},pixel_loc{2}{2},pixel_loc{2}{3},'ShowMessages','on','ShowImages','off');
            I{4} = colorcorrection(I{3},I{4},pixel_loc{3}{3},pixel_loc{3}{4},'ShowMessages','on','ShowImages','off');
            I{5} = colorcorrection(I{4},I{5},pixel_loc{4}{4},pixel_loc{4}{5},'ShowMessages','on','ShowImages','off');
            I{6} = colorcorrection(I{5},I{6},pixel_loc{5}{5},pixel_loc{5}{6},'ShowMessages','on','ShowImages','off');
            waitbar(0.6/length(imageSet)*i,h,sprintf('Processing Frame %i of %i | Remaining time: %2.2f min\n Placing Images on Panorama...',i,length(imageSet),est_time/60))

            % Place Images on Panorama
            for j=1:6
                I{j} = place_on_pano(I{j},T_align{j},panoView,'ShowMessages','on','ShowImages','off');
            end
            waitbar(0.7/length(imageSet)*i,h,sprintf('Processing Frame %i of %i | Remaining time: %2.2f min\n Blending Images...',i,length(imageSet),est_time/60))

            % Blending
            I{1} = image_stitching_mask(I{1},I{2},blending_masks{1},'ShowMessages','on','ShowImages','off');
            I{2} = image_stitching_mask(I{1},I{3},blending_masks{2},'ShowMessages','on','ShowImages','off');
            I{3} = image_stitching_mask(I{2},I{4},blending_masks{3},'ShowMessages','on','ShowImages','off');
            I{4} = image_stitching_mask(I{3},I{5},blending_masks{4},'ShowMessages','on','ShowImages','off');
            I{5} = image_stitching_mask(I{4},I{6},blending_masks{5},'ShowMessages','on','ShowImages','off');
            waitbar(0.8/length(imageSet)*i,h,sprintf('Processing Frame %i of %i | Remaining time: %2.2f min\n Ice Concentration Analysis...',i,length(imageSet),est_time/60))

            % Function Output
            PrePanorama = I{5};

            % Crop Image and make Space for IceAnalysis Results
            if i==1 % Only do it in the first loop to improve speed
                cols=max(PrePanorama,[],1);
                colsmin = find(cols,1,'first');
                colsmax = find(cols,1,'last');
                rows=max(PrePanorama,[],2);
                rowsmin = find(rows,1,'first');
                rowsmax = find(rows,1,'last');
                rect = [colsmin rowsmin colsmax-colsmin rowsmax-rowsmin];
            end
            PrePanorama =  imcrop(PrePanorama,rect);
            PrePanorama = [PrePanorama; zeros(400,size(PrePanorama,2))];

            % Detect Ice Concentration Simple Thres
            [Imap1,PercentageST] = ice_detection(PrePanorama,'SimpleThreshold',1,90,130,8,'ShowMessages','on','ShowImages','off');
            % Insert into Image
            SPanorama = rgb2gray(insertText(PrePanorama,[1 size(PrePanorama,1)-300],sprintf('Ice-Water Concentration(SimpleThreshold): Ice: %2.2f%% Dark Ice: %2.2f%% Water: %2.2f%%',PercentageST(1),PercentageST(2),PercentageST(3)),'FontSize',48,'TextColor','black','BoxColor','white'));
            % Detect Ice Concentration KMeans
            [Imap2,PercentageKM] = ice_detection(PrePanorama,'KmeansAlgorithm',1,90,130,8,'ShowMessages','on','ShowImages','off');
            % Insert into Image
            SPanorama = rgb2gray(insertText(SPanorama,[1 size(PrePanorama,1)-200],sprintf('Ice-Water Concentration(Kmean): Ice: %2.2f%% Dark Ice: %2.2f%% Water: %2.2f%%',PercentageKM(1),PercentageKM(2),PercentageKM(3)),'FontSize',48,'TextColor','black','BoxColor','white'));
            waitbar(0.9/length(imageSet)*i,h,sprintf('Processing Frame %i of %i | Remaining time: %2.2f min\n Ice Floe Distribution Analysis...',i,length(imageSet),est_time/60))
            
            % Detect Ice Floe Distribution Map
            [Imap3,PercentageFloeDis] = ice_detection(PrePanorama,'FloeDistAlgorithm',4,90,130,8,'ShowMessages','on','ShowImages','off');
            % Insert into Image
            SPanorama = rgb2gray(insertText(SPanorama,[1 size(PrePanorama,1)-100],sprintf('Floe Size Distribution: Small: %2.2f%% Medium: %2.2f%% Large: %2.2f%%',PercentageFloeDis(1),PercentageFloeDis(2),PercentageFloeDis(3)),'FontSize',48,'TextColor','black','BoxColor','white'));
            % Detect Ice Floe Distribution Edges
            [Imap4,~] = ice_detection(PrePanorama,'Edges',4,90,110,32,'ShowMessages','on','ShowImages','off');
           
            % Store Images in Sequence Variable
%             Panorama(:,:,1,i) = SPanorama;
%             Map1(:,:,:,i) = Imap1;
%             Map2(:,:,:,i) = Imap2;
%             Map3(:,:,:,i) = Imap3;
%             Map4(:,:,:,i) = Imap4;

            IceConST(i,:) = [PercentageST(1),PercentageST(2),PercentageST(3)];
            IceConKM(i,:) = [PercentageKM(1),PercentageKM(2),PercentageKM(3)];
            IceFloeSize(i,:) = [PercentageFloeDis(1),PercentageFloeDis(2),PercentageFloeDis(3)];
            
            Panorama(:,:,1) = imresize(SPanorama,[1920,NaN]);
            Map1(:,:,:) = uint8(imresize(Imap1,[1920,NaN]));
            Map2(:,:,:) = uint8(imresize(Imap2,[1920,NaN]));
            Map3(:,:,:) = uint8(imresize(Imap3,[1920,NaN]));
            Map4(:,:,:) = uint8(imresize(Imap4,[1920,NaN]));
            
            % Write into VideoFiles
            writeVideo(moviefile1,Panorama);
            writeVideo(moviefile2,Map1);
            writeVideo(moviefile3,Map2);
            writeVideo(moviefile4,Map3);
            writeVideo(moviefile5,Map4);
            
            clear Panorama Map1 Map2 Map3 Map4
            
            % update waitbar
            time = toc;
            timedone = timedone+time;
            est_time = timedone/i*length(imageSet)-timedone;
            
        end
        waitbar(0.99/length(imageSet)*i,h,sprintf('Processing complete | Remaining time: %2.2f min\n Generating Video File...',0.08))
%         createVideoFile(Panorama,fullfile(VideoPath1,VideoFile1),4) %Save Videofile
%         createVideoFile((Map1),fullfile(VideoPath2,VideoFile2),4) %Save Videofile
%         createVideoFile((Map2),fullfile(VideoPath3,VideoFile3),4) %Save Videofile
%         createVideoFile((Map3),fullfile(VideoPath4,VideoFile4),1) %Save Videofile
%         createVideoFile((Map4),fullfile(VideoPath5,VideoFile5),1) %Save Videofile
        close(moviefile1);
%         close(moviefile2);
%         close(moviefile3);
%         close(moviefile4);
%         close(moviefile5);
        waitbar(1,h,sprintf('Processing Succeded!'))
end

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


function Icc = colorcorrection(I1,I2,PixelLoc1,PixelLoc2,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Inputs
% I1: Reference Image | grayscale or rgb uint8 0..255
% I2: Correcting Image | grayscale or rgb uint8 0..255
% PixelLoc1: Locations of Reference Feature Points | numeric Nx2 Array
% PixelLoc2: Locations of Correcting Feature Points | numeric Nx2 Array
% 
% Optional Input Arguments (Name value pairs)
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 'ShowImages' | 'on' or 'off' | default is 'off' | Displays Images
% 
% Outputs
% Icc: Colorcorrected Image | grayscale uint8 0..255
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
addRequired(p,'PixelLoc1',@isnumeric);
addRequired(p,'PixelLoc2',@isnumeric);
addParameter(p,'ShowImages',defaultShowImages,...
                 @(x) any(validatestring(x,expectedShowImages)));
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,I1,I2,PixelLoc1,PixelLoc2,varargin{:})

%Make RGB to Gray if RGB Image is input:
if size(I1,3) == 3
    I1 = rgb2gray(I1);
end
if size(I2,3) == 3
    I2 = rgb2gray(I2);
end

% Mask valid Pixels inside Image
%maskidx = find(I1>0);
mask = I2;
mask(mask>0)=1;
% Calculate Average Intensity of Pixels in Image
AvgIntense = mean(I2(sub2ind(size(I2),PixelLoc2(:,2),PixelLoc2(:,1))));
% Calculate Average Intensity of Pixels in Reference Image
AvgRefIntense = mean(I1(sub2ind(size(I1),PixelLoc1(:,2),PixelLoc1(:,1))));
% Calculate Difference
diff = AvgIntense - AvgRefIntense;
Icc = I2;
% Apply Correction on masked Pixels
%Icc(maskidx) = Icc(maskidx) + diff;
Icc = Icc - diff;
Icc = Icc.*mask;

if strcmp(p.Results.ShowImages,'on')
    figure; imshowpair(Icc,I2,'montage');
    title('Left Side: Corrected Image - Right Side: Original Image');
end
if strcmp(p.Results.ShowMessages,'on')
    fprintf('Color Correction of Image complete!\n');
end

end


function Ipano = place_on_pano(I,T,panoramaView,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Inputs
% I: Image | grayscale or rgb uint8 0..255
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


function panorama = image_stitching(I1,I2,mask,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Inputs
% I1: Image 1 | grayscale uint8 0..255
% I2: Image 2 | grayscale uint8 0..255
% mask: Binary blending mask | same size as I1,I2 | rgb double 0..1
%
% Optional Input Arguments (Name value pairs)
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 'ShowImages' | 'on' or 'off' | default is 'off' | Displays Images
% 
% Outputs
% panorama: Blended Panorama Image | grayscale uint8 0..255
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = inputParser;
defaultShowImages = 'off';
expectedShowImages = {'on','off'};
defaultShowMessages = 'off';
expectedShowMessages = {'on','off'};
classes = {'uint8'};
attributes = {'size',[NaN,NaN,1]};
classes2 = {'logical'};
attributes2 = {'size',[NaN,NaN,3]};
addRequired(p,'I1',@(x)validateattributes(x,classes,attributes));
addRequired(p,'I2',@(x)validateattributes(x,classes,attributes));
addRequired(p,'mask',@(x)validateattributes(x,classes2,attributes2));
addParameter(p,'ShowImages',defaultShowImages,...
                 @(x) any(validatestring(x,expectedShowImages)));
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,I1,I2,mask,varargin{:})

% Converting Images to double
im1p{1}=double(I1);
im2p{1}=double(I2);

% Converting and inverting Mask to Double and Grayscale
mp{1} = rgb2gray(double(~mask));

M = floor(log2(max(size(I1))));
 
 % Gaussian pyramid
for n = 2 : M
    % downsample image
    im1p{n} = imresize(im1p{n-1}, 0.5);
    im2p{n} = imresize(im2p{n-1}, 0.5);
    % downsample blending mask
    mp{n} = imresize(mp{n-1}, 0.5, 'bilinear');
end
 
 % Laplician pyramid
for n = 1 : M-1
    im1p{n} = im1p{n} - imresize(im1p{n+1}, [size(im1p{n},1), size(im1p{n},2)]);
    im2p{n} = im2p{n} - imresize(im2p{n+1}, [size(im2p{n},1), size(im2p{n},2)]);   
end   
 
 % Multi-band blending Laplician pyramid
for n = 1 : M
    imp{n} = im1p{n} .* mp{n} + im2p{n} .* (1-mp{n});
end
 
 % Laplician pyramid reconstruction
panorama = imp{M};
for n = M-1 : -1 : 1
    panorama = imp{n} + imresize(panorama, [size(imp{n},1) size(imp{n},2)]);
end

% Masking out Blending Issues
mask = I1+I2;
mask = double(logical(mask));
panorama = mask.*panorama;

% Converting Image back to uint8
panorama = uint8(panorama);

if strcmp(p.Results.ShowImages,'on')
	figure; imshow(panorama);
    titel('Resulting blended Panorama');
end
if strcmp(p.Results.ShowMessages,'on')
    fprintf('Blended Panorama complete!\n');
end

end


function [Iout,Percentage] = ice_detection(I,method,sf,th1,th2,blocksize,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Inputs
% I: Panoramic Image | grayscale uint8 0..255
% method: 'SimpleThreshold' or 'KmeansAlgorithm' or 'FloeDistAlgorithm'
% or 'Edges'
% sf: Scalefactor | double
% th1: Threshold -> Water is found when pixel intensity < th1 | double
% th2: Threshold -> Ice is found when pixel intensity > th2 | double
% blocksize: Size of Blocks in FloeDistAlgorithm | double
%
% Optional Input Arguments (Name value pairs)
% 'ShowMessages' | 'on' or 'off' | default is 'off' | Displays Messages
% 'ShowImages' | 'on' or 'off' | default is 'off' | Displays Images
% 
% Outputs
% Percentage: 1x3 vector with percentage | double
%             -> (1) = Ice or Small Ice Floes
%             -> (2) = Dark Ice or Medium Ice Floes
%             -> (3) = Water or Large Ice Floes
% Iout: Ice Floe Size Distribution Map
% Iout: Ice Concentration Map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = inputParser;
defaultShowImages = 'off';
expectedShowImages = {'on','off'};
defaultShowMessages = 'off';
expectedShowMessages = {'on','off'};
classes = {'uint8'};
attributes = {'size',[NaN,NaN,1]};
classes2 = {'double'};
attributes2 = {'size',[1,1,1]};
addRequired(p,'I',@(x)validateattributes(x,classes,attributes));
addRequired(p,'method');
addRequired(p,'sf',@(x)validateattributes(x,classes2,attributes2));
addRequired(p,'th1',@(x)validateattributes(x,classes2,attributes2));
addRequired(p,'th2',@(x)validateattributes(x,classes2,attributes2));
addRequired(p,'blocksize',@(x)validateattributes(x,classes2,attributes2));
addParameter(p,'ShowImages',defaultShowImages,...
                 @(x) any(validatestring(x,expectedShowImages)));
addParameter(p,'ShowMessages',defaultShowMessages,...
                 @(x) any(validatestring(x,expectedShowMessages)));
parse(p,I,method,sf,th1,th2,blocksize,varargin{:})


% Scale for Performance
I = imresize(I,1/sf);

% Clear out holes inside mask
Imask = zeros(size(I));
Imask(I>0) = 1;
Imaskfilled = imfill(Imask,'holes');
se = strel('disk',2);
Imaskerode = imerode(Imaskfilled,se);
mask = find(Imaskerode == 0);

%Typecast to enable writing NAN
I=double(I);

% Mask ROI
I(mask)=nan;

switch method
    case 'SimpleThreshold'
        % Simple Threshold Method
        water = find(I<th1);
        total = find(~isnan(I));
        ice = find(I>th2);

        % Show Water
        ImaskWater = zeros(size(I));
        ImaskWater(water) = 255;
        ImaskIce = zeros(size(I));
        ImaskIce(ice) = 255;
        ImaskDarkIce = ones(size(I)).*255;
        ImaskDarkIce = ImaskDarkIce - ImaskIce - ImaskWater;
        
        Ic = zeros(size(I,1),size(I,2),3);
        Ic(:,:,1) = I./1.5 + ImaskDarkIce;
        Ic(:,:,2) = I./1.5 + ImaskIce;
        Ic(:,:,3) = I./1.5 + ImaskWater;

        % Calculate Ice Concentration
        TotalPixel = length(total);
        WaterPixel = length(water);
        IcePixel = length(ice);
        DarkIcePixel = TotalPixel-WaterPixel-IcePixel;

        Percentage(1) = (IcePixel/TotalPixel)*100.0;
        Percentage(2) = (DarkIcePixel/TotalPixel)*100.0;
        Percentage(3) = (WaterPixel/TotalPixel)*100.0;
        Iout = Ic;
        if strcmp(p.Results.ShowImages,'on')
            figure; imshow(uint8(Ic));
            title('SIMPLE THRESHOLD METHOD | Blue=Water, Red=DarkIce, Green=Ice, Black=OutsideROI');
        end
        if strcmp(p.Results.ShowMessages,'on')
            fprintf('Ice Analysis complete!\n');
        end
        
    case 'KmeansAlgorithm'
        nrows = size(I,1);
        ncols = size(I,2);
        img_vec = reshape(I,nrows*ncols,1); %Image->Vector
        [idx,C,~] = kmeans(img_vec,3,'distance','sqEuclidean','Replicates',3);
        
        cluster_water = find(C==min(C));
        idx_water = find(idx==cluster_water);
        img_vec_water = zeros(size(I,1)*size(I,2),1);
        img_vec_water(idx_water) = 255;
        ImaskWater = reshape(img_vec_water,nrows,ncols); %Vector->Image
        
        cluster_ice = find(C==max(C));
        idx_ice = find(idx==cluster_ice);
        img_vec_ice = zeros(size(I,1)*size(I,2),1);
        img_vec_ice(idx_ice) = 255;
        ImaskIce = reshape(img_vec_ice,nrows,ncols); %Vector->Image
        
        ImaskDarkIce = ones(size(I)).*255;
        ImaskDarkIce = ImaskDarkIce - ImaskIce - ImaskWater;
        
        Ic = zeros(size(I,1),size(I,2),3);
        Ic(:,:,1) = I./1.5 + ImaskDarkIce;
        Ic(:,:,2) = I./1.5 + ImaskIce;
        Ic(:,:,3) = I./1.5 + ImaskWater;

        TotalPixel = length(find(~isnan(I)));
        WaterPixel = length(idx_water);
        IcePixel = length(idx_ice);
        DarkIcePixel = TotalPixel-WaterPixel-IcePixel;
        
        Percentage(1) = (IcePixel/TotalPixel)*100.0;
        Percentage(2) = (DarkIcePixel/TotalPixel)*100.0;
        Percentage(3) = (WaterPixel/TotalPixel)*100.0;
        Iout = Ic;
        if strcmp(p.Results.ShowImages,'on')
            figure; imshow(uint8(Ic));
            title('KMEANS METHOD | Blue=Water, Red=DarkIce, Green=Ice, Black=OutsideROI');
        end
        if strcmp(p.Results.ShowMessages,'on')
            fprintf('Ice Analysis complete!\n');
        end
    case 'FloeDistAlgorithm'
        IceFloeDist = blockproc(I,[blocksize,blocksize],@icefloespecies);
        
        IceFloeDist(:,:,1) = [];
        
        img_small = zeros(size(IceFloeDist));
        img_small(IceFloeDist==0.1) = 255;

        img_medium = zeros(size(IceFloeDist));
        img_medium(IceFloeDist==0.5) = 255;

        img_large = zeros(size(IceFloeDist));
        img_large(IceFloeDist==1.0) = 255;

        Id = zeros(size(I,1),size(I,2),3);
        Id(:,:,1) = I./1.5+img_medium;
        Id(:,:,2) = I./1.5+img_large;
        Id(:,:,3) = I./1.5+img_small;
        
        small = sum(img_small(:));
        medium = sum(img_medium(:));
        large = sum(img_large(:));
        total = small+medium+large;

        Percentage(1) = (small/total)*100.0;
        Percentage(2) = (medium/total)*100.0;
        Percentage(3) = (large/total)*100.0;
        Iout = Id;
        
        if strcmp(p.Results.ShowImages,'on')
            figure; imshowpair(I,uint8(Id),'montage');
            title('ICE FLOE DISTRIBUTION | Red=medium Floes, Blue=large Floes, Green=small Floes, Black=OutsideROI');
            figure; bar([1 2 3],[small/total medium/total large/total]);
            title('1: small Ice floes | 2: medium Ice floes | 3: large ice floes');
        end
        if strcmp(p.Results.ShowMessages,'on')
            fprintf('Ice Analysis complete!\n');
        end
    case 'Edges'
        Iout = blockproc(I,[blocksize,blocksize],@icefloespecies);
        Iout(:,:,2) = [];
        Iout = Iout.*255;
        Percentage = [NaN NaN NaN];
end

function Outputs = icefloespecies(Img)
            ice1 = find(Img.data>th1);
            ImaskIce1 = zeros(size(Img.data));
            ImaskIce1(:,:) = 0;
            ImaskIce1(ice1) = 1;

            [~, MSGID] = lastwarn();
            warning('off',MSGID);

            Edges = edge(Img.data,'Roberts',8);%'zerocross',0.4);
        
            IceEdges = Edges.*ImaskIce1;
            iceblocksize = sum(ImaskIce1(:));
            species = zeros(size(Img.data));
            
            value = sum(IceEdges(:))/iceblocksize;
            if iceblocksize == 0
                species(:,:) = 0.0; %completely not ice
            else
                if value <0.05
                    species(ice1) = 1.0; %large
                end
                if value >=0.05 && value <=0.10
                    species(ice1) = 0.5; %medium
                end
                if value >0.2
                    species(ice1) = 0.1; %small
                end
            end
            
            Outputs = cat(3, IceEdges, species);
end

end


function createVideoFile(ImageSec,VideoFile,sf)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Inputs:
% ImageSequence: Set of Panoramic Images | double grayscale 0..255
% VideoFile: Directory and Filename of output Video
% sf: Scale Factor to make Videofile not too large
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeObj = VideoWriter(VideoFile);
fps= 25;
writeObj.FrameRate = fps;

%Pre Allocate Frames Variable
for i=1:size(ImageSec,4)
Frames(:,:,:,i) = uint8(imresize(ImageSec(:,:,:,i),1/sf));
end

open(writeObj);
writeVideo(writeObj,Frames);
close(writeObj);

end