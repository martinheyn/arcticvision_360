function panorama = image_stitching_mask(I1,I2,mask,varargin)
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
temp = rgb2gray(double(~mask));
mp{1} = imresize(temp,[size(im1p{1},1) size(im1p{1},2)],'bilinear');
clear temp

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
%    titel('Resulting blended Panorama');
end
if strcmp(p.Results.ShowMessages,'on')
    fprintf('Blended Panorama complete!\n');
end

end