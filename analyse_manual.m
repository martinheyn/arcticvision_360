function waterquota = analyse_manual(waterimage,rawimage)


    water = waterimage;
    graywater = rgb2gray(water);

    waterpixel = zeros(size(graywater));
    waterpixel(graywater<255)=1;
    imshow(waterpixel)


    I = rawimage;
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

    total = find(~isnan(I));
    totalnum = length(total);
    countpix = find(waterpixel==1);
    countpixnum = length(countpix);

    waterquota = countpixnum/totalnum;

end