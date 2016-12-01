for m = 2
    for n = -2:1:2
        
        load(strcat('Init_C1-',num2str(m),'-0-0_C2-',num2str(n),'-0-0.mat'),'panorama')
        
        figure
        imshow(panorama)
        title(strcat('SB: ',num2str(m),'PR: ',num2str(n)))
    end
end