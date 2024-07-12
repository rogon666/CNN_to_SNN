clc
clear
close all
%%
for i=1:26
    str1=['ref/',num2str(1) ,' (',num2str(i),').dcm'];
    im=double(dicomread(str1));
    str2=['GT/',num2str(1) ,' (',num2str(i),').dcm'];
    im2=double(dicomread(str2));
    im=imresize(mapminmax(im,0,1)*255,0.25);
    im2=imresize(mapminmax(im2,0,1)*255,0.25);
    imwrite(uint8(im),['ref/',num2str(1) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(1) ,' (',num2str(i),').jpg'])
    
    imn=imnoise(uint8(im),'gaussian');
    imwrite(uint8(imn),['ref/',num2str(5) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(5) ,' (',num2str(i),').jpg'])
    
    imc=histeq(uint8(im));
    imwrite(uint8(imc),['ref/',num2str(6) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(6) ,' (',num2str(i),').jpg'])
    
    imm=medfilt2(uint8(im));
    imwrite(uint8(imm),['ref/',num2str(7) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(7) ,' (',num2str(i),').jpg'])
    
    im=imrotate(im,90);
    im2=imrotate(im2,90);
    imwrite(uint8(im),['ref/',num2str(2) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(2) ,' (',num2str(i),').jpg'])
    imn=imnoise(uint8(im),'gaussian');
    imwrite(uint8(imn),['ref/',num2str(8) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(8) ,' (',num2str(i),').jpg'])
    
    imc=histeq(uint8(im));
    imwrite(uint8(imc),['ref/',num2str(9) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(9) ,' (',num2str(i),').jpg'])
    
    imm=medfilt2(uint8(im));
    imwrite(uint8(imm),['ref/',num2str(10) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(10) ,' (',num2str(i),').jpg'])
    
    
    im=imrotate(im,90);
    im2=imrotate(im2,90);
    imwrite(uint8(im),['ref/',num2str(3) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(3) ,' (',num2str(i),').jpg'])
    
    imn=imnoise(uint8(im),'gaussian');
    imwrite(uint8(imn),['ref/',num2str(11) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(11) ,' (',num2str(i),').jpg'])
    
    imc=histeq(uint8(im));
    imwrite(uint8(imc),['ref/',num2str(12) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(12) ,' (',num2str(i),').jpg'])
    
    imm=medfilt2(uint8(im));
    imwrite(uint8(imm),['ref/',num2str(13) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(13) ,' (',num2str(i),').jpg'])
    
    im=imrotate(im,90);
    im2=imrotate(im2,90);
    imwrite(uint8(im),['ref/',num2str(4) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(4) ,' (',num2str(i),').jpg'])
    imn=imnoise(uint8(im),'gaussian');
    imshow(imn)
    imwrite(uint8(imn),['ref/',num2str(14) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(14) ,' (',num2str(i),').jpg'])
    
    imc=histeq(uint8(im));
    imwrite(uint8(imc),['ref/',num2str(15) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(15) ,' (',num2str(i),').jpg'])
    
    imm=medfilt2(uint8(im));
    imwrite(uint8(imm),['ref/',num2str(16) ,' (',num2str(i),').jpg'])
    imwrite(logical(im2),['GT/',num2str(16) ,' (',num2str(i),').jpg'])
end       