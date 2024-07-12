clc
close all
%
load snn.mat
k=1;
disp('------------------------------------------')
disp('     Train RESULTS:   ');
for i=1:1  % max=16
    for j=1:26
        str1=['ref/',num2str(i) ,' (',num2str(j),').jpg'];
        im=imread(str1);
        str2=['GT/',num2str(i) ,' (',num2str(j),').jpg'];
        GT=imread(str2);
        subplot(1,3,1), imshow(im), title('input')
        subplot(1,3,2), imshow(GT), title('Ground Truth')

        GT(GT>0)=1;
        [C,scores] = semanticseg(im,net);
        B=(C=='Cancer');
        
        nResult=sum(sum(B==1));
        nGT=sum(sum(GT==1));
        nUNI=0;
        for w=1:numel(GT)
            if B(w)==1 && GT(w)==1
                nUNI=nUNI+1;
            end
        end
        k
        Qc= nUNI/nGT * nUNI/nResult
        
        acc= sum(sum(B==logical(GT)))/numel(GT)
        accuracy(k)=acc;
        Q(k)=Qc;
        k=k+1;
        subplot(1,3,3), imshow(B), title('SNN result')
        pause;
    end
end

disp('------------------------------------------')
disp('     TEST RESULTS:   ');
for i=1:1    % max=16
    for j=27:33
        str1=['ref - test/',num2str(i) ,' (',num2str(j),').jpg'];
        im=imread(str1);
        str2=['GT - test/',num2str(i) ,' (',num2str(j),').jpg'];
        GT=imread(str2);
        subplot(1,3,1), imshow(im), title('input')
        subplot(1,3,2), imshow(GT), title('Ground Truth')

        GT(GT>0)=1;
        [C,scores] = semanticseg(im,net);
        B=(C=='Cancer');
        
        nResult=sum(sum(B==1));
        nGT=sum(sum(GT==1));
        nUNI=0;
        for w=1:numel(GT)
            if B(w)==1 && GT(w)==1
                nUNI=nUNI+1;
            end
        end
        k
        Qc= nUNI/nGT * nUNI/nResult
        
        acc= sum(sum(B==logical(GT)))/numel(GT)
        accuracy(k)=acc;
        Q(k)=Qc;
        k=k+1;
        subplot(1,3,3), imshow(B), title('SNN result')
        pause;
    end
end
disp('----------------------------------');
total_accuracy= mean(accuracy)
total_Q=nanmean(Q)

std_accuracy= std(accuracy)
std_Q=nanstd(Q)
