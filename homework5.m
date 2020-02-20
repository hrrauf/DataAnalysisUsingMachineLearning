% ECE 271A Homework #5 Problem 6

tic 

image = imread('cheetah.bmp');
padded_image1 = padarray(image, [3 3],'symmetric','pre');
padded_image2 = padarray(padded_image1, [4 4],'symmetric','post');
final_image = im2double(padded_image2);
[x,y] = size(final_image);

Prior_FG = 250/1303;
Prior_BG = 1053/1303;

training = load('TrainingSamplesDCT_8_new.mat');
FG = training.TrainsampleDCT_FG;
BG = training.TrainsampleDCT_BG;

zigzag = load('Zig-Zag Pattern.txt');
zigzag = zigzag + 1;

c = 1;
block_array = zeros(68850,64);
zigzag_array = zeros(1,64);
for m = 4:size(final_image,1) - 4
    for n = 4:size(final_image,2) - 4
        block = final_image(m-3:m+4,n-3:n+4);
        DCT = dct2(block);
        zigzag_array(zigzag) = DCT;
        block_array(c,:) = zigzag_array;
        c = c + 1;
    end
end

% Part A
%Initialize Parameters
C = 8;
dims = [1 2 4 8 16 24 32 40 48 56 64];
% 
mu_BG = zeros(5,64*C);
mu_FG = zeros(5,64*C);
sigma_FG = zeros(5,64*C);
sigma_BG = zeros(5,64*C);
pi_FG = zeros(5,C);
pi_BG = zeros(5,C);
    
for i = 1:5
    [mu_c_FG, sigma_c_FG,pi_c_FG] =  EMAlgorithm(C, FG);
    [mu_c_BG, sigma_c_BG,pi_c_BG] =  EMAlgorithm(C, BG);
   
    mu_BG(i,:) = mu_c_BG;
    mu_FG(i,:) = mu_c_FG;
    sigma_BG(i,:) = sigma_c_BG;
    sigma_FG(i,:) = sigma_c_FG;
    pi_BG(i,:) = pi_c_BG;
    pi_FG(i,:) = pi_c_FG; 
end

toc




%Probability of error
error_prob = zeros(25,length(dims));
for m=1:5;

    for n = 1:5
            
        for idx_dim = 1:length(dims)
            A = zeros(255,270);
            
            % mean assigning
            mean_BG = mu_BG(n,:);
            mean_FG = mu_FG(m,:);

            % sigma assigning
            sig_BG = sigma_BG(n,:);
            sig_FG = sigma_FG(m,:);

            % pi assigning 
            temp_pi_BG = pi_BG(n,:);
            temp_pi_FG = pi_FG(m,:);

            %% cal likelihood
            for u = 1:255 
                for v = 1:270
                    X = block_array(270*(u-1) + v,:);
                    %% EM likelihood
                    % BG                   
                    class = size(mean_BG,2)/64;
                    mean_c = zeros(class, dims(idx_dim));
                    sigma_c = zeros(class*dims(idx_dim), class*dims(idx_dim));
                    X = X(1:dims(idx_dim)); 

                    for j = 1:class
                        mean_c(j,:) = mean_BG((j-1)*64 + 1:(j-1)*64 + dims(idx_dim));
                        sigma_c((j-1)*dims(idx_dim) + 1:j*dims(idx_dim), (j-1)*dims(idx_dim) +1:j*dims(idx_dim)) = ...
                            diag(sig_BG((j-1)*64 + 1: (j-1)*64 + dims(idx_dim)));
                    end
                    prob_BG = 0.0;
                    for h = 1:class  
                        prob_BG = prob_BG + mvnpdf(X,mean_c(h,:),sigma_c((h-1)*dims(idx_dim) +1:h*dims(idx_dim), (h-1)*dims(idx_dim) +1:h*dims(idx_dim)))*temp_pi_BG(h);
                    end
                    
                    % FG                   
                    class = size(mean_FG,2)/64;
                    mean_c = zeros(class, dims(idx_dim));
                    sigma_c = zeros(class*dims(idx_dim), class*dims(idx_dim));
                    X = X(1:dims(idx_dim)); 

                    for j = 1:class
                        mean_c(j,:) = mean_FG((j-1)*64 + 1:(j-1)*64 + dims(idx_dim));
                        sigma_c((j-1)*dims(idx_dim) +1:j*dims(idx_dim), (j-1)*dims(idx_dim) +1:j*dims(idx_dim)) = ...
                            diag(sig_FG((j-1)*64+1: (j-1)*64 + dims(idx_dim)));
                    end
                    prob_FG = 0.0;
                    for h = 1:class  
                        prob_FG = prob_FG + mvnpdf(X,mean_c(h,:),sigma_c((h-1)*dims(idx_dim) +1:h*dims(idx_dim), (h-1)*dims(idx_dim) +1:h*dims(idx_dim)))*temp_pi_FG(h);
                    end
                    
                    %% predict
                    if Prior_FG * prob_FG > Prior_BG * prob_BG
                        A(u,v) = 1;
                    end
                end
            end
            
            % Errors
            mask = im2double(imread('cheetah_mask.bmp'));

            detect = 0;
            false = 0;
            for x = 1:255
                for y = 1:270
                    if mask(x,y) == 1 && A(x,y) == 0
                        detect = detect + 1;
                    end
                    if mask(x,y) == 0 && A(x,y) == 1
                        false = false + 1;
                    end
                end
            end

            % Output our results (errors, detection, etc...)
            detection_rate = (detect/sum(mask(:)==1))*Prior_FG;
            false_alarm = (false/sum(mask(:)==0))*Prior_BG;
            f_rate = ['False Alarm Rate: ' num2str(false_alarm)];
            d_rate = ['Detection Rate: ' num2str(detection_rate)];
            p_error = detection_rate+false_alarm;
            prob_error = ['Probability of Error: ' num2str(p_error)]

            error_prob(m*n,idx_dim) = p_error;
            size(error_prob)
        end
    end
    
    toc
end
% % 
% toc
% % 
% % Plot poe vs dimension
% 
% % for k = 1:5
% %     subplot(1,5,k);
% %     plot(dims,error_prob((k-1)*5 + 1,:),'r');
% %     hold on;
% %     plot(dims,error_prob((k-1)*5+2,:),'g');
% %     plot(dims,error_prob((k-1)*5+3,:),'b');
% %     plot(dims,error_prob((k-1)*5+4,:),'c');
% %     plot(dims,error_prob((k-1)*5+5,:),'m');
% %     legend({'FG Mixture 1', 'FG Mixture 2','FG Mixture 3', 'FG Mixture 4', 'FG Mixture 5'});
% %     imageName = 'BG Mixture %d and all FG Mixtures';
% %     name = sprintf(imageName,k);
% %     title(name); 
% %     xlabel('Dimensions'); ylabel('Probability of Error');
% %     hold off;
% % end

%% Part b)
% tic
% 
% new_C = [8];
% 
% new_error_prob = zeros(length(new_C),length(dims));
% 
% %Training and Testing
% for c  = 1:length(new_C)
% %    parameters
%     class = new_C(c);
%     [mean_c_BG, sigma_c_BG,pi_c_BG] =  EMAlgorithm(class, BG);
%     [mean_c_FG, sigma_c_FG,pi_c_FG] =  EMAlgorithm(class, FG);
%     mu_BG = mean_c_BG;
%     mu_FG = mean_c_FG;
%     sigma_BG = sigma_c_BG;
%     sigma_FG= sigma_c_FG;
%     pi_BG = pi_c_BG;
%     pi_FG = pi_c_FG; 
%         for idx_dim = 1:length(dims)
%             A = zeros(255,270);
% 
%             %cal likelihood
%             for u = 1:255 
%                 for v = 1:270
%                     X = block_array(270*(u-1) + v,:);
%                     %                  
%                     class_new = size(mu_BG,2)/64;
%                     mean_c = zeros(class_new, dims(idx_dim));
%                     sigma_c = zeros(class_new*dims(idx_dim), class_new*dims(idx_dim));
%                     X = X(1:dims(idx_dim)); 
% 
%                     for j = 1:class_new
%                         mean_c(j,:) = mu_BG((j-1)*64 + 1:(j-1)*64 + dims(idx_dim));
%                         sigma_c((j-1)*dims(idx_dim) +1:j*dims(idx_dim), (j-1)*dims(idx_dim) +1:j*dims(idx_dim)) = ...
%                             diag(sigma_BG((j-1)*64+1: (j-1)*64 + dims(idx_dim)));
%                     end
%                     prob_BG = 0.0;
%                     for h = 1:class_new  
%                         sigma_c((h-1)*dims(idx_dim) +1:h*dims(idx_dim), (h-1)*dims(idx_dim) +1:h*dims(idx_dim))
%                         prob_BG = prob_BG + (mvnpdf(X,...
%                             mean_c(h,:),...
%                             sigma_c((h-1)*dims(idx_dim) +1:h*dims(idx_dim), (h-1)*dims(idx_dim) +1:h*dims(idx_dim))))...
%                             *pi_BG(h);
%                     end
%                     
%                     %FG                   
%                     class_new = size(mu_FG,2)/64;
%                     mean_c = zeros(class_new, dims(idx_dim));
%                     sigma_c = zeros(class_new*dims(idx_dim), class_new*dims(idx_dim));
%                     X = X(1:dims(idx_dim)); 
% 
%                     for j = 1:class_new
%                         mean_c(j,:) = mu_FG((j-1)*64 + 1:(j-1)*64 + dims(idx_dim));
%                         sigma_c((j-1)*dims(idx_dim) +1:j*dims(idx_dim), (j-1)*dims(idx_dim) +1:j*dims(idx_dim)) = ...
%                             diag(sigma_FG((j-1)*64+1: (j-1)*64 + dims(idx_dim)));
%                     end
%                     prob_FG = 0.0;
%                     for h = 1:class_new  
%                         prob_FG = prob_FG + (mvnpdf(X,...
%                             mean_c(h,:),...
%                             sigma_c((h-1)*dims(idx_dim) +1:h*dims(idx_dim), (h-1)*dims(idx_dim) +1:h*dims(idx_dim))))...
%                             *pi_FG(h);
%                     end
%                     
%                   
%                     if Prior_FG * prob_FG > Prior_BG * prob_BG
%                         A(u,v) = 1;
%                     end
%                 end
%             end
%             
%             %Errors
%             mask = im2double(imread('cheetah_mask.bmp'));
% 
%             detect = 0;
%             false = 0;
%             for x = 1:255
%                 for y = 1:270
%                     if mask(x,y) == 1 && A(x,y) == 0
%                         detect = detect + 1;
%                     end
%                     if mask(x,y) == 0 && A(x,y) == 1
%                         false = false + 1;
%                     end
%                 end
%             end
% 
%             %Output our results (errors, detection, etc...)
%             detection_rate = (detect/sum(mask(:)==1))*Prior_FG;
%             false_alarm = (false/sum(mask(:)==0))*Prior_BG;
%             f_rate = ['False Alarm Rate: ' num2str(false_alarm)];
%             d_rate = ['Detection Rate: ' num2str(detection_rate)];
%             p_error = detection_rate+false_alarm;
%             prob_error = ['Probability of Error: ' num2str(p_error)];
% 
%             new_error_prob(c,idx_dim) = p_error;
%         end
%         toc
% end
% 
% 
% % % plot poe vs dimension for multiple values of C
% % figure;
% % hold on;
% % for w = 1:length(new_C)
% %     plot(dims,new_error_prob(w,:));
% % end
% % hold off;
% % legend({'C = 1', 'C = 2','C = 4','C = 8','C = 16','C = 32'});
% % xlabel('Dimension'); ylabel('Probability of Error');
% % title('Probability of Error vs Dimension with C = [1,2,4,...]');
% % 
% % figure;
% % semilogx(new_C,new_error_prob(:,7),'r');
% % title('Probability of Error vs Class = [1,2,4,...]');