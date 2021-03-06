% Test image datasets (ImageCLEF or Office-Home) using resnet50 features
clear
img_dataset = 'office-31';  % 'image-clef' or 'office-home'

if strcmp(img_dataset,'image-clef')
    str_domains = {'c', 'i', 'p'};  
    addpath('D:\TLdata\imageCLEF_resnet50');  % You need to change this path
end
if strcmp(img_dataset,'office-home')
    str_domains = {'Art', 'Clipart', 'Product', 'RealWorld'}; 
    addpath('D:\TLdata\Office-Home_resnet50');    % You need to change this path
end
if strcmp(img_dataset,'office-31')
    str_domains = {'amazon', 'dslr', 'webcam'}; 
    addpath('D:\TLdata\office31_resnet50');    % You need to change this path
end
list_acc = [];
for i = 1 :length(str_domains)
    for j = 1 : length(str_domains)
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        fprintf('%s - %s\n',src, tgt);

        data = load([src '_' src '.csv']);
        fts = data(1:end,1:end-1);
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xs = zscore(fts, 1);
        Ys = data(1:end,end) + 1;
        
        data = load([src '_' tgt '.csv']);
        fts = data(1:end,1:end-1);
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = zscore(fts, 1);
        Yt = data(1:end,end) + 1;

        Xt = Xt(ind,:);
        Yt = Yt(ind);
        options.lambda=1;
        options.p=10;
        options.eta=0.1;
        [Acc1, ~] = SLMC_UDA(Xs',Ys,Xt',Yt,options);
        fprintf('Acc: %f\n',Acc1);
        

        list_acc = [list_acc;[Acc1]];

    end
end
mean(list_acc)
