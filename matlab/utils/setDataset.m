function dataset = setDataset(args)
dataset.config.bUnaryEnable = 0;        % bUnaryEnable=1 use point-wise unary similarity, otherwise not
dataset.config.bEdgeEnable = 1;         % bEdgeEnable=1 use edge-wise 2nd-order similarity, otherwise not
dataset.config.bSaveRandom = 0;         % not to save the random graphs. it is used when reproducing the exact same results
dataset.config.bCreateRandom = 1;       % if set to 0, will read the random graphs from files
dataset.config.affinityBiDir = 1;       % used for mOpt
dataset.config.bPermute = 1;            % set the ground truth by identity matrix, other choices are not used in demo

dataset.config.testType = args.dataset.testType;            % to control the test type: 'formal','massOutlier';
dataset.config.datasetType = args.dataset.datasetType;      % to control the affinity generate: 'sync','real';
dataset.config.visualization = args.dataset.visualization;	% whether to show the pairwise matching on images

if strcmp(dataset.config.datasetType,'real')
    dataset.datasetDir = args.dataset.datasetDir;                            
    dataName = [dataset.datasetDir,'\*.mat'];       
    dataList = dir(dataName);
    imageName = [dataset.datasetDir,'\*.png'];       
    imgList = dir(imageName);
    
    dataset.config.Sacle_2D = 0.05;
    dataset.config.totalCnt = length(dataList);
    dataset.config.graphCnt = min(args.dataset.graphCnt, length(dataList));  
    dataset.config.connect = 'delaunay';
    dataset.config.bGraphMatch = 0; 
    dataset.config.inCntType = 'all'; 
    dataset.config.category = 'none';
    dataset.config.nInlier = args.dataset.inlier; 
    dataset.config.nOutlier = args.dataset.outlier; 
    dataset.config.complete = 1;    
    nInlier = dataset.config.nInlier;
    nOutlier = dataset.config.nOutlier;
    
    dataset.data = cell(dataset.config.graphCnt,1);
    dataset.image = cell(dataset.config.graphCnt,1);
    for i = 1:dataset.config.totalCnt
        img_name = [imgList(i).folder,'\',imgList(i).name];
        image = imread(img_name);
        dataset.image{i} = image;
        [h, w, ~] = size(image);
        
        name = [dataList(i).folder,'\',dataList(i).name];
        data = load(name);
        inlier = data.pts_coord(:, 1:nInlier);
        nPoints = size(data.pts_coord, 2);        
        
        if nOutlier == 0
            dataset.data{i}.point = inlier(:,1:nInlier);
        else
            num_rand_outlier = args.dataset.random_outlier;
            max_x = max(inlier(1,:));   min_x = min(inlier(1,:));
            max_y = max(inlier(2,:));   min_y = min(inlier(2,:));
            rand_outlier = zeros(2,num_rand_outlier);
%             rand_outlier(1,:) = (rand(1,num_rand_outlier)-0.5)*1.5*(max_x-min_x)+(min_x+max_x)/2;
%             rand_outlier(2,:) = (rand(1,num_rand_outlier)-0.5)*1.5*(max_y-min_y)+(min_y+max_y)/2;
            rand_outlier(1,:) = mean(inlier(1,:)) + (rand(1, num_rand_outlier) - 0.5) * (max_x-min_x);
            rand_outlier(2,:) = mean(inlier(2,:)) + (rand(1, num_rand_outlier) - 0.5) * (max_y-min_y);
            
            if nPoints - nInlier < args.dataset.select_outlier
                error('The # points is less than # inliers and # selected outlier');
            else
                num_sel_outlier = args.dataset.select_outlier;
                randidx = nInlier + randperm(nPoints-nInlier);
                sel_outlier = inlier(:, randidx(1:num_sel_outlier));
            end
            
            dataset.data{i}.point = [inlier, sel_outlier, rand_outlier];
        end
        dataset.data{i}.shuffle = [1:nPoints, nPoints + 1: nPoints + nOutlier];
%         dataset.data{i}.shuffle = [randperm(nPoints), nPoints + 1: nPoints + nOutlier];
        dataset.data{i}.point = dataset.data{i}.point(:, dataset.data{i}.shuffle)';
    end
else
    dataset.config.Sacle_2D = 0.05;
    dataset.config.totalCnt = 32;
    dataset.config.graphCnt = 32;
    dataset.config.connect = 'fc';
    if strcmp(dataset.config.testType,'massOutlier')
        dataset.config.bGraphMatch = 0;         % set to 1 use random graphs, otherwise use random points as set in the MPM code/paper
        dataset.config.category = 'outlier';    % only outlier are supported here
        dataset.config.inCntType = 'exact';     % set 'exact' for "more outlier case"
        dataset.config.nInlier = 6;     dataset.config.nOutlier = 12;   dataset.config.deform = .05;
        dataset.config.density = 1;     dataset.config.complete = 1;	dataset.config.scale = .1;
    else
        dataset.config.bGraphMatch = 1;
        dataset.config.inCntType = 'all';       % set 'all' for "only a few outlier case"
        dataset.config.category = 'deform';     %'deform','outlier','density','complete'
        switch dataset.config.category
            case 'deform'
                dataset.config.nInlier = 10;    dataset.config.nOutlier = 0;    dataset.config.deform = 0.15;
                dataset.config.density = .9;    dataset.config.complete = 1;    dataset.config.scale = .05;
            case 'outlier'
                dataset.config.nInlier = 6;     dataset.config.nOutlier = 4;    dataset.config.deform = 0.0;
                dataset.config.density = 1;     dataset.config.complete = 1;    dataset.config.scale = .05;
            case 'density'
                dataset.config.nInlier = 10;    dataset.config.nOutlier = 0;    dataset.config.deform = 0.0;
                dataset.config.density = 0.5;   dataset.config.complete = 1;    dataset.config.scale = .05;
            case 'complete'
                dataset.config.nInlier = 10;    dataset.config.nOutlier = 0;    dataset.config.deform = 0.05;
                dataset.config.density = 1;     dataset.config.complete = 0.1;  dataset.config.scale = .05;
        end
    end
end

dataset.config.nodeCnt = dataset.config.nInlier+dataset.config.nOutlier;

dataset.config.edgeAffinityWeight = args.dataset.edgeAffinityWeight;     
dataset.config.angleAffinityWeight = args.dataset.angleAffinityWeight;
dataset.config.selectNodeMask = 1:1:dataset.config.nInlier+dataset.config.nOutlier;
dataset.config.selectGraphMask{1} = 1:dataset.config.graphCnt;