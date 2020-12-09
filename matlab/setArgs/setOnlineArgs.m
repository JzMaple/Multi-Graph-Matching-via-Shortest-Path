function args = setOnlineArgs
args.test_cnt = 10;
args.metric_eye = 0;
args.baseGraphCnt = 25;

% pairwise solver
args.pairwise.method = 'RRWM';
args.pairwise.iterMax = 10;
args.pairwise.bDisc = 1;
args.pairwise.iccvIterMax = 1;

% dataset
args.dataset.testType = 'formal';                   % to control the test type: 'formal','massOutlier';
args.dataset.datasetType = 'real';                  % to control the affinity generate: 'sync','real';
args.dataset.visualization = 0;                     % whether to show the pairwise matching on images
if strcmp(args.dataset.testType,'sync') && args.dataset.visualization == 1
    error('Cannot visualize on synthetic dataset.');
end
args.dataset.edgeAffinityWeight = 0.9;
args.dataset.angleAffinityWeight = 0.1;
if args.dataset.edgeAffinityWeight + args.dataset.angleAffinityWeight ~= 1
    error('The sum of edge weight and angle weight should be 1.')
end
% real dataset
args.datast.scale_2D = 0.1;
args.dataset.dataset = 'WILLOW-ObjectClass';
args.dataset.class = 'Car';
args.dataset.datasetDir = ['data\', args.dataset.dataset, '\', args.dataset.class];
args.dataset.graphCnt = 40;
args.dataset.inlier = 10;
args.dataset.outlier = 2;
args.dataset.random_outlier = 2;
args.dataset.select_outlier = 0;
if args.dataset.random_outlier + args.dataset.select_outlier ~= args.dataset.outlier
    error('Outlier number should equal to the sum of random ouliers and selected outliers.')
end

% Parameters for CAO
args.caoconfig.initConstWeight = .3;         
args.caoconfig.constStep = 1.0;             
args.caoconfig.constWeightMax = 1;
args.caoconfig.iterRange = 6;
args.caoconfig.constIterImmune = 2;

% Parameters for MGM-Floyd
args.floydconfig.alpha = 0.3;

% Parameters for IMGM
args.imgmconfig.n = args.dataset.inlier + args.dataset.outlier;
args.imgmconfig.visualization = 0;
args.imgmconfig.iterMax = 6;
args.imgmconfig.method = 1;
