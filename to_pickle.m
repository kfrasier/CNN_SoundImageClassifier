% load in a bunch of images saved as .mat matrices, 
inDir = 'E:\Data\CNN_images\half_hr_windows\single_label\test';
outDir = 'E:\Data\CNN_images\half_hr_windows\single_label';

outFileName = 'ts_features.mat';


fList = dir(fullfile(inDir,'*.mat'));
imageStack = [];
labelList = [];
nImages = length(fList);
fprintf('%d images found\n',nImages)
for iFile = 1:nImages
    
    inFileName = fList(iFile).name;
    thisFile = load(fullfile(inDir,inFileName));
    thisImage =  thisFile.ltsa_image; 
    if isempty(imageStack)
        % first time through, initialize imageStack to proper size
        nHigh = size(thisImage,1);
        nWide = size(thisImage,2);
        imageStack = zeros(nImages,nHigh,nWide);        
        labelList = zeros(nImages,10);
    end
    if nHigh == size(thisImage,1) &&  nWide == size(thisImage,2)
        % check that dimensions match previous images
        imageStack(iFile,:,:,1) = thisImage;
        % parse names to one-hot encoding
        labelTokens = regexp(inFileName,'-(\d+)','tokens');
        labelNums = str2num(char([labelTokens{:}]'));
        labelList(iFile,labelNums)=1;
    else
        fprintf('problem on file # %d: %s\n',iFile,inFileName)
        error('Error: figure dimensions are inconsistent')
    end
end
% transform images if desired 

% flatten
flatImageStack = reshape(imageStack,[nImages,nHigh*nWide]);

% v1 = reshape(imageStack(50,:,:),[1,nHigh*nWide]);
% v2 = flatImage(50,:);
% v1 should equal v2 at all points
% pickle
labelOutFileName = strrep(outFileName,'features','labels');
featOutFile = fullfile(outDir,outFileName);
labelOutfile = fullfile(outDir,labelOutFileName);
save(featOutFile,'imageStack', '-mat');
save(labelOutfile,'labelList','-mat');

fclose all
