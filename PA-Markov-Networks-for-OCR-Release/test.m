#
# data manipulation
data = load('PA3Data.mat');
words = data.allWords{1};
words(1);
words(1).img;
words(1).groundTruth;


#model manipulation
models = load('PA3Models.mat');
models.imageModel;
models.imageModel.K;
models.imageModel.params;
models.pairwiseModel;
models.tripletList;
models.tripletList.chars;
models.tripletList.factorVal;


#sample manipulation
samples = load('PA3SampleCases.mat');
part1 = samples.Part1SampleImagesInput;
part1_img = part1(1).img;
part1_groundTruth = part1(1).groundTruth;



#images = samples
%imageModel = models.imageModel;

#[charAcc, wordAcc] = ScoreModel(data.allWords, models.imageModel, [], [])
#ComputeEqualPairwiseFactors (images, K)

images = samples.Part2SampleImagesInput;
imageModel = models.imageModel;
pairwiseModel = models.pairwiseModel;
tripletList = models.tripletList;
K = models.imageModel.K;

imageModel.ignoreSimilarity = true;
allFactors = BuildOCRNetwork (images, imageModel, pairwiseModel, tripletList);
RunInference(allFactors);
#ScoreModel(data.allWords, imageModel, [], []);
#ScoreModel(data.allWords, imageModel, pairwiseModel, []);
ScoreModel(data.allWords, imageModel, pairwiseModel, tripletList)
#z = ComputeEqualPairwiseFactors(images, K);
#z1 = ComputePairwiseFactors(images, pairwiseModel, K)
#factors = ComputeTripletFactors (images, tripletList, K)


%n = length(images);
%
%% If the word has fewer than three characters, then return an empty list.
%if (n < 3)
%    factors = [];
%    return
%end
%
%factors = repmat(struct('var', [], 'card', [], 'val', []), n - 2, 1);
%
%for i=1:n-2
%  factors(i).var = [i,i+1,i+2];
%  factors(i).card = [K,K,K];
%  factors(i).val = ones(prod(factors(i).card),1);
%  A = tripletList(:,:).chars;
%  v = tripletList(:,:).factorVal;
%  factors(i) = SetValueOfAssignment(factors(i), A, v);
%endfor





