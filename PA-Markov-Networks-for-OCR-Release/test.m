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


imageModel = models.imageModel;
pairwiseModel = models.pairwiseModel;
tripletList = models.tripletList;


#imageModel.ignoreSimilarity = true;

#ScoreModel(data.allWords, imageModel, [], []);
#ScoreModel(data.allWords, imageModel, pairwiseModel, []);
#ScoreModel(data.allWords, imageModel, pairwiseModel, tripletList)
#z = ComputeEqualPairwiseFactors(images, K);
#z1 = ComputePairwiseFactors(images, pairwiseModel, K)
#factors = ComputeTripletFactors (images, tripletList, K)

#imageModel.ignoreSimilarity = true;
allFactors = samples.Part6SampleFactorsInput;
K = models.imageModel.K;
F = 2;
top = ChooseTopSimilarityFactors (allFactors, F)

samples.Part6SampleFactorsOutput.var
top.var
#allFactors = BuildOCRNetwork (images, imageModel, pairwiseModel, tripletList);

%z = RunInference(allFactors);
%A = (65:90);
%S = char(A);
%A = A - 64;
%S(z)
%

