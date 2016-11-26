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
K = models.imageModel.K;

#imageModel.ignoreSimilarity = true;
#allFactors = BuildOCRNetwork (images, imageModel, pairwiseModel, tripletList);
#RunInference(allFactors);
#ScoreModel(data.allWords, imageModel, [], []);
#ScoreModel(data.allWords, imageModel, pairwiseModel, []);
#ScoreModel(data.allWords, imageModel, pairwiseModel, tripletList)
#z = ComputeEqualPairwiseFactors(images, K);
#z1 = ComputePairwiseFactors(images, pairwiseModel, K)
#factors = ComputeTripletFactors (images, tripletList, K)


images = samples.Part5SampleImagesInput;
K = models.imageModel.K;
allFactors = ComputeAllSimilarityFactors(images, K)



