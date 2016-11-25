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



images = samples.Part2SampleImagesInput;
imageModel = models.imageModel;



