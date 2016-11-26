function factor = ComputeSimilarityFactor (images, K, i, j)
% This function computes the similarity factor between two character images
% in one word --- which characters is given by indices i and j (a
% description of how the factor should be computed is given below).
%
% Input:
%   images: A struct array of character images from one word.
%   K: The alphabet size.
%   i,j: The scope of that factor. That is, you should construct a factor
%     between characters i and j in the images array.
%
% Output:
%   factor: The similarity factor between these two characters. For any
%     assignment C_i != C_j, the factor value should be one. For any
%     assignment C_i == C_j, the factor value should be
%     ImageSimilarity(I_i, I_j) --- ie, the computed value given by
%     ImageSimilarity.m on the two images.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

factor = struct('var', [], 'card', [], 'val', []);

% Your code here:
factor = struct('var', [], 'card', [], 'val', []);
factor.var = [i, j];
factor.card = [K, K];
factor.val = ones(1, prod(factor.card));

A = IndexToAssignment(1:prod(factor.card), factor.card);
i_equal_j = A(find(A(:,1) == A(:,2)), :);

sim = ImageSimilarity(images(i).img, images(j).img);
v = zeros(length(i_equal_j),1) + sim;
factor = SetValueOfAssignment(factor,i_equal_j,v);



end

