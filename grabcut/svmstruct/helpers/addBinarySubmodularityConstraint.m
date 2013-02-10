function svm = addBinarySubmodularityConstraint(svm, num_unary, num_pair)

% TODO: review this function and document!

G = zeros(num_pair,size(svm.qp.G,2));
for i=1:num_pair
    G(i,num_unary+i) = 1;
end
h = zeros(num_pair,1);

svm.qp.G = [svm.qp.G; G];
svm.qp.h = [svm.qp.h; h];
