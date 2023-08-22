# Decision-tree from scratch
#### Description:

Implementation from scratch for Decision Trees. Currently only using the sample datasets from sklearn, I plan to extend the usage and enhance the performance for external datasets as well.

I decided to do a Decision Tree model since it very closely mirrors the Binary tree data structure which I have studied previously as part of my DSA studies. It can also be portrayed very easily both mentally and visually and as such, can be understood slightly easily than other algorithms.

The decision tree methods involve stratifying or segmenting the predictor space into a number of simpler regions. When making a prediction, we simply use the mean or mode of the region the new observation belongs to as a response value.


Since the splitting rules to segment the predictor space can be best described by a tree-based structure, the supervised learning algorithm is called a Decision Tree. Decision trees can be used for both regression and classification tasks.
The process of building one such tree involves two steps:
> Dividing the predictor space into several distinct, non-overlapping regions

> Predicting the most-common class label for the region any new observation belongs to

In order to split the predictor space into distinct regions, we use binary recursive splitting, which grows our decision tree until we reach a stopping criterion. Since we need a reasonable way to decide which splits are useful and which are not, we also need a metric for evaluation purposes.

In information theory, entropy describes the average level of information or uncertainty
We can utilise the same concept to calculate the total information gain for each possible split

In order to calculate the split’s information gain (IG), we simply compute the sum of weighted entropies of the children and subtract it from the parent’s entropy.
Equipped with the concepts of entropy and information gain, we simply need to evaluate all possible splits at the current growth stage of the tree (top-down/greedy approach), select the best one, and continue growing recursively until we reach a stopping criterion.

After finishing my implementation for the decision tree, I have then compared the results of my model with scikit-learn's model and found out that my model outperforms the module's model by an average of ~4% nearly everytime. I also decided to perform a visual analysis and plotted confusion matrices for both the models.
![Alt text](image-5.png)


## References
> https://towardsdatascience.com/the-complete-guide-to-decision-trees-17a874301448
> https://developers.google.com/machine-learning/decision-forests/decision-trees
