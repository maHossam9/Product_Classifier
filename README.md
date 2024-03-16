The Product Classifier implements "Bag of Visual Words" algorithm to do the following :
extract features from the train images using sift extractor
and then uses these features to create visual vocabulary to group such features for each product,
so that later on when testing, the new products can be categorized based on extracted features that are matched to what the model had trained on

the used train dataset is randomly collected and hand labelled,
whilst test data is the screenshots from slash products
