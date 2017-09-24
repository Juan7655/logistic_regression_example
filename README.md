# Logistic regression example

In this code, I've used the glogistic regression technique from machine learning to estimate the curve of classification
for the given dataset points. The data file Contains values for one variable, and so, the algorithm is implemented for 
single-variable datasets. Multi-variable algorithm will be coming soon. 

The decision curve represents the probability of
the variable being of one class or another (belongs or not, binary decision, computed as 0 or 1). As the value in the curve 
gets near 1, it means that the point has a higher probability of belonging to the classification. 

The curve general form has the following structure: 
`f(x) = 1 / (1 + e^-(ax + b))`

To run this code, you must have python installed. No other dependencies or libraries were used. You just have to download 
the dataset file and run the code. The console will output values of m and b for each step. Both files (code and dataset) 
must be in the same folder. Otherwise, you might have to change the path specified in en code qhere the dataset is imported.
