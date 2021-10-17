# ai-research-dev
CDS lecture notes, notebooks and assignments

# __Machine Learning__ ___(Beginners)___

### __Types of Machine Learning__

    - Supervised Machine learning
    - Unsupervised Machine Learning
    - Semi-supervised machine learning
    - Reinforcement Learning
        
### __Supervised Machine Learning__

1. Linear Regression
    - works on continuous data
    - uses predictors to predict the target value
    - oldest simplest and widely used ML algorithms
    - Regression line minimizes the {Sum of "Square of Residuals"} or "SSR"
    - Regression line is there fore also know as "Line of Best Fit"
    - a.k.a Ordinary Least Square (OLS)
    - __Types:__
        1. _Simple Linear Regression_
            - Linear Regression line equation Y=B0 + B1*X
        2. _Multiple Linear Regression_
            - Linear Regression line equation Y=B0 + B1*X1 + B2*X2 + B3*X3 + ....
    - __Properties:__
        1. _Linearity:_ There should be a linear relationship between dependent and independent variable
        2. _Multivariate normality:_ Linear regression analysis requires all variables to be multivariate normal. Residuals should be normal.
        3. _No or little collinearity:_ Linear regression assumes that there is little or no multicollinearity in the data. MultiCollinearity occurs when the independent variables are too highly correlated with each other. For example. In a data set of housing prices when you have sqft there is no need for length and width as other variables so it is usually ignored to avoid collinearity.
        
    - __Pros:__
        1. Linear regression is extremely simple method.
        2. It is easy and intuitive to understand.
        3. Widely used as many problems can be transformed into linear regression problems(e.g Polynomial regression. generalized Linear Model (GLM))
        
    - __Cons:__
        1. It assumes there is a linear relationship b/w predictor and target
        2. If data is intrinsically non-linear than it will not yield best results
        3. Sensitive to the anomalies in data
        4. Number of samples should be significantly more than the number of parameters other noise enters the model rather than relationship between variables
        
2. Logistic Regression
    - works on structured, labelled, data
    - used for classification
    - Works only on binary classification i.e. cannot classify more than 2 categories of target
    - Sigmoid curve is one of the example used in logistic regression
    - Logistic regression can make use of large number of features that include continuous and discrete variables and non-linear features
    - The dependant variable must be a binary variable
    - __Function:__
        1. 1/x+e^(-(ax+b))
        2. For multiple linear regression 1/x+e^(-(a0x0+a1x1+a2x2+b))
    - __Assumptions:__
        1. Binary logistic regression requires dependant variable must be binary
        2. For a binary regression the factor level 1 of the dependent variable should represent the desired outcome
        3. Only meaningful variables should be included
        4. The independent variable should be independent of each other i.e. little to no multicollinearity 
        5. Requires quite a large sample       
    - __Pros:__
        1. Powerful in deciding two classes
        2. More Robust i.e. independent variables does not have to be normally distributed
        3. It does not assume a linear relationship between the independent and dependent variables
        4. There is no homogeneity of variance assumed        
    - __Cons:__
        1. Limited outcome classification: Only does binary classification
        2. if wrong independent variables are supplied then the it will have little or no predictive value
        3. Models are vulnerable to overfitting / over confidence
    - __Applications:__
        1. Image Segmentation or Categorization
        2. Handwriting recognition
        3. Spam filtering - spam or not
        4. Cell image - cancer or not
        5. Production line part scan - good or defective
    
    
3. K-Nearest Neighbours

    - It works on structured labelled data
    - KNN is widely disposable in real-life scenarios since it is non-parametric
    - This means it does not make any underlying assumption about the distribution of data
    - data is classified based on its nearest neighbours
    - KNN works based on minimum distance from the query instance to the training samples to determine the K-Nearest-Neighbours
    - K = the number of nearest neighbours to check for our target
    - the higher the number of nearest neighbours out of the respective classes the target is classified of that class
    - for the larger data it is slow as the ALGO checks for distance of target with each and every class points 
    - its time complexity is O(n)
    - if there is a tie between two or more classes for a given K then we select the nearest class as our class for the target
    - The above points makes the computation cost very high
    - Needs to store all the data
    - must know we have have a meaningful distance function (rarely required)
    
    - __Application:__
        1. Concept search: searching for semantically similar documents(i.e. documents containing similar topics)
        2. Recommender systems: If a customer likes a particular item, then you can recommend similar items for them. Also in Ads display
        3. Facial recognition, Finger print detection
        
    - __Pros:__
        1. no assumption about distribution of data required
        
    
    - __Cons:__
        1. Computationally expensive
        2. The time complexity is O(n)


3. Decision Tree
    - it is a supervised ML algorithm considered to be one of the best and most widely used for classification problem
    - Decision tree works for both categorical and continuous input and output variable
    - Decision tree based methods empower predictive models with high accuracy, stability, and ease of interpretation
    - Types of Decision tree:
        1. __categorical Variable Decision tree:
            - Target variable is categorical
            - eg. is it gonna rain or not etc
        2. __Continuous Variable Decision tree:
            - Target variable is continuous
            - eg. BMI of a person
            - moderately ok for this particular purpose
            
    - __Decision Tree terminology:__
        1. __Root Node:__ it represents the entire population or sample and this further gets divided into two or more homogeneous set
        2. __Splitting:__ It is a process of dividing a node into two or more sub-nodes
        3. __Decision Node:__ When a sub-node splits into further sub-nodes, then it is called decision node
        4. __Leaf/Terminal Node:__ Nodes with no children is called leaf or terminal node
        5. __Pruning:__ When we reduce the size of decision trees by removing nodes (opposite of splitting) its called pruning
        6. __Branch / sub-tree:__ A sub section of decision tree is called a branch or sub-tree
        7. __Parent and child Node:__ A node, which is divided into sub-nodes is called parent node of sub-nodes and sub-nodes are called child of parent node
        
     - __Working:__
         - Decision tree builds classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes
         - To find the root node we calculate something called a gini factor
         - gini-factor of a child node = 1- P(A)^2 - P(B)^2 - .... - A, B, .... being the events that can happen in a node
         - gini factor of parent = weighted avg of gini-factor of all children
         
     - __Pros:__
         - Easy to understand: The output is very easy to understand even for people from non analytical background
         - Less data cleaning required: It is not influenced by outliers or missing values to a fair degree
         - Non parametric method: It means it has no prior assumptions about space distribution or classifier structure
         - Data type is not a constraint: It can handle both numerical as well as categorical variables
         - Decision Tree Versatility: It can be customized for variety of situations
     
     - __Cons:__
         - Overfitting: Over fitting is one of the most practical difficulty for this model
         - Low accuracy for continuous variables: While working with continuous numerical variables it looses information when it categorizes variables into different categories
         - Unstable: A small change in the data can lead to a large change in the structure of the optimal decision tree. This is self-evident as it relies on gini-factor which is sensitive to change
     
     - __Application:__
         - Business decision support: Building tree with past business data to support decision making for new product launches, features, analyzing customer satisfaction factors etc.
         - Fraudulent Statement Detection: It can make a significant contribution for the detection of FFS due to a higly accurate rate
         - Healthcare Management: It is useful tool to discover and explore hidden information in health-care management, diagnose health condition
         - Pharamcology: For use in drug analysis
         - Agriculture: Application of a range of ML methods to problems in agriculture and horticulture
         
     - __Optimization:__
         1. Overfitting problem: A tree that is too large risks overfitting the training data and poorly generalizing to new samples
         2. Pruning:
             - A common strategy is to grow the tree until each node contains a small number of instances then use pruning to remove nodes that do not provide additional information
             - Pruning should reduce the size of a learning tree without reducing predictive accurayc as measured by cross-validation set
         3. Setting constraints on Tree size: Imposing contraints such as maximum depth of tree, Minimum samples for a node split etc are commonly used techniques
         

4. Random Forest
    - Resolves overfitting in decision trees
    - It is a supervised ML algorithm that works on structured labelled data
    - It can be used for classification and regression but it may lag behind in regression. best used for classification
    - It is a type of ensemble models because it is not a single algo, it is an implementation of multiple Decision trees
    - It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes
    - Random forest corrects for decision tree's habit of overfitting to their training set
    - It does this by creating random subsets of the features and building smaller shallower trees using the subsets and then ti combines the subtress
    
    - __Pros:__
        1. Solves the overfitting problem generally observed with decision trees
        2. Random Forest can be used for both classification and regression
        3. RF are extremely flexible and have a very high accuracy
        4. RF default hyper-parameters often produce a good prediction result, so considered easy/handy algorithm
        5. RF maintains accuracy even when a large proportion of data are missing
        
    - __Cons:__
        1. RF is complex, harder, time-consuming to construct than decision trees
        2. Large number of trees can make the algorithm to slow down and ineffective for real-time predictions
        3. No interpretability. Random Forest model is not interpretable as Decision tree
        
    - __Application:__
        1. Banking: Customer segmentation - loyal or fraud customers
        2. Health care: Identifying the treatment/medicine based on history and symptoms
        3. E-Commerce: Recommended products and customer experience
        4. Stock Market: Predicting stock market portfolio performance
        
5. Support Vector Machine

    - Requires structured labelled data
    - It can be used for both classification as well as regression challanges. However, it is mostly used in classification problems
    - An SVM model is a representation of the examples as point in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
    - We try to plot our data in such a way that gap is maximum enough to fit a hyper-plane into it.
    - In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernal trick, implicitly mapping their inputs into higher dimensional feature space.
    - hyperplanes are planer formations inside the n dimensional feature space that separate the feature space into separte spaces
    - For e.g. a line can be a separator into a 2 dim feature space
    - kernal trick is mapping a data into our required scale for example a 1 dim feature space can be transformed into 2 dim space by sqaring the data points thereby us being able to draw a separator line between different clusters of data
    - If the data is linearly separable we can directly apply  SVM
    - If not then we use a kernal trick then use SVM
    - It constructs a hyperplane or a set of hyperplane in a high or infinite dim space which can be used for classification, regression or other tasks like outlier detection
    - Distance between hyperplane and datapoint is called margin
    - The best hyperplane is chosen which has highest margin among the set of hyperplane
    - This means when the points are separated by highest possible margin with hyperplane our model classifies clearly
    - For e.g 2D plane will have line as hyperplane, 3D plane will have a plane and 4D and so on will have a hyper plane
    - C value and Gamma value are most imp hyperparameters to watch for
    - if C is too small the model will over generalize and if C is too high the model will overfit
    
    - __Pros:__
        1. Very good at dealing with higher dimensional data
        2. Works well with smaller datasets
        3. It is useful for both linearly separable(hard margin) and non linearly separable(soft margin) data
        4. Guaranteed Optimality: Due to the nature of Convex optimization, the solution is guaranteed to be the global minimum

    - __Cons:__
        1. Picking right kernel and parameters is computationally intensive
        2. In NLP, structured representations of text yield better performances. Sadly, SVMs can not accomodate such structures(word embeddings)
        
    - __Applications:__
        1. Face Detection
        2. Text and hypertext detection
        3. Classification of images
        4. Bioinformatics
        5. Protein fold and remote homology detection
        6. Handwriting recognition
        

6. XGBoost
    - It is based on decision tree
    - it takes input from one decision tree and try to fix error using another decision tree with the output of first as input of second
    - performs excellently in both classification and regression problems
    - XGBoost is an implementation of gradient boosted decision trees designed for speed and performance
    - XGBoost is a popular algorithm, widely adapted both in production and in machine learning competition winning models
    - 
        
        
### __Unsupervised Machine Learning__

1. K-means Clustering

    - K number of centroids are selected from the messy data at ramdom
    - data is classified into k clusters based on each data point's euclidean distance with centroids
    - Now another K number of centroids are selected from data with each centroid being center most point of the previously created k cluster
    - data is classified into another set of k clusters based on each data point's euclidean distance with centroids
    - at the end it returns K clusters that it found after iteratively centering the centroids to found clusters
    - The algorihm works iteratively to assign each data point to one of K groups based on the features that are provided
    - Data points are clustered based on their feature similarity
    - __Steps of K means Clustering__
        1. Clusters the data into k groups where k is predefined
        2. select k points at random as cluster centers
        3. Assign objects to the closest cluster center according to the euclidean distance function
        4. Calculate the centroid or mean of all objects in each cluster
        5. repeat steps 2,3,4 until the same points are assigned to each cluster in consecutive rounds
        6. The repeatation happens until the centroids do not move any further
        
    - __Pros:__
        1. Practically works well even if some assumptions are broken
        2. Simple and easy to implement
        3. Fast and efficient in implementation
        
    - __Cons:__
        1. __Uniform Effect:__ Often produces clusters with relatively uniform size even if the input data have different cluster size
        2. __Different densities:__ May work poorly with the clusters with different densities but spherical shape
        3. Highly Sensitive to outliers
        4. K value needs to be known before K-means clustering
        
    - __Applications:__
        1. Market segmentation
        2. Computer Vision
        3. Geostatistics
        4. Astronomy and Agriculture
        5. Feature Learning
        

### Neural Networks
   #### Math required to understand deep learning
    1. Linear algebra (Upto Eigen values and Eigen vectors and vector algebra)
    2. Statistics and probability (all concepts in NPTEL statistics and probability for deep understanding)
    3. Calculus (ODE, Partial differential equations, Transforms etc)

    - Geofrey Hilton is the godfather of deep learning - person who came up with back propagation
    - One way to create neural network is to randomly initialize the weights and bias of your inputs for first layer of neurons
    - We have an error or a cost function to calculate the error 
    - We use gradient descent to reduce the error shown by error function
    - We chose the global minima values of our weights and bias as our weights and bias for our neural network layer
    - Shallow and deep neural networks
    - Feed Forward propagation and back propagation are used to evaluate how bad or good model performs and weights and bias are optimized to reach the least error possible using these two methods
    - Activation function is used to generate inputs out of the z for next neural layer
    - Sigmoid function, ReLU etc are used as activation function
    
1. Artificial Neural Network
    - A supervised ML algorithm
    - Input layer - It contains those units which receive input from the outside world on which network will learn, recognize about or otherwise process
    - Output layer - It contains units that respond to the information about how its learned any task
    - Hidden layers - These units are in between input and output layers. The job of hidden layer is to transform the input into something that output unit can use in some way
    - Most neural networks are fully connected that is to say each hidden neuron is fully connected to every neuron in its previous layer and to the next layer
    - Stopping criteria
        1. Max iteration (no of times back propogation can be applied)
        2. validation score not improving 
        3. Errors below threshold (if error value is below our threshold value i.e. f(x) >= f(c) for every x in D(domain) f(c) is the global minima)
    - Threshold is usually set manually but it depends on which loss function or cost function we are using
    - sigmoid function -> sigma(z) = 1/(1+e^(-1))
    - ReLU function (Rectified Linear Unit) -> R(z) = max(0, z) if output < 0 its taken as 0 else it is taken as z
    - ReLU can be analogus to half wave rectification and it is also known as ramp function
    - Activation function
        1. Classification
            - softmax (multiclass classification)
            - Sigmoid
        2. Regression
            - Linear
    - ANN architectures
        1. Single layer perceptron
        2. Radial Basis Network(RBN)
        3. Multilayer Perceptron
        4. Recurrent Neural Network
        5. LSTM(Long Short Term Memory) Recurrent Neural Network
        6. Hopfield Network
        7. Boltzmann Machine
    - __Pros:__
        1. Good to model the non-linear data with large number of input features
        2. Widely used for high dimensionality data
        3. ANN can generalize - After learning from the initial inputs and their relationships
        4. ANN does not impose any restrictions on input variables(like how they should be distributed)
        
    - __Cons:__
        1. NNs are useable only for numerical inputs, vectors with constant number of values and datasets with non missing data
        2. ANNs are computationally expensive
        3. Black box, difficult to understand the modeling
        4. Multi-layer neural networks are usually hard to train
        5. Requires significant training data for model ANN compared to other ML algorithms
        
    - __Application:__
        1. Image Processing and Character recognition: Given its ability to take in a lot of inputs, process them to infer hidden as well as complex, non linear relationships, ANNs play a big role in image and character classification
        2. Forecasting: Forecasting is required extensively in everyday business decisions(e.g sales, financial allocation between products, capacity utilization etc), in economic and monetary policy, in finance and stock market
        3. Game playing and decision making: e.g Backgammon, chess, poker
        4. Medical Diagnosis: e.g. Cancer detection 
        5. To accelerate reliablity analysis of infrastructures subject to natural disasters
        
2. Convolution Neural Networks:
    - Images are made up of grid of pixels
    - Grayscale images
        - Each pixel can have any value between 0 - 255 where 0 = black and 255 = white and rest all in between 0 and 255 are shades of grey
    - Colored images
        - In a colour image each pixel comprises of 3 values each ranging from 0-255 0-255 0-255 i.e R G B
    - Covulution - usage of filters to detect different features of an image
    - CNN detects what filters to use for what features automatically
    - ReLU activation function is used for avoiding linearity in CNN so any negative value is mapped to 0
    - Removing linearity requires understanding of deep convulution neural network
    - Pooling - It is used to generalize the model
    - Most popular method of pooling is called max-pooling
    - It just takes the maximum value from the provided matrix so as to generalize any pixel matrix
    - each iteration of conv-relu-max-pooling is called a pass_n where n -> (0, n)
    - After all the above processes we flatten the image
    - After flattenning we create an ANN to categories - here we send our images to be classified
    - Rewatch the video for complete explaination in case you forget why convultuion -> Activation -> pooling -> repeat -> flatten -> Dense (ANN) -> activation -> Droput or training generators, testing generators, model.fit(test_batch_size, batch_size, epochs etc)
        
        
### Hyperparameter tuning

1. GridSearchCV
    - Slower
    - evaluate all provided parameters

2. RandomSearchCV
    - faster
    - does not evaluate all parameters but instead chooses random ones for evaluation
    
    
### Natural Language Processing
    
    - Bag Of Words - old and not so popular method
        - It converts a sentence into its numerical representation
        - The flaw here is that all words are given same importance value
    - TF-IDF (Term Frequency - Inverse Document Frequency)
        - TF - how frequent is each word in each sentence in a list of sentences
        - TF = No of occurance of a word in sentence / total number of words in sentence
        - IDF = log(Total number of sentences / no. of sentences containing the word)
        - TO calculate IDF first arrange all the words in order of their frequencies
        - for each word take the product of its TF and IDF value
        