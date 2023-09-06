# CUSTOMER_CHURN
OBJECTIVE 

Develop a machine learning model to predict customer churn based on historical customer data. You will follow a typical machine learning project pipeline, from data preprocessing to model deployment. 

Data Preprocessing: 

No null values, mistyped values were found

Attributes containing categorical values: Gender 

Attributes containing different unique values in text format: Location  

Gender which has categorical values are replaced with binary digits (1 and 0) 

By using One hot coding Location attribute values are turned into different attributes
Intially, Location contains the follwing attributes as shown:
Location: ['Los Angeles' 'New York' 'Miami' 'Chicago' 'Houston'] 
After applying one hot coding technique, Los Angeles, New York, Miami, Chicago, Houston became different attributes in the dataset 

If newly generated attributes are in Boolean datatype and attributes other than the newly generated attributes are of integer data type, then their categorical values are replaced with binary digits (1 and 0). 

Finally, all the columns are changed into integer datatype. 

Other than some columns remaining are in the range of 0 to 1.  

Hence, the dataset is preprocessed and cleaned. 

Feature Engineering: 

CustomerID, Name rows are primarily not necessary. So, New data frame is created by keeping other rows.  

The attributes containing values more than 1 are normalized to the scale of 0 and 1. 

Model Building: 

Build a model using Artificial Neural Networks (ANN) in tensorflow/keras 

Scikit-learn, TensorFlow, Keras, NumPy, Pandas are used for coding. 

Accuracy, precision, recall, F1-score are calculated 

Deployment: 

We can use falsk (Python web framework) to deploy the code. 

Visualization: 

Histograms are used to represent the comparison between number of customers and feature  

Confusion matrix is used to represent the final result 

 

RESULT:

 Customer churn prediction is successfully completed for the given dataset. 

Testing and training datasets: 

The training dataset comprises 80% of the sample data, while the testing dataset consists of the remaining 20%. 


