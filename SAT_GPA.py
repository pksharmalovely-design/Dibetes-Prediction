import pandas as pd
import numpy as np
df = pd.read_csv("Simple_Linear_Regression.csv")
print(df.head())
X = df [['SAT']]  # Features
y= df ['GPA']      # Targets 
#Check for and handle any missing or duplicate values.
#Data Cleaning
#Check for missing values
print(df.isnull().sum())
#Check for duplicate rows
print (df.duplicated().sum())
#Drop duplicate rows if any
df=df.drop_duplicates ()
from sklearn.model_selection import train_test_split
#Split the data: 80% training and 20% testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
#Initialize the Linear Regression Model
model=LinearRegression()
#Train the Model on the Training Data
model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
# Predict GPA values for the test set
y_pred = model.predict(X_test)
# Calculate MSE (Mean Squared Error)
MSE = mean_squared_error(y_test,y_pred)
print(f'Mean Squred Error : {MSE:.2f}')
import matplotlib.pyplot as plt
#Plot the training data
plt.scatter(X_train,y_train,color='blue',label='Training data')
#Plot the testing data
plt.scatter(X_test,y_test,color='green',label='Testing data')
#Plot the regression libe
plt.plot(X_test,y_pred,color='red',linewidth=2,label='Regression line')
# Add Labels and legend
plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.legend()
plt.title('Simple Linear Regression : SAT Score Vs GPA')
#Show the plot
plt.show()
import numpy as np
import pandas as pd
def predict_gpa ():
    #Accept user input for SAT Score
    sat_score = input("Enter SAT score to predict GPA (or type 'exit' to stop): ")
    if(sat_score.lower()=="exit"):
        print("Prediction process stopped.")
        return
    #Convert Input to a Dataframe to match training format
    sat_score=float(sat_score)
    #Ensure column name matches training data
    sat_score_df=pd.DataFrame([[sat_score]], columns=['SAT'])
    #Predict GPA using the trained model
    predicted_gpa = model.predict(sat_score_df)[0]
    # Display the predicted GPA
    print (f"Predicted GPA for SAT score --->  {sat_score}:{predicted_gpa:.2f}")
#Input Prediction Function
predict_gpa()