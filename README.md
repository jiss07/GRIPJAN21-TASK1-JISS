
 # Author: JISS PETER , Data Science Intern @ The Spark Foundation
The task related dataset is available on the web url http://bit.ly/w-data. This dataset can be downloaded locally or can access directly in the code
## **Data Science & Business Analytics Internship at The Sparks Foundation.**

#GRIPJAN21
## Task-1 : Prediction using Supervised ML (Level - Begniner)

### **Task Description**
In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. 
For Predicting the percentage of an student based on the no. of study hours  I have used python and its built in libraries such as numpy,pandas, linear regression techniques as well as used matplot lib library for ploting the predicted results as well as the variable ploting.

#### The intent of this task related coding work is to answer the original question "What will be predicted score if a student studies for 9.25 hrs/ day?"

```python
# Importing all libraries required in the task related activities
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
```


```python
# Reading data from remote link
url = "http://bit.ly/w-data"
my_data = pd.read_csv(url)
print("Data imported successfully")

my_data.head(10)
```

    Data imported successfully
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.2</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.5</td>
      <td>60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.3</td>
      <td>81</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.7</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



Now lets try to plot the given dataset using matplotlib library as a 2-D graph and see if we can manually find any relationship between the data.


```python
# Plotting the distribution of scores
my_data.plot(x='Hours', y='Scores', style='D')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()
```


![png](GRIP-Task1-PredictionUsingSupervisedML_files/GRIP-Task1-PredictionUsingSupervisedML_5_0.png)


**From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

### **Preparing the data**

The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).


```python
X = my_data.iloc[:, :-1].values  
y = my_data.iloc[:, 1].values  
```

Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:


```python
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 
```

### **Training the Algorithm**
We have split our data into training and testing sets, and now is finally the time to train our algorithm. 


```python
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")
```

    Training complete.
    


```python
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()
```


![png](GRIP-Task1-PredictionUsingSupervisedML_files/GRIP-Task1-PredictionUsingSupervisedML_13_0.png)


### **Making Predictions**
Now that we have trained our algorithm, it's time to make some predictions.


```python
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
```

    [[1.5]
     [3.2]
     [7.4]
     [2.5]
     [5.9]]
    


```python
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>16.884145</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>33.732261</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>75.357018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>26.794801</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>60.491033</td>
    </tr>
  </tbody>
</table>
</div>



### **Evaluating the model**

The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.


```python
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
```

    Mean Absolute Error: 4.183859899002975
    

# Now Lets try to find out the accuracy of our Linear Regression Model


```python
from sklearn import metrics
print('Accuracy of the model is : ',round((100*metrics.r2_score(y_test,y_pred)),2),'%')
```

    Accuracy of the model is :  94.55 %
    

# Lets try to do final prediction of the score for a student who studies 9.25 hours.
#### The orignial Question tobe asnwered for the completion of this GRIP Internship-Task 1 was 
#### "What will be predicted score if a student studies for 9.25 hrs/ day?"


```python
hours= 9.25

rows, cols= (1,1)
arr=[[hours]*cols]*rows

my_pred=regressor.predict(arr)

print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(my_pred[0]))
```

    No of Hours = 9.25
    Predicted Score = 93.69173248737535
    

# Conclusion
### From the above we can infer that if a student studied for 9.25 hours/daily then the student will secure 93.69 marks.
## Completed Task 1.
### Thank you for going through this solution
