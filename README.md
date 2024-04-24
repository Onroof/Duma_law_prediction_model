<div id="header" align="center">
    <h3>Predicting the bill's possibility of adoption in the parliament</h3>
</div>

<div id="description" align="left">
In this project, we trained machine learning and neural network models to predict the outcome of bill's consideration in the Russian State Duma. The dataset encompasses the laws' information and attached documents' texts for the period from October 24, 1994 to December 1, 2022.
<br>
The models demonstrated the following results: 94% accuracy (F1 weighted metric) when predicting based on the documents' texts attached to the bill and 87% accuracy when trained on the bill passport parameters Models trained only on the bill's texts showed an accuracy of 75.6%. 
 The most important factor influencing the prediction result was the governmental conclusion text. The second most important attribute was "Subject of the legislative initiative" with 31.5% of significance in prediction. 

</div>

### Information about the project
- 🧐 Goal: Prediction the outcome of bill's consideration in the Russian State Duma 
- 📊 Dataset: 27176 laws, 76 parameters (including texts)
- 🔮Preprocessing instrument: ru-Bert tiny
- 🛠 Models used: **Random Forest algorithm, Logistic regression, Neural network**
- 📈 Metrics: F1 weighted metric, F1 macro, Ballanced accuracy, ROC AUC 

