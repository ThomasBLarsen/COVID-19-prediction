# COVID-19-prediction
A repository aimed to predict the hospitalization of COVID-19 infected. So far it only fit's a modified Sigmoid to the dataset. This seem to work good in China where the number of new cases has decreased. For other countries, the Sigmoid is not constrained enough. The goal is to develope the code to be a rich predictor for the hospital capasity needed.

The repository uses the https://github.com/CSSEGISandData dataset.

The analysis does the following:
1. Read the dataset
2. Format it to a Python friendly format
3. Fit a Constrained modified Sigmoid to the dataset
4. Plot the results

The library depends on Pandas, Numpy, Scipy and Matplotlib
They can be installed by running 

'''pip3 install pandas numpy scipy matplotlib'''

Before you run the script, replace the "folder" variable with the path to the directory where you downloaded the dataser.
