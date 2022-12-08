from scipy import stats
import pandas as pd 
import math

 
def estimate(slope, intercept, x):
    return slope * x + intercept

def linear_regression_rmse(x_data, y_data):
    result = stats.linregress(x_data, y_data)
    slope = result.slope
    intercept = result.intercept

    estimated_y = []
    for x in x_data:
        estimated_y.append(estimate(slope, intercept, x))
    rMSE = 0
    for actual, pred in zip(y_data, estimated_y):
        rMSE = rMSE + ((actual-pred)*(actual-pred))
    rMSE = math.sqrt(rMSE/len(estimated_y))
    return rMSE



directory = "ProcessedData"
features = pd.read_csv(directory + "/features.csv")
labels = pd.read_csv(directory + "/labels.csv")
weights_df = features["weight"]
final_weights_df = labels["latest_weight"]
print("Starting weight and final weight linear regression rMSE: " + str(linear_regression_rmse(weights_df, final_weights_df)))

ages_df=features["age"]
weight_losses = []
for start, final in zip(weights_df, final_weights_df):
    weight_losses.append(start-final)
print("Age and net weight loss linear regression rMSE: " + str(linear_regression_rmse(ages_df, weight_losses)))

