import pandas as pd 

# get feature/label data
directory = "ProcessedData"
features = pd.read_csv(directory + "/features.csv")
labels = pd.read_csv(directory + "/labels.csv")

initial_weights = features["weight"]
heights = features["height"]
bmis = features["bmi"]
ages = features["age"]
genders = features["gender"]
final_weights = labels["latest_weight"]

men = 0
women = 0 
for gender in genders:
    if gender == 1.0:
        men = men + 1
    elif gender == 2.0:
        women = women + 1

print("number of men " + str(men))
print("number of women " + str(women))
print("Average initial weight: " + str(sum(initial_weights)/33015))
print("Average height: " + str(sum(heights)/33015))
print("Average BMI: " + str(sum(bmis)/33015))
print("Average age: " + str(sum(ages)/33015))

print("Average final weight: " + str(sum(final_weights)/33015))

