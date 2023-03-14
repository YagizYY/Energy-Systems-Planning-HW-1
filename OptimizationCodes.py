#%%
import os

# %%
import pandas as pd
import numpy as np
from pulp import *
import openpyxl
import matplotlib.pyplot as plt

# %%
# import sys
# print(sys.prefix)

# %%

weekly = pd.read_excel("C:/Users/yagiz.yaman/Desktop/IE453-HW1/weekly_data.xlsx")

# %%
power_curve_wind = {
    "speed (m/s)": list(range(4, 26)),
    "Power (MW)": [
        0.043,
        0.131,
        0.25,
        0.416,
        0.64,
        0.924,
        1.181,
        1.359,
        1.436,
        1.481,
        1.494,
    ]
    + [1.5] * 11,
}

power_curve = pd.DataFrame(power_curve_wind)

# %%
def ms_to_kmh(speed_ms):
    # convert m/s to km/h
    speed_kmh = speed_ms * 3600 / 1000
    return speed_kmh


power_curve["speed (km/h)"] = power_curve["speed (m/s)"].apply(ms_to_kmh)

# %%
# merge two dataframes

my_new = weekly.merge(power_curve, left_on="Wind Speed (km/h)", right_on="speed (km/h)")

my_new["GeneratedPower (W)"] = my_new["Power (MW)"] * 10**6

my_new["GeneratedEnergy (Wh)"] = my_new["GeneratedPower (W)"] * 1

my_new = my_new.drop(
    ["speed (km/h)", "speed (m/s)", "Power (MW)", "GeneratedPower (W)"], axis=1
)

my_new = my_new.sort_values(["Date (Y-M-D-H)"])

my_new = my_new.reset_index(drop=True)


# %%


# Assuming the 'GeneratedEnergy(Wh)' column is in my_new dataframe
my_new.plot(y="GeneratedEnergy (Wh)")

# Add title and labels
plt.title("Energy Generation of One Regular Wind Turbine")
plt.xlabel("The Hour of the Week")
plt.ylabel("Energy Generated (Wh)")

# Set the x-axis tick labels
n = 10  # number of tick labels
tick_labels = [
    int(i)
    for i in pd.Series(range(len(my_new))).quantile(q=[i / (n - 1) for i in range(n)])
]  # compute tick labels
plt.xticks(tick_labels, [str(i + 1) for i in tick_labels])  # set tick labels

# Disable scientific notation on y-axis
plt.ticklabel_format(style="plain", axis="y")

# Remove legend
plt.legend().remove()

# Show the plot
plt.show()


# %%
############################# PART A #############################
##### GIVEN STORAGE SIZE & VARIABLE NUMBER OF WIND GENERATORS #####
# initialize model
model = LpProblem(name="MinimizeCost", sense=LpMinimize)

# Define decision variables
wind_generator = LpVariable("wind_generator", lowBound=0, upBound=None, cat="Integer")

wind_used = LpVariable.dicts(
    "wind_used", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

wind_stored = LpVariable.dicts(
    "wind_stored", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

storage_discharged = LpVariable.dicts(
    "storage_discharged",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)


wind_curtailed = LpVariable.dicts(
    "wind_curtailed",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

storage_level = LpVariable.dicts(
    "storage_level",
    [i for i in list(range(0, len(my_new.index) + 1))],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

cost_turbine = 2000000
gamma = 0.85
storage_capacity = 10**6
interest = 0.5 / 100
lifetime_wind = -52 * 25

# Define Objective Function
model += (
    interest / (1 - ((1 + interest) ** (lifetime_wind))) * cost_turbine * wind_generator
)

# Define Constraints

# (1)
for i in my_new.index:
    model += wind_used[i] + storage_discharged[i] == my_new["Demand(Wh)"][i]

# (2)
for i in my_new.index:
    model += (
        my_new["GeneratedEnergy (Wh)"][i] * wind_generator
        == wind_used[i] + wind_curtailed[i] + wind_stored[i]
    )

# (2.1)
storage_level[0] = 0

# (3)
for i in my_new.index:
    model += (
        storage_level[i + 1]
        == storage_level[i]
        + gamma * wind_stored[i]
        - (1 / gamma) * storage_discharged[i]
    )

# (4)
for i in my_new.index:
    model += storage_level[i] <= storage_capacity

# Solve model
model.solve()


with open("decision_variable_values.txt", "w") as f:
    for v in model.variables():
        f.write(f"{v.name} = {v.varValue}\n")

print("Buy {} of wind turbines.".format(wind_generator.varValue))
print("Objective function value:", pulp.value(model.objective))

# Create a list of variable names and sort them numerically
var_names = sorted(
    [v.name for v in model.variables()],
    key=lambda x: (int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1, x),
)

# Create an empty dictionary to hold the values of the decision variables
var_dict = {}

# Loop through the decision variables and populate the dictionary
for v in model.variables():
    var_dict[v.name] = v.varValue

# Convert the dictionary to a pandas dataframe and sort the rows by variable name
df = pd.DataFrame.from_dict(var_dict, orient="index", columns=["Value"]).loc[var_names]

# %%
############################# PART A #############################
##### GIVEN NUMBER OF WIND GENERATORS & VARIABLE STORAGE SIZE #####

# initialize model
model = LpProblem(name="MinimizeCost", sense=LpMinimize)

# Define decision variables
storageCap = LpVariable("storageCap", lowBound=0, upBound=None, cat="Integer")

wind_used = LpVariable.dicts(
    "wind_used", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

wind_stored = LpVariable.dicts(
    "wind_stored", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

storage_discharged = LpVariable.dicts(
    "storage_discharged",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)


wind_curtailed = LpVariable.dicts(
    "wind_curtailed",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

storage_level = LpVariable.dicts(
    "storage_level",
    [i for i in list(range(0, len(my_new.index) + 1))],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

wind_generator = 10
cost_storage = 0.60
gamma = 0.85
interest = 0.5 / 100
lifetime_wind = -52 * 25
lifetime_storage = -52 * 20
cost_turbine = 2000000

# Define Objective Function

model += (
    interest / (1 - ((1 + interest) ** (lifetime_storage))) * cost_storage * storageCap
    + interest
    / (1 - ((1 + interest) ** (lifetime_wind)))
    * wind_generator
    * cost_turbine
)

# Define Constraints

# (1)
for i in my_new.index:
    model += wind_used[i] + storage_discharged[i] == my_new["Demand(Wh)"][i]

# (2)
for i in my_new.index:
    model += (
        wind_used[i] + wind_curtailed[i] + wind_stored[i]
        == my_new["GeneratedEnergy (Wh)"][i] * wind_generator
    )

# (2.1)
storage_level[0] = 0

# (3)
for i in my_new.index:
    model += (
        storage_level[i + 1]
        == storage_level[i]
        + gamma * wind_stored[i]
        - (1 / gamma) * storage_discharged[i]
    )

# (4)
for i in my_new.index:
    model += storage_level[i] <= storageCap

# Solve model
model.solve()


with open("decision_variable_values.txt", "w") as f:
    for v in model.variables():
        f.write(f"{v.name} = {v.varValue}\n")

print("The capacity of the battery should be {} Wh.".format(storageCap.varValue))

print("Objective function value:", pulp.value(model.objective))

# Create a list of variable names and sort them numerically
var_names = sorted(
    [v.name for v in model.variables()],
    key=lambda x: (int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1, x),
)

# Create an empty dictionary to hold the values of the decision variables
var_dict = {}

# Loop through the decision variables and populate the dictionary
for v in model.variables():
    var_dict[v.name] = v.varValue

# Convert the dictionary to a pandas dataframe and sort the rows by variable name
df = pd.DataFrame.from_dict(var_dict, orient="index", columns=["Value"]).loc[var_names]


# %%
########### PART B #########

# initialize model
model = LpProblem(name="MinimizeCost", sense=LpMinimize)

# Define decision variables
wind_generator = LpVariable("wind_generator", lowBound=0, upBound=None, cat="Integer")

wind_used = LpVariable.dicts(
    "wind_used", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

wind_stored = LpVariable.dicts(
    "wind_stored", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

storage_discharged = LpVariable.dicts(
    "storage_discharged",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

wind_curtailed = LpVariable.dicts(
    "wind_curtailed",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

storage_level = LpVariable.dicts(
    "storage_level",
    [i for i in list(range(0, len(my_new.index) + 1))],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

interest = 0.5 / 100
cost_turbine = 2000000
gamma = 0.85
storage_capacity = 1500000
lifetime_wind = -52 * 25

# Define Objective Function

model += (
    interest / (1 - ((1 + interest) ** (lifetime_wind))) * cost_turbine * wind_generator
)

# Define Constraints

# (1)
for i in my_new.index:
    model += wind_used[i] + storage_discharged[i] == my_new["Demand(Wh)"][i]

# (2)
for i in my_new.index:
    model += (
        my_new["GeneratedEnergy (Wh)"][i] * wind_generator
        == wind_used[i] + wind_curtailed[i] + wind_stored[i]
    )

# (2.1)
storage_level[0] = 0

# (3)
for i in my_new.index:
    model += (
        storage_level[i + 1]
        == storage_level[i]
        + gamma * wind_stored[i]
        - (1 / gamma) * storage_discharged[i]
    )

# (4)
for i in my_new.index:
    model += storage_level[i] <= storage_capacity

# Solve model
model.solve()


with open("decision_variable_values.txt", "w") as f:
    for v in model.variables():
        f.write(f"{v.name} = {v.varValue}\n")

print("Buy {} of wind turbines.".format(wind_generator.varValue))
print("Objective function value = $", pulp.value(model.objective))

# Create a list of variable names and sort them numerically
var_names = sorted(
    [v.name for v in model.variables()],
    key=lambda x: (int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1, x),
)

# Create an empty dictionary to hold the values of the decision variables
var_dict = {}

# Loop through the decision variables and populate the dictionary
for v in model.variables():
    var_dict[v.name] = v.varValue

# Convert the dictionary to a pandas dataframe and sort the rows by variable name
df = pd.DataFrame.from_dict(var_dict, orient="index", columns=["Value"]).loc[var_names]


# %%
############################# PART B #############################
########### Calculate Capacity Factor #######

actual_output = my_new["GeneratedEnergy (Wh)"].sum()
potential_output = 1500000 * 168

capacity_factor = actual_output / potential_output

# %%

######## PART C ########
# initialize model
model = LpProblem(name="MinimizeCost", sense=LpMinimize)

# Define decision variables
storageCap = LpVariable("storageCap", lowBound=0, upBound=None, cat="Integer")

wind_used = LpVariable.dicts(
    "wind_used", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

wind_stored = LpVariable.dicts(
    "wind_stored", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

storage_discharged = LpVariable.dicts(
    "storage_discharged",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)


wind_curtailed = LpVariable.dicts(
    "wind_curtailed",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

storage_level = LpVariable.dicts(
    "storage_level",
    [i for i in list(range(0, len(my_new.index) + 1))],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

wind_generator = 18
cost_storage = 0.60
gamma = 0.85
interest = 0.5 / 100
lifetime_wind = -52 * 25
lifetime_storage = -52 * 20
cost_turbine = 2000000

# Define Objective Function

model += (
    interest / (1 - ((1 + interest) ** (lifetime_storage))) * cost_storage * storageCap
    + interest
    / (1 - ((1 + interest) ** (lifetime_wind)))
    * wind_generator
    * cost_turbine
)

# Define Constraints

# (1)
for i in my_new.index:
    model += wind_used[i] + storage_discharged[i] == my_new["Demand(Wh)"][i]

# (2)
for i in my_new.index:
    model += (
        wind_used[i] + wind_curtailed[i] + wind_stored[i]
        == my_new["GeneratedEnergy (Wh)"][i] * wind_generator
    )

# (2.1)
storage_level[0] = 0

# (3)
for i in my_new.index:
    model += (
        storage_level[i + 1]
        == storage_level[i]
        + gamma * wind_stored[i]
        - (1 / gamma) * storage_discharged[i]
    )

# (4)
for i in my_new.index:
    model += storage_level[i] <= storageCap

# Solve model
model.solve()


with open("decision_variable_values.txt", "w") as f:
    for v in model.variables():
        f.write(f"{v.name} = {v.varValue}\n")

print("The capacity of the battery should be {} Wh.".format(storageCap.varValue))

print("Objective function value:", pulp.value(model.objective))

# Create a list of variable names and sort them numerically
var_names = sorted(
    [v.name for v in model.variables()],
    key=lambda x: (int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1, x),
)

# Create an empty dictionary to hold the values of the decision variables
var_dict = {}

# Loop through the decision variables and populate the dictionary
for v in model.variables():
    var_dict[v.name] = v.varValue

# Convert the dictionary to a pandas dataframe and sort the rows by variable name
df = pd.DataFrame.from_dict(var_dict, orient="index", columns=["Value"]).loc[var_names]

# %%
######## PART D ########
### DATA PREPARATION ####
my_new["NewGeneratedEnergy (Wh)"] = my_new["GeneratedEnergy (Wh)"] * 4

# Assuming the 'GeneratedEnergy(Wh)' column is in my_new dataframe
my_new.plot(y="NewGeneratedEnergy (Wh)")

# Add title and labels
plt.title("Energy Generation of Double Sized Wind Turbine")
plt.xlabel("The Hour of the Week")
plt.ylabel("Energy Generated (Wh)")

# Set the x-axis tick labels
n = 10  # number of tick labels
tick_labels = [
    int(i)
    for i in pd.Series(range(len(my_new))).quantile(q=[i / (n - 1) for i in range(n)])
]  # compute tick labels
plt.xticks(tick_labels, [str(i + 1) for i in tick_labels])  # set tick labels

# Disable scientific notation on y-axis
plt.ticklabel_format(style="plain", axis="y")

# Remove legend
plt.legend().remove()

# Show the plot
plt.show()


# %%
######## PART D ########
# initialize model
model = LpProblem(name="MinimizeCost", sense=LpMinimize)

# Define decision variables
storageCap = LpVariable("storageCap", lowBound=0, upBound=None, cat="Integer")
wind_generator = LpVariable("wind_generator", lowBound=0, upBound=None, cat="Integer")

wind_used = LpVariable.dicts(
    "wind_used", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

wind_stored = LpVariable.dicts(
    "wind_stored", [i for i in my_new.index], lowBound=0, upBound=None, cat="Continuous"
)

storage_discharged = LpVariable.dicts(
    "storage_discharged",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)


wind_curtailed = LpVariable.dicts(
    "wind_curtailed",
    [i for i in my_new.index],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)

storage_level = LpVariable.dicts(
    "storage_level",
    [i for i in list(range(0, len(my_new.index) + 1))],
    lowBound=0,
    upBound=None,
    cat="Continuous",
)


cost_storage = 0.60
gamma = 0.85
interest = 0.5 / 100
lifetime_wind = -52 * 30
lifetime_storage = -52 * 20
cost_turbine = 2500000

# Define Objective Function

model += (
    interest / (1 - ((1 + interest) ** (lifetime_storage))) * cost_storage * storageCap
    + interest
    / (1 - ((1 + interest) ** (lifetime_wind)))
    * wind_generator
    * cost_turbine
)

# Define Constraints

# (1)
for i in my_new.index:
    model += wind_used[i] + storage_discharged[i] == my_new["Demand(Wh)"][i]

# (2)
for i in my_new.index:
    model += (
        wind_used[i] + wind_curtailed[i] + wind_stored[i]
        == my_new["NewGeneratedEnergy (Wh)"][i] * wind_generator
    )

# (2.1)
storage_level[0] = 0

# (3)
for i in my_new.index:
    model += (
        storage_level[i + 1]
        == storage_level[i]
        + gamma * wind_stored[i]
        - (1 / gamma) * storage_discharged[i]
    )

# (4)
for i in my_new.index:
    model += storage_level[i] <= storageCap

# Solve model
model.solve()


with open("decision_variable_values.txt", "w") as f:
    for v in model.variables():
        f.write(f"{v.name} = {v.varValue}\n")

print("The capacity of the battery should be {} Wh.".format(storageCap.varValue))
print("Number of generator should be {}.".format(wind_generator.varValue))
print("Objective function value:", pulp.value(model.objective))

# Create a list of variable names and sort them numerically
var_names = sorted(
    [v.name for v in model.variables()],
    key=lambda x: (int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1, x),
)

# Create an empty dictionary to hold the values of the decision variables
var_dict = {}

# Loop through the decision variables and populate the dictionary
for v in model.variables():
    var_dict[v.name] = v.varValue

# Convert the dictionary to a pandas dataframe and sort the rows by variable name
df = pd.DataFrame.from_dict(var_dict, orient="index", columns=["Value"]).loc[var_names]
