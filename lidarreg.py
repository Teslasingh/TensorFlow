import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from keras.models import Sequential
from keras.layers import Dense
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator, ScalarFormatter



dataset = pd.read_excel("generated_data.xlsx")


#////////////////////////////////////////////////////////////////////////////////////////////no 3d color//////////////////////

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set labels for axes
ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_zlabel('Values')

# # Plot the data
# # for i in range(dataset.shape[0]):
# #     xs = range(dataset.shape[1])
# #     ys = [i] * dataset.shape[1]
# #     zs = dataset.iloc[i,:]
# #     ax.plot(xs, ys, zs)

for i in range(dataset.shape[0]):  # Select all rows
    xs = range(1,181)
    ys = [i] * 180
    zs = dataset.iloc[i,:180]  # Select first 180 columns
    ax.plot(xs, ys, zs)
    

# # Show the plot
plt.show()
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Create a new figure and subplot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Create arrays of X, Y, and Z values using your dataset
X = np.arange(1, 181)  # Assuming your data has 180 columns
Y = np.arange(0, dataset.shape[0])  # Index values for each row
X, Y = np.meshgrid(X, Y)
Z = dataset.iloc[:, :180].values  # Select first 180 columns of all rows

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis
ax.set_zlim(Z.min(), Z.max())  # Set limits based on data range
ax.zaxis.set_major_locator(LinearLocator(10))
formatter = ScalarFormatter()
formatter.set_powerlimits((-2, 2))
ax.zaxis.set_major_formatter(formatter)

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# Set labels for axes
ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_zlabel('Values')

plt.show()


#/////////////////////////////////////////////////////////////////////////////////////
df_subset = dataset.iloc[:, 180]  # Select only the 181st column
# plot the data
plt.plot(df_subset)
plt.show()

#////////////////////////////////////////////////////////////////////////////////////////
print(dataset.head(5))

print(dataset.shape)

obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))

# plt.figure(figsize=(12, 6))
# sns.heatmap(dataset.corr(),
# 			cmap = 'BrBG',
# 			fmt = '.2f',
# 			linewidths = 2,
# 			annot = True)
# plt.show()

# unique_values = []
# for col in object_cols:
#     unique_values.append(dataset[col].unique().size)
# plt.figure(figsize=(10,6))
# plt.title('No. Unique values of Categorical Features')
# plt.xticks(rotation=90)
# sns.barplot(x=object_cols,y=unique_values)
# plt.show()

# plt.figure(figsize=(18, 36))
# plt.title('Categorical Features: Distribution')
# plt.xticks(rotation=90)
# index = 1

# for col in object_cols:
# 	y = dataset[col].value_counts()
# 	plt.subplot(11, 4, index)
# 	plt.xticks(rotation=90)
# 	sns.barplot(x=list(y.index), y=y)
# 	index += 1
# plt.show()

#dataset.drop(['Id'],axis=1,inplace=True)

dataset[['FY', 'FZ']] = dataset[['FY', 'FZ']].fillna(dataset[['FY', 'FZ']].mean())

new_dataset = dataset.dropna()

#print(new_dataset.isnull().sum())


s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ',len(object_cols))

# OH_encoder = OneHotEncoder(sparse=False)
# OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
# OH_cols.index = new_dataset.index
# OH_cols.columns = OH_encoder.get_feature_names()
# df_final = new_dataset.drop(object_cols, axis=1)
# df_final = pd.concat([df_final, OH_cols], axis=1)
# print(df_final)
df_final = new_dataset

#df_final.to_excel('output.xlsx', index=False)     # save numerical dataset to xlsx file

# Splitting Dataset into Training and Testing

X = df_final.drop(['FY','FZ'], axis=1)
Y = df_final[['FY','FZ']]

# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)



# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set labels for axes
ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_zlabel('Values')

# Plot the data
# for i in range(dataset.shape[0]):
#     xs = range(dataset.shape[1])
#     ys = [i] * dataset.shape[1]
#     zs = dataset.iloc[i,:]
#     ax.plot(xs, ys, zs)
dataset_df = pd.DataFrame(X_train)

for i in range(500):
    xs = range(1, 180+1)
    ys = [i] * 180
    zs = dataset_df.iloc[0,:180]
    ax.plot(xs, ys, zs)
    

# Show the plot
plt.show()
# #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# model_SVR = svm.SVR()
# model_SVR.fit(X_train,Y_train)                                # Support vector machine 
# Y_pred = model_SVR.predict(X_valid)
# print(mean_absolute_percentage_error(Y_valid, Y_pred))


# model_RFR = RandomForestRegressor(n_estimators=10)
# model_RFR.fit(X_train, Y_train)
# Y_pred = model_RFR.predict(X_valid)                           #random forest regression 
# print(mean_absolute_percentage_error(Y_valid, Y_pred))

# model_LR = LinearRegression()
# model_LR.fit(X_train, Y_train)
# Y_pred = model_LR.predict(X_valid)                              #linear regression ,280000 ,185000, ,92000S  ,185000 ,175000, 266500  ,84500
# print(mean_absolute_percentage_error(Y_valid, Y_pred))

#  / / /  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
model = Sequential([
    Dense(124, input_shape=(180,), activation='relu'),
    Dense(124, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='linear')
])
print(model.summary())
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history=model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_valid, Y_valid))

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
print(model.evaluate(X_valid, Y_valid)[1])
model_path = r"C:\Users\FE871\Music\my_ann_model.h5"

# Save the model to the file path
model.save(model_path)

# Print a message to confirm that the model has been saved
print('Model saved to', model_path)

#input_data = (60,12012,5,1998,1998,0,1085,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0)
input_data = (2798,2799,2800,2801,2802,2803,2804,2805,2806,2807,2808,2809,2810,2811,2812,2813,2814,2815,2816,2817,2818,2819,2820,2821,2822,2823,2824,2825,2826,2827,2828,2829,2830,2831,2832,2833,2834,2835,2836,2837,2838,2839,2840,2841,2842,2843,2844,2845,2846,2847,2848,2849,2850,2851,2852,2853,2854,2855,2856,2857,2858,2859,2860,2861,2862,2863,2864,2865,2866,2867,2868,2869,2870,2871,2872,2873,2874,2875,2876,2877,2878,2879,2880,2881,2882,2883,2884,2885,2886,2887,2888,2889,2890,2891,2892,2893,2894,2895,2896,2897,2898,2899,2900,2901,2902,2903,2904,2905,2906,2907,2908,2909,2910,2911,2912,2913,2914,2915,2916,2917,2918,2919,2920,2921,2922,2923,2924,2925,2926,2927,2928,2929,2930,2931,2932,2933,2934,2935,2936,2937,2938,2939,2940,2941,2942,2943,2944,2945,2946,2947,2948,2949,2950,2951,2952,2953,2954,2955,2956,2957,2958,2959,2960,2961,2962,2963,2964,2965,2966,2967,2968,2969,2970,2971,2972,2973,2974,2975,2976,2977)
# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped.shape)

#scaler = StandardScaler()
input_data_reshaped = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_reshaped)
print(prediction)

