import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

regressor = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=5)
tree = DecisionTreeRegressor()
forest = RandomForestRegressor(n_estimators=100)
svm = SVR(kernel='rbf')


def load_data():

    df = pd.read_csv("table.csv")
    # Remove na values
    df = df.dropna()
    print(df)

    # histogram
    min_range = min(df['Age'])
    max_range = max(df['Age'])
    figure1 = plt.figure(figsize=(10, 7))
    plt.hist(df['Age'], bins=range(min_range - 2, max_range + 2, 1), edgecolor='blue')
    plt.xticks(range(min_range -2 , max_range + 2, 1))
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Histogram of Age')
    # plt.show()
    figure1.savefig('Images/histogram.png')
    plt.close()

    X = df['Age'].values
    Y = df['Join_month'].values

    # Scatter plot
    figure2 = plt.figure(figsize=(10, 7))
    plt.scatter(X, Y, color='blue')
    plt.xlabel('Age')
    plt.ylabel('Join_month')
    plt.title('Scatter plot of Age and Join_month')
    # plt.show()
    figure2.savefig('Images/month_join_scatter.png')
    plt.close()

    X = df['Age'].values
    Y = df['Join_year'].values

    # Scatter plot
    figure3 = plt.figure(figsize=(10, 7))
    plt.scatter(X, Y, color='blue')
    plt.xlabel('Age')
    plt.ylabel('Join_year')
    plt.title('Scatter plot of Age and Join_year')
    # plt.show()
    figure3.savefig('Images/year_join_scatter.png')
    plt.close()




    mse_score = {}





    # Linear regression
    X = df['Join_month'].values.reshape(-1, 1)
    Y = df['Age'].values.reshape(-1, 1)
    new_df = pd.DataFrame({'Join_month': X[:, 0], 'Age': Y[:, 0]})


    # regressor = LinearRegression()
    regressor.fit(X, Y)
    # Plot regression line
    Y_pred = regressor.predict(X)
    figure4 = plt.figure(figsize=(10, 7))
    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.xlabel('Join_month')
    plt.ylabel('Age')
    plt.title('Linear regression of Age and Join_month')
    # plt.show()
    figure4.savefig('Images/month_join_regression.png')
    plt.close()

    # Calculate MSE of linear regression
    mse_score['Linear Regression'] = mean_squared_error(Y, Y_pred)



    # KNN regression
    # knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X, Y)
    # Plot regression line
    Y_pred = knn.predict(X)
    figure5 = plt.figure(figsize=(10, 7))
    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.xlabel('Join_month')
    plt.ylabel('Age')
    plt.title('KNN regression of Age and Join_month')
    # plt.show()
    figure5.savefig('Images/month_join_knn.png')
    plt.close()

    # Calculate MSE of KNN regression
    mse_score['KNN Regression'] = mean_squared_error(Y, Y_pred)



    # Decision tree regression
    # tree = DecisionTreeRegressor()
    tree.fit(X, Y)
    # Plot regression line
    Y_pred = tree.predict(X)
    figure6 = plt.figure(figsize=(10, 7))
    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.xlabel('Join_month')
    plt.ylabel('Age')
    plt.title('Decision tree regression of Age and Join_month')
    # plt.show()
    figure6.savefig('Images/month_join_tree.png')
    plt.close()

    # Calculate MSE of decision tree regression
    mse_score['Decision Tree Regression'] = mean_squared_error(Y, Y_pred)


    # Random forest regression
    # forest = RandomForestRegressor(n_estimators=100)
    forest.fit(X, Y)
    # Plot regression line
    Y_pred = forest.predict(X)
    figure7 = plt.figure(figsize=(10, 7))
    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.xlabel('Join_month')
    plt.ylabel('Age')
    plt.title('Random forest regression of Age and Join_month')
    # plt.show()
    figure7.savefig('Images/month_join_forest.png')
    plt.close()

    # Calculate MSE of random forest regression
    mse_score['Random Forest Regression'] = mean_squared_error(Y, Y_pred)



    # Support vector regression
    # svm = SVR(kernel='rbf')
    svm.fit(X, Y)
    # Plot regression line
    Y_pred = svm.predict(X)
    figure8 = plt.figure(figsize=(10, 7))
    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.xlabel('Join_month')
    plt.ylabel('Age')
    plt.title('Support vector regression of Age and Join_month')
    # plt.show()
    figure8.savefig('Images/month_join_svm.png')
    plt.close()

    # Calculate MSE of support vector regression
    mse_score['Support Vector Regression'] = mean_squared_error(Y, Y_pred)




    print(mse_score)

    # Plot accuracy of all regression models
    figure9 = plt.figure(figsize=(10, 7))
    plt.bar(range(len(mse_score)), list(mse_score.values()), align='center')
    plt.xticks(range(len(mse_score)), list(mse_score.keys()))
    plt.xlabel('Regression models')
    plt.xticks(rotation=20)
    plt.ylabel('MSE')
    plt.title('MSE of all regression models')
    # plt.show()
    figure9.savefig('Images/mse.png')
    plt.close()


if(__name__ == '__main__'):
    print("Loading data...")
    load_data()
    print("Data loaded successfully\n")

    while(True):
        print("\n\nEnter a month to predict the age of the user : ", end="")
        month = int(input())

        if(month >= 1 and month <= 12):
            print("Predictions")
            print("Linear Regression : ", regressor.predict([[month]])[0][0])
            print("KNN Regression : ", knn.predict([[month]])[0])
            print("Decision Tree Regression : ", tree.predict([[month]])[0])
            print("Random Forest Regression : ", forest.predict([[month]])[0])
            print("Support Vector Regression : ", svm.predict([[month]])[0])
        else:
            print("Month should be between 1 and 12")

        print("Do you want to continue? (y/n)")
        choice = input()
        if(choice == 'n'):
            os._exit(0)
        else:
            continue
