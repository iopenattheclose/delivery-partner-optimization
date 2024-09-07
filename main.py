from preprocess import pre_process,fetch_data
from constants import input_file_path,output_file_path
import numpy as np


def predict():
    print("Enter datapoint index and k-value:")
    #the test datapoint is taken from val dataset
    n = int(input())
    k = int(input())

    print("Reading data..........")
    df = fetch_data(input_file_path)
    print("Completed reading data")

    X_train,X_val,X_test, y_train,y_val,y_test = pre_process(df)

    pred,neighbors = knn(X_train,y_train, X_val[n],k)

    print(f'k nearest neighbors with the distance and class label :{neighbors}')

    print(f'The predicted class label: {pred}')

    print(f'Actual class: ',y_val[n])

def knn(X,Y,queryPoint,k):
    """Predict the class label for the query point"""
    # Euclidean Distance
    dist = np.sqrt(np.sum((queryPoint-X)**2,axis=1) )

    # Storing distance and Class labels together
    distances = [(dist[i],Y[i]) for i in range(len(dist))]    
    # sort the distances
    distances = sorted(distances)
    # Nearest/First K points
    distances = distances[:k]

    distances = np.array(distances)

    #returns unique classes and count of each class as two sep np arrays
    classes_counts = np.unique(distances[:,1],return_counts=True)

    #gives index of max value of count array
    index = classes_counts[1].argmax()

    #returns the class having max count
    pred = classes_counts[0][index]

    # print('Neighbors[(distance),(Label)]:',distances)
    return int(pred),distances
    


if __name__ == "__main__":
    predict()