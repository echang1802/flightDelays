
import pickle
import pandas as pd
from sys import argv

if __name__ == "__main__":
    
    with open("model/final_randomForest", "rb") as file:
        flightModel = pickle.load(file)
        
    filenameInput = argv[1]
    filenameOutput = argv[1]
    
    newData = pd.read_csv(filenameInput)
    
    newData["Canceled"] = flightModel.predict(newData, 0.1667)

    newData.to_csv(filenameOutput)
