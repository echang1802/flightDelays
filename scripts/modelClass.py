
import logging
import pandas as pd
from numpy import arange
from random import seed, sample, choices, uniform, randint

def addVariables(dataSet):
    dataSet["DepartureAtLabourHour"] = ((dataSet["DepartureTime"] >= 600) & (dataSet["DepartureTime"] < 2100)).astype(int)
    dataSet["DistanceGreater1000"] = (dataSet["Distance"] > 1000).astype(int)
    
    return 
        

def oneHotEncoding(dataSet, column, values, newColumn = None):
    for value in values:
        colName = (column if newColumn is None else newColumn) + str(value)
        dataSet[colName] = 0
        dataSet.loc[dataSet[column] == value, colName] = 1
        
    dataSet.drop(columns = column, inplace  = True)
    
    return 

def scaleDataSet(dataSet):
    dataSet["Month"] = (dataSet["Month"] - 1)/ 11
    
    minVar = min(dataSet["DepartureTime"])
    dataSet["DepartureTime"] = (dataSet["DepartureTime"] - minVar) / (max(dataSet["DepartureTime"]) - minVar)
    
    minVar = min(dataSet["SchedElapsedTime"])
    dataSet["SchedElapsedTime"] = (dataSet["SchedElapsedTime"] - minVar) / (max(dataSet["SchedElapsedTime"]) - minVar)
    
    minVar = min(dataSet["Distance"])
    dataSet["Distance"] = (dataSet["Distance"] - minVar) / (max(dataSet["Distance"]) - minVar)
    
    return   

def balanceDataSet(dataSet, balanceType, labelProportion):
    if balanceType == "undersampling":
        labelProportion = round(sum(dataSet["Canceled"] == 1) * (1 - labelProportion) / labelProportion)
        undersampling = sample(set(dataSet.index[dataSet["Canceled"]== 0]), labelProportion )
        dataSet = dataSet.loc[dataSet["Canceled"] == 1].append(dataSet.loc[undersampling])
        dataSet.reset_index(drop = True, inplace = True)
    elif balanceType == "oversampling":
        labelProportion = round((sum(dataSet["Canceled"] == 0) * labelProportion) / (1 - labelProportion))
        oversampling = choices(dataSet.index[dataSet["Canceled"] == 1], k = labelProportion)
        dataSet = dataSet.loc[dataSet["Canceled"] == 0].append(dataSet.loc[oversampling])
        dataSet.reset_index(drop = True, inplace = True)

def getMetrics(preds, labels):
    tp = sum((preds == 1) & (labels == 1))
    fp = sum((preds == 1) & (labels == 0))
    tn = sum((preds == 0) & (labels == 0))
    fn = sum((preds == 0) & (labels == 1))
    
    
    return {
            'precision' : tp / (tp + fp),
            'recall' : tp / (tp + fn),
            'accuracy' : (tp + tn) / (tp + fp + tn + fn),
            'f-score' : tp / (tp + (fp + fn)/2)
        }

class canceledFlightsModel:
    
    def __init__(self):
        self._balanced = False
        self._scale = False
        self._data =  pd.read_csv("data/FlightDelays_Data_3.0.csv")
        
    def cleanInconsistentData(self):
        N = self._data.shape[0]
        self._data = self._data.loc[self._data["Canceled"].isin([0,1])]
        logging.info("{} rows deleted by inconsistent label".format(N - self._data.shape[0]))
        N = self._data.shape[0]
        self._data = self._data.loc[self._data["Month"].isin(range(1,13))]
        logging.info("{} rows deleted by inconsistent month".format(N - self._data.shape[0]))
        N = self._data.shape[0]
        self._data = self._data.loc[(self._data["DepartureTime"] >= 0) & self._data["DepartureTime"] < 2400]
        logging.info("{} rows deleted by inconsistent departure time".format(N - self._data.shape[0]))
        N = self._data.shape[0]
        self._data = self._data.loc[self._data["SchedElapsedTime"] > 0]
        logging.info("{} rows deleted by inconsistent scheduled elapsed time".format(N - self._data.shape[0]))
        N = self._data.shape[0]
        self._data = self._data.loc[self._data["Distance"] > 0]
        logging.info("{} rows deleted by inconsistent distance".format(N - self._data.shape[0]))
                
    def trainTestSplit(self, validPrc = 0.2, trainPrc = 0.7, setSeed = 1):
        # ArrDealay and DepDelay will be droped because this information is not 
        # accequible before known the flight is not canceled
        seed(setSeed)
        validationIndex = sample(set(self._data.index),round(validPrc * self._data.shape[0]))
        self._validationSet = self._data.loc[validationIndex].drop(columns = ["ArrDelay","DepDelay"])
        dataSet = self._data.loc[[not x in validationIndex for x in self._data.index]].drop(columns = ["ArrDelay","DepDelay"])
        trainIndex = sample(set(dataSet.index),round(trainPrc * dataSet.shape[0]))
        self._trainSet = dataSet.loc[trainIndex]
        self._testSet = dataSet.loc[[not x in trainIndex for x in dataSet.index]]
        logging.info("{} rows in train - {} rows in test - {} rows in validation".format(self._trainSet.shape[0],self._testSet.shape[0],self._validationSet.shape[0]))

    def addVariables(self):
        addVariables(self._trainSet)
        addVariables(self._testSet)
        addVariables(self._validationSet)
        logging.info("Variables added")
        
    def convertData(self, scale = True):
        # Apply One Hot encoding to UniqueCarriers
        oneHotEncoding(self._trainSet, "UniqueCarrier", self._data["UniqueCarrier"].unique(), newColumn = "Carrier")
        oneHotEncoding(self._testSet, "UniqueCarrier", self._data["UniqueCarrier"].unique(), newColumn = "Carrier")
        oneHotEncoding(self._validationSet, "UniqueCarrier", self._data["UniqueCarrier"].unique(), newColumn = "Carrier")
            
        self._scale = scale
        if scale:
            scaleDataSet(self._trainSet)
            scaleDataSet(self._testSet)
            scaleDataSet(self._validationSet)
        logging.info("OneHotEncoding aplicated" + (" and variables scaled" if scale else ""))
            
    def balanceClass(self, balanceType, labelProportion):
        self._balanced = True
        self._balanceType = balanceType
        self._labelProportion = labelProportion
        balanceDataSet(self._trainSet, balanceType, labelProportion)
        logging.info("Data balanced with {} methodology".format(balanceType))
            
    def trainModel(self, model = "logit", parameters = {}, train = "train"): 
        trainSet  = self._trainSet
        if train == "all":
            if self._balanced:
                balanceDataSet(self._testSet, self._balanceType, self._labelProportion)
            trainSet  = trainSet .append(self._testSet)
        if model == "logit":
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression()
        elif model == "randomForest":
            from sklearn.ensemble import RandomForestClassifier
            self._model = RandomForestClassifier(n_estimators = parameters["n_estimators"] if "n_estimators" in parameters.keys() else 100,
                                                 max_depth = parameters["max_depth"] if "max_depth" in parameters.keys() else None,
                                                 min_samples_split = parameters["min_samples_split"] if "min_samples_split" in parameters.keys() else 2,
                                                 max_samples = parameters["max_samples"] if "max_samples" in parameters.keys() else None,
                                                 max_features = parameters["max_features"] if "max_features" in parameters.keys() else "auto")
        
        self._model.fit(self._trainSet.drop(columns = ["Canceled"]), self._trainSet["Canceled"])
        logging.info("Trained " + model + (" model" if train == "train" else " as final model"))
        
    def evaluate(self, threshold = 0.5, setToEvaluate = "test"):
        if setToEvaluate == "test":
            dateSet = self._testSet
        elif setToEvaluate == "validation":
            dateSet = self._validationSet
        preds = self._model.predict_proba(dateSet.drop(columns = ["Canceled"]))
        preds = pd.Series([1 if x[1] >= threshold else 0 for x in preds], index = dateSet.index)
        try:
            metrics = getMetrics(preds, dateSet["Canceled"])
        except ZeroDivisionError:
            logging.warning("No metrics avalaible for model")
            return "No metrics avalaible"
            
        logging.info("model metrics: ")
        logging.info(metrics)
        
        return metrics
    
    def createGrid(self, N, parameters):
        self._grid = pd.DataFrame(index = range(N))
        for param in parameters.keys():
            if parameters[param][0] == 0: 
                self._grid[param] = [uniform(parameters[param][1],parameters[param][2]) for x in range(N)]
            else:
                self._grid[param] = [randint(parameters[param][1],parameters[param][2]) for x in range(N)]
        self._grid["precision"] = 0
        self._grid["recall"] = 0
        self._grid["accuracy"] = 0
        self._grid["f-score"] = 0
        logging.info("{} size grid created".format(N))
        
    def optimizeModel(self, metric, threshold = 0.5):
        for ind, params in self._grid.iterrows():
            logging.info("training model {}".format(ind))
            self.trainModel("randomForest", parameters = {
                "n_estimators" : int(params.n_estimators),
                "max_depth" : int(params.max_depth),
                "min_samples_split" : params.min_samples_split,
                "max_samples" : params.max_samples,
                "max_features" : params.max_features
            })
        
            metrics = self.evaluate(threshold, "test")
            if type(metrics) == str:
                continue
            self._grid.at[ind, "precision"] = metrics["precision"]
            self._grid.at[ind, "recall"] = metrics["recall"]
            self._grid.at[ind, "accuracy"] = metrics["accuracy"]
            self._grid.at[ind, "f-score"] = metrics["f-score"]
        
        self._grid.sort_values(by = metric, ascending = False, inplace = True)
        self._grid.reset_index(drop = True, inplace = True)
        
        self.trainModel("randomForest", parameters = {
            "n_estimators" : int(self._grid.at[0,"n_estimators"]),
            "max_depth" : int(self._grid.at[0,"max_depth"]),
            "min_samples_split" : self._grid.at[0,"min_samples_split"],
            "max_samples" : self._grid.at[0,"max_samples"],
            "max_features" : self._grid.at[0,"max_features"]
        }, train = "all")
        
        metrics = self.evaluate(threshold, setToEvaluate = "validation")
        logging.info("Final model metrics:")
        logging.info(metrics)
        logging.info(self._grid.head(1))
        
        return metrics

    def predict(self, newData, labels = None, threshold = 0.5):
        oneHotEncoding(newData, "UniqueCarrier", self._data["UniqueCarrier"].unique(), "Carrier")
        if self._scale:
            scaleDataSet(newData)
        if "DistanceGreater1000" in self._trainSet.columns:
            addVariables(newData)            
            
        preds = self._model.predict(newData)
        
        if labels is None:
            return preds
        
        predsVals = pd.Series([1 if x[1] > threshold else 0 for x in preds], index = labels.index)
        try:
            metrics = getMetrics(predsVals,labels)
        except ZeroDivisionError:
            metrics = "No metrics avalaible"
        return preds, metrics
    
        