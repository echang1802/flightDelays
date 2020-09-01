
import pickle
import logging
from datetime import datetime
from scripts.modelClass import canceledFlightsModel

if __name__ == '__main__':
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s',
                            filename = 'logs/{}.log'.format(datetime.now().strftime("%Y%m%d_%H%M%S")), 
                            level=logging.INFO)
    
    flightModel = canceledFlightsModel()
    
    flightModel.cleanInconsistentData()
    
    flightModel.trainTestSplit()
    
    flightModel.addVariables()
    
    flightModel.convertData(scale = True)
    
    #flightModel.balanceClass(balanceType = "undersampling", 
    #                         labelProportion = 0.4)
    
    # Benchmark
    """
    flightModel.trainModel(model = "logit", train = "all")
    
    flightModel.evaluate(threshold = 0.1667, setToEvaluate = "validation")
    """
    
    # Random Forest
    
    flightModel.createGrid(50, {
            "n_estimators" : [1, 1000, 10000],
            "max_depth" : [1, 15, 20],
            "min_samples_split" : [0, 0.001, 0.05],
            "max_samples" : [0, 0.8, 1],
            "max_features" : [0, 0.8, 1]
        })
    
    flightModel.optimizeModel(metric = "f-score",
                              threshold = 0.1667)
      
    with open("models/final_randomForest", "wb") as file:
        pickle.dump(flightModel,file)
        logging.info("Model Saved")

    

    

