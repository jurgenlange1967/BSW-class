from bsw import BSW

# Instantiate a object type BSW
third_simul = BSW()
# Load um arquivo .csv and not uses the Kalman filter for this data (see docs/documentation)
third_simul.load_data('all-5-oil.csv', kalman_filtered=False)
# Fit the model to the loaded data. The optimization of parameters in the model is performed
# with constraints (see docs/documentation)
third_simul.fit()
# The method predict() uses all load data to make prediction (test set = training set)   
# This method does not shows confidence bands and mean prediction (see docs/documentation).
third_simul.predict(showErr=False)
