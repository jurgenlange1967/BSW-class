from bsw import BSW

# Instantiate a object type BSW
first_simul = BSW()
# Load um arquivo .csv and uses a Kalman filter for this data (see documentation)
# Note that this method does not return nothing and the cut value is by default = 0
# indicating that ther is no test data, thereby the train data is equal to test data
first_simul.load_data('p-66-otra-forma-input-2-without-atypical-values-with-all-values.csv')
# fit the model to the data loaded. The optimization of parameters in the model is performed
# without using restriction 
first_simul.fit(with_restriction=False)
# The method predict shows the simulated and experimental data together confidence bands and 
# mean prediction.
first_simul.Predict()
# Shows 
first_simul.plot_correlation_matrix_with_heatmap()
first_simul.show_data_filtered()
first_simul.show_only_experimental_data()
