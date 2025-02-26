from bsw import BSW

# Instantiate a object type BSW
first_simul = BSW()
# Load um arquivo .csv and uses a Kalman filter for this data (see docs/documentation)
# Note that this method does not return nothing because the cut value is by default = 0
# indicating that there is no test data, thereby the train data is equal to test data
first_simul.load_data('p-66-otra-forma-input-2-without-atypical-values-with-all-values.csv')
# fit the model to the loaded data. The optimization of parameters in the model is performed
# without using restriction (see docs/documentation)
first_simul.fit(with_restriction=False)
# The method predict() shows the simulated and experimental data together confidence bands and 
# mean prediction (see docs/documentation).
first_simul.predict()
# Shows a correlation matrix between parameters (see docs/documentation)
first_simul.plot_correlation_matrix_with_heatmap()
# The method shows the all filtered data using the Kalman filter (see docs/documentation)
first_simul.show_data_filtered()
# The method is alternative one to the show_data_filtered. It shows the experimental data
# without the filtered data (see docs/documentation
first_simul.show_only_experimental_data()
