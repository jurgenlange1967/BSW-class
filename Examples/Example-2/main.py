from bsw import BSW

# Instantiate a object type BSW
second_simul = BSW()
# Load um arquivo .csv and not uses the Kalman filter for this data (see docs/documentation)
# Use cut=0.3 indicating that the data is broken down into 30% for test and 70% for training sets
# This method return the sets in the variables X_train and X_test
X_train, X_test = second_simul.load_data('p-66-otra-forma-input-2-without-atypical-values-with-all-values-Kfiltered-'
                                         'without-api.csv', kalman_filtered=False, cut=0.3)
# Fit the model to the data loaded. The optimization of parameters in the model is performed
# without using restriction (see docs/documentation)
second_simul.fit(with_restriction=False)
# The method predict() use the variable X_test as argument to predict BSW with the test sets   
# This method shows the simulated and experimental data together confidence bands and 
# mean prediction (see docs/documentation).
second_simul.predict(X_test)
# Shows a correlation matrix between parameters (see docs/documentation)
second_simul.plot_correlation_matrix_with_heatmap()
# The method shows the all filtered data using the Kalman filter (see docs/documentation)
second_simul.show_data_filtered()
# The method error_ellipses() show error ellipses for the combination of parameters selected in
# this_boolean_parameters  
this_boolean_parameters = {"B1": True, "B2": True, "B4": True, "B5": False, "B6": False}
# In the case B1 is true, a plotting will be shown with: B1 and B2, B1 and B4, B1 and B5, B1 and B6 
# (if there exists API in data), B1 and C1
second_simul.error_ellipse(this_boolean_parameters, multiples_plot=True)











