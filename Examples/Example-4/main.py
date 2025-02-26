from bsw import BSW

# Instantiate a object type BSW
four_simul = BSW()
# Load um arquivo .csv and not uses the Kalman filter for this data (see docs/documentation)
four_simul.load_data('all-5-oil-with-api.csv', kalman_filtered=False)
# Fit the model to loaded data. The optimization of parameters in the model is performed
# without using restriction (see docs/documentation)
four_simul.fit(with_restriction=False)
# The method predict() does not use arguments, therefore  the prediction is performed with all data  
# This method shows the experimental data together confidence bands and 
# mean prediction of the model (see docs/documentation).
four_simul.predict()
# The method sensitivity_analysis_on_input_variables() executes a sensitivity analysis on variables
# using the global sensitivity analysis based on Sobol indices (see docs/documentation)
# the parameter passed to the method (exp_nsample) is an exponent of the sample space.
# The exponent can be increased if some sensitivity values are less than zero. The higher exp_nsample
# longer computing time. By default this method uses radial=False indicating a normal plotting of sensitivity
# indices: S1 (first-order), S2 (second-order) and ST (total) is desirable. If radial=True a radial plotting 
# is executed (see docs/documentation)
four_simul.sensitivity_analysis_on_input_variables(exp_nsample=12)
# The method sensitivity_analysis_on_parameters() executes a sensitivity analysis on variables
# using the global sensitivity analysis based on Sobol indices (see docs/documentation).
# Here, the parameter passed to the method (exp_nsample) is an exponent of the sample space
# The exponent can be increased if some sensitivity values are less than zero. The higher exp_nsample
# longer computing time
four_simul.sensitivity_analysis_on_parameters()





