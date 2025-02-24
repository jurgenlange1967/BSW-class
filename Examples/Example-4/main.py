from bsw import BSW

four_simul = BSW()
four_simul.load_data('all-5-oil-with-api.csv', kalman_filtered=False)
four_simul.fit(with_restriction=False)
four_simul.Predict()
four_simul.sensitivity_analysis_on_input_variables(exp_nsample=12)
four_simul.sensitivity_analysis_on_parameters()
