from bsw import BSW

first_simul = BSW()
first_simul.load_data('p-66-otra-forma-input-2-without-atypical-values-with-all-values.csv')
# In the case of bad fitting by the model - use then with_restriction=False or modify the initial guesses
first_simul.fit(with_restriction=False)
first_simul.Predict()
first_simul.plot_correlation_matrix_with_heatmap()
first_simul.show_data_filtered()
first_simul.show_only_experimental_data()


second_simul = BSW()
X_train, X_test = second_simul.load_data('p-66-otra-forma-input-2-without-atypical-values-with-all-values-Kfiltered-'
                                         'without-api.csv', kalman_filtered=False, cut=0.3)
second_simul.fit(with_restriction=False)
second_simul.Predict(X_test)
second_simul.plot_correlation_matrix_with_heatmap()
second_simul.show_data_filtered()
this_boolean_parameters = {"B1": True, "B2": True, "B4": True, "B5": False, "B6": False}
second_simul.error_ellipse(this_boolean_parameters, multiples_plot=True)


third_simul = BSW()
third_simul.load_data('all-5-oil.csv', kalman_filtered=False)
third_simul.fit()
third_simul.Predict(showErr=False)


four_simul = BSW()
four_simul.load_data('all-5-oil-with-api.csv', kalman_filtered=False)
four_simul.fit(with_restriction=False)
four_simul.Predict()
four_simul.sensitivity_analysis_on_input_variables(exp_nsample=12)
four_simul.sensitivity_analysis_on_parameters()






