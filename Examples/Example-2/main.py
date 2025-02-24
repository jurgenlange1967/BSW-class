from bsw import BSW

second_simul = BSW()
X_train, X_test = second_simul.load_data('p-66-otra-forma-input-2-without-atypical-values-with-all-values-Kfiltered-'
                                         'without-api.csv', kalman_filtered=False, cut=0.3)
second_simul.fit(with_restriction=False)
second_simul.Predict(X_test)
second_simul.plot_correlation_matrix_with_heatmap()
second_simul.show_data_filtered()
this_boolean_parameters = {"B1": True, "B2": True, "B4": True, "B5": False, "B6": False}
second_simul.error_ellipse(this_boolean_parameters, multiples_plot=True)
