from bsw import BSW

first_simul = BSW()
first_simul.load_data('p-66-otra-forma-input-2-without-atypical-values-with-all-values.csv')
# In the case of bad fitting by the model - use then with_restriction=False or modify the initial guesses
first_simul.fit(with_restriction=False)
first_simul.Predict()
first_simul.plot_correlation_matrix_with_heatmap()
first_simul.show_data_filtered()
first_simul.show_only_experimental_data()
