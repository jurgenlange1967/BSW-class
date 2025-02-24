from bsw import BSW

third_simul = BSW()
third_simul.load_data('all-5-oil.csv', kalman_filtered=False)
third_simul.fit()
third_simul.Predict(showErr=False)

