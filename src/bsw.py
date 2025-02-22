import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from decimal import Decimal
import seaborn as sns
from scipy.stats import chi2
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
import math
import itertools
import warnings
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")


class BSW:
    # class variable
    _api = True
    fig = np.empty(300, dtype=object)
    ax = np.empty(300, dtype=object)
    y_mean_pred = []
    y_max_conf = []
    y_min_conf = []
    counter_fig = 0  # Counter of plotted figure

    # constructor
    def __init__(self):
        self.file_name = ""
        self._data_train = []
        self._data_test = []
        self._sac_ratio_4 = []
        self._data = []
        self._p_opt = []
        self._p_conv = []
        self.df_filtered = pd.DataFrame()
        self.err_parameter = []
        self.data1 = pd.DataFrame()
        self.kalman_filtered = False

    def is_significant(self, value, confidence_interval, threshold="conf"):
        if threshold == "conf":
            return value - abs(confidence_interval) > 0
        else:
            return value - abs(float(threshold)) > 0

    # --Print sobol sensitivity with a radial scheme--
    # https://waterprogramming.wordpress.com/2019/08/27/a-python-implementation-of-grouped-radial-convergence-plots-to-
    # visualize-sobol-sensitivity-analysis-results/
    # Example Radial Convergence Plot for the Lake Problem reliability objective. Each of the points on the plot
    # represents a sampled uncertain parameter in the model. The size of the filled circle represents the first order
    # Sobol Sensitivity Index, the size of the open circle represents the total order Sobol Sensitivity Index and the
    # thickness of lines between points represents the second order Sobol Sensitivity Index
    def grouped_radial(self, SAresults, parameters, radSc=2.0, scaling=1, widthSc=0.5, STthick=1, varNameMult=1.3,
                       colors=None, groups=None, gpNameMult=1.5, threshold="conf"):
        # Derived from https://github.com/calvinwhealton/SensitivityAnalysisPlots
        sns.set_style('whitegrid', {'axes_linewidth': 0, 'axes.edgecolor': 'white'})
        fig, ax = plt.subplots(1, 1)
        color_map = {}

        # initialize parameters and colors
        if groups is None:

            if colors is None:
                colors = ["k"]

            for i, parameter in enumerate(parameters):
                color_map[parameter] = colors[i % len(colors)]
        else:
            if colors is None:
                colors = sns.color_palette("deep", max(3, len(groups)))

            for i, key in enumerate(groups.keys()):
                # parameters.extend(groups[key])

                for parameter in groups[key]:
                    color_map[parameter] = colors[i % len(colors)]

        n = len(parameters)
        angles = radSc * math.pi * np.arange(0, n) / n
        x = radSc * np.cos(angles)
        y = radSc * np.sin(angles)

        # plot second-order indices
        for i, j in itertools.combinations(range(n), 2):
            # key1 = parameters[i]
            # key2 = parameters[j]

            if self.is_significant(SAresults["S2"][i][j], SAresults["S2_conf"][i][j], threshold):
                angle = math.atan((y[j] - y[i]) / (x[j] - x[i]))

                if y[j] - y[i] < 0:
                    angle += math.pi

                line_hw = scaling * (max(0, SAresults["S2"][i][j]) ** widthSc) / 2

                coords = np.empty((4, 2))
                coords[0, 0] = x[i] - line_hw * math.sin(angle)
                coords[1, 0] = x[i] + line_hw * math.sin(angle)
                coords[2, 0] = x[j] + line_hw * math.sin(angle)
                coords[3, 0] = x[j] - line_hw * math.sin(angle)
                coords[0, 1] = y[i] + line_hw * math.cos(angle)
                coords[1, 1] = y[i] - line_hw * math.cos(angle)
                coords[2, 1] = y[j] - line_hw * math.cos(angle)
                coords[3, 1] = y[j] + line_hw * math.cos(angle)

                ax.add_artist(plt.Polygon(coords, color="0.75"))

        # plot total order indices
        for i, key in enumerate(parameters):
            if self.is_significant(SAresults["ST"][i], SAresults["ST_conf"][i], threshold):
                ax.add_artist(plt.Circle((x[i], y[i]), scaling * (SAresults["ST"][i] ** widthSc) / 2, color='w'))
                ax.add_artist(
                    plt.Circle((x[i], y[i]), scaling * (SAresults["ST"][i] ** widthSc) / 2, lw=STthick, color='0.4',
                               fill=False))

        # plot first-order indices
        for i, key in enumerate(parameters):
            if self.is_significant(SAresults["S1"][i], SAresults["S1_conf"][i], threshold):
                ax.add_artist(plt.Circle((x[i], y[i]), scaling * (SAresults["S1"][i] ** widthSc) / 2, color='0.4'))

        # add labels
        for i, key in enumerate(parameters):
            ax.text(varNameMult * x[i], varNameMult * y[i], key, ha='center', va='center',
                    rotation=angles[i] * 360 / (2 * math.pi) - 90,
                    color=color_map[key])

        if groups is not None:
            for i, group in enumerate(groups.keys()):
                print(group)
                group_angle = np.mean([angles[j] for j in range(n) if parameters[j] in groups[group]])

                ax.text(gpNameMult * radSc * math.cos(group_angle), gpNameMult * radSc * math.sin(group_angle), group,
                        ha='center', va='center',
                        rotation=group_angle * 360 / (2 * math.pi) - 90,
                        color=colors[i % len(colors)])

        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('equal')
        plt.axis([-2 * radSc, 2 * radSc, -2 * radSc, 2 * radSc])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

        return fig

    # filtro de kalman implemented by Jurgen Lange - similar results to Pykalman
    def kalmanfilter(self, y, init_state_mean, Q=1e-5, R=0.1 ** 2):
        # initial parameters
        z = y.values
        n_iter = len(z)
        sz = (n_iter,)  # size of array
        # allocate space for arrays
        x_hat = np.zeros(sz)  # a posteriori estimate of x
        P = np.zeros(sz)  # a posteriori error estimate
        x_hat_minus = np.zeros(sz)  # a priori estimate of x
        p_minus = np.zeros(sz)  # a priori error estimate
        K = np.zeros(sz)  # gain or blending factor

        # initial guesses
        x_hat[0] = init_state_mean
        x_hat[0] = np.squeeze(init_state_mean)
        P[0] = 1.0
        for k in range(1, n_iter):
            # time update
            x_hat_minus[k] = x_hat[k - 1]
            p_minus[k] = P[k - 1] + Q
            # measurement update
            K[k] = p_minus[k] / (p_minus[k] + R)
            x_hat[k] = x_hat_minus[k] + K[k] * (z[k] - x_hat_minus[k])
            P[k] = (1 - K[k]) * p_minus[k]
        return x_hat

    # convert input Kp from mm2/s to cm2/s, Ee from V/mm to kV/cm
    @staticmethod
    def divide_by_100(x):
        return x / 100

    # Validate if filename has .csv extension
    @staticmethod
    def validator_of_namefile(value):
        ext = value[value.index('.') + 1:len(value)]
        if ext == 'csv':
            return True
        else:
            return False

    # compute X2 statistic from simulation
    def chi2_test(self, y_exp, y_theo, K):
        ss_res = pow((y_exp - y_theo), 2) / y_theo
        csq = 0
        for i in range(len(ss_res)):
            csq += ss_res[i]
        n_dof = len(y_exp) - K  # number of data points - number of parameters in the model regression
        P = chi2.isf(q=0.05, df=n_dof)
        return csq, P

    # (Modelo 3) without API - 5 parameters
    def BSWo_4(self, data, a1, a2, a4, a5, c):
        sac_ratio = a1 * data['T'] - a2 * np.power(data['Ee'], 2) - a4 * data['Ql'] - a5 * data['Kp']
        self._sac_ratio_4 = sac_ratio
        return (c * np.exp(np.sqrt(sac_ratio))) / (sac_ratio * (1 + np.sqrt(1 / sac_ratio)))

    # (Model 3) with API - 6 parameters
    def BSWo_6(self, data, a1, a2, a4, a5, c, a6):
        sac_ratio = a1 * data['T'] - a2 * np.power(data['Ee'], 2) - a4 * data['Ql'] - a5 * data['Kp']
        self._sac_ratio_4 = sac_ratio
        return ((c * np.exp(np.sqrt(sac_ratio))) / (sac_ratio * (1 + np.sqrt(1 / sac_ratio)))) + a6 * data['API']

    # function to define if there is API field in the loaded data
    def api_boolean(self, data):
        column_length = len(data.columns.tolist())
        # select api considering the number of columns in dataframe
        if column_length == 7:
            self._api = True
            col_names = ['BS&Wo', 'Ee', 'T', 'Kp', 'Ql', 'NoExp', 'API']
        else:
            self._api = False
            col_names = ['BS&Wo', 'Ee', 'T', 'Kp', 'Ql', 'NoExp']
        return col_names

    @staticmethod
    def file_name_without_extension(name_file):
        file_name_without_ext = name_file[0: name_file.index('.')]
        return file_name_without_ext

    # Function to load data after constructor calling
    def load_data(self, value, cut=0.0, kalman_filtered=True, process_cov=1e-5, noise_cov=0.01):
        # T in (°C), KP in (cm2/s), Ql (L/h), Ee (KV/cm) units - in which the program work
        # Data is entered in units T in (°C), KP in (mm2/s), Ql (L/s), Ee (V/mm) units
        # The input data is assumed to have Ql values instead TRP, if TRP values are entered in the Ql field,
        # please put is_entered_trp boolean to True to convert it to Ql values
        # The process_cov and noise_cov are considered for all fields in the Kalman filter
        # smaller process_cov and higher noise_cov values result in smoother curves
        try:
            if self.validator_of_namefile(value):
                self.file_name = value

                if kalman_filtered:
                    try:
                        self.data1 = pd.read_csv(self.file_name, header=None)
                        # Filtering BSW data with kalman filter
                        # --------------------------------------------------
                        for column in range(len(self.data1.columns)):
                            y = self.data1[self.data1.columns[column]]
                            if column != 5:
                                self.df_filtered[self.data1.columns[column]] = self.kalmanfilter(y, y[0], process_cov,
                                                                                                 noise_cov)
                        # fill the column NoExp (or column 5) with index starting from 1
                        self.df_filtered.insert(5, 5, self.data1[5])

                        # Saving the filtered data to file .csv
                        File_Name_temp = self.file_name_without_extension(self.file_name)
                        File_Name_temp = File_Name_temp + '_filtered.csv'
                        self.df_filtered.to_csv(File_Name_temp, header=False, index=False)
                        # populate the dataframe with a new data from filtered file
                        data = pd.read_csv(File_Name_temp, header=None)

                        # update the name of file having the filtered data
                        self.file_name = File_Name_temp
                        data.columns = self.api_boolean(data)
                        # ---------------------------------------------------
                        # boolean indicating that the data was filtered
                        self.kalman_filtered = True
                        # Create data1 formatted with column names
                        self.data1.columns = self.api_boolean(self.data1)

                    except:
                        print("Error - Failed to execute data filtering")
                else:
                    data = pd.read_csv(self.file_name, header=None)
                    data.columns = self.api_boolean(data)

                self._data = data
                self._data['NoExp'] = self._data['NoExp'].astype(int)
                # Here is divided by 100 the values of Kp e Ee for a higher stability in the calculations
                self._data['Kp'] = self._data['Kp'].apply(self.divide_by_100)
                self._data['Ee'] = self._data['Ee'].apply(self.divide_by_100)
                self._data_train, self._data_test = np.split(self._data, [int((1 - cut) * len(self._data))])
                self._data_test = self._data_test.reset_index(drop=True)
                "Now the column No. Exp in self._data_test should be re-indexed"
                for i in self._data_test.index:
                    self._data_test.at[i, 'NoExp'] = i + 1
            else:
                print('Please enter a csv file')

        except FileNotFoundError:
            print("Error: The CSV file could not be found.")
        except csv.Error as e:
            print(f"Error: {e}")
        return self._data_train, self._data_test

    # save data computed
    # This method is important because the optimized parameters are scaled by 1e-4 for Ee e 1e-2 for Kp
    # In fact the optimized parameters correspond to this saved data
    def save_computed_data(self, filename):
        self._data.to_csv(filename, sep=",", index=False, header=False, float_format='%.3f')

    # fitting the model 3 to data
    def fit(self, p1=0.5, p2=0.4, p3=-0.4, p4=-0.2, p5=0.5, p6=-0.5, with_restriction=True, with_method='trf'):
        fitting_methods = ['trf', 'dogbox']
        if with_method not in fitting_methods:
            print('The method used to fit the data is not suitable... Please use only trf or doxbog in the '
                  'with_method parameter')
            exit()

        if not self._api:
            # model 3 without api - call BSWo_4
            try:

                if with_restriction:
                    bounds = ((-np.inf, 0, -np.inf, -np.inf, 0), (np.inf, np.inf, 0, 0, 1))
                else:
                    bounds = ((-np.inf, -np.inf, -np.inf, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf,
                                                                        1))
                self._p_opt, self._p_conv = curve_fit(self.BSWo_4, self._data_train, self._data_train['BS&Wo'],
                                                      p0=[p1, p2, p3, p4, p5],
                                                      bounds=bounds, method=with_method)
                print("Fitting was successful!")
                # Save to the parameter file
                self.print_parameter_by_oil('Model - 3 (without API) -- applied to ' +
                                            self.file_name_without_extension(self.file_name), self._data_train,
                                            self._p_opt, self._p_conv, 4)
                self.print_covariance_by_oil('Model - 3 -- applied to ' +
                                             self.file_name_without_extension(self.file_name), self._p_conv, 4)
                self.print_data_BSW_estimated_by_oil('Model - 3 (without API) -- applied to ' +
                                                     self.file_name_without_extension(self.file_name),
                                                     self._data_train['BS&Wo'],
                                                     self.BSWo_4(self._data_train, *self._p_opt), self._sac_ratio_4, 4)
            except RuntimeError:
                print("Error - curve_fit failed. Try with different initial guesses [p1, p2, ...] or "
                      "pass the boolean with_restriction = False")

        else:
            # model 3 with api - call BSWo_6
            try:
                if with_restriction:
                    bounds = ((-np.inf, 0, -np.inf, -np.inf, 0, -np.inf), (np.inf, np.inf, 0, 0, 1, np.inf))
                else:
                    bounds = ((-np.inf, -np.inf, -np.inf, -np.inf, 0, -np.inf), (np.inf, np.inf, np.inf
                                                                                 , np.inf, 1, np.inf))

                self._p_opt, self._p_conv = curve_fit(self.BSWo_6, self._data_train, self._data_train['BS&Wo'],
                                                      p0=[p1, p2, p3, p4, p5, p6], bounds=bounds)
                print("Fitting is done!")
                # Save to the parameter file
                # Second create the results text in .txt
                self.print_parameter_by_oil('Model - 3 (with API) -- applied to ' +
                                            self.file_name_without_extension(self.file_name), self._data_train,
                                            self._p_opt, self._p_conv, 6)
                self.print_covariance_by_oil('Model - 3 (with API) -- applied to ' +
                                             self.file_name_without_extension(self.file_name), self._p_conv, 6)
                self.print_data_BSW_estimated_by_oil('Model - 3 (with API) -- applied to ' +
                                                     self.file_name_without_extension(self.file_name),
                                                     self._data_train['BS&Wo'],
                                                     self.BSWo_6(self._data_train, *self._p_opt), self._sac_ratio_4, 6)
            except RuntimeError:
                print("Error - curve_fit failed. Try with different initial guesses [p1, p2, ...] or "
                      "pass the boolean with_restriction = False")

        return

    # function to calculate confidence interval for fitted curves
    @staticmethod
    def calculate_Confidence_Intervals(Model, Data):
        m_func = Model
        y_gp = Data['BS&Wo'] - m_func
        noise_std = 0.15  # measurement noise level
        length_scale = 6  # the smoothness of the true function
        kernel = RBF(length_scale, length_scale_bounds="fixed")
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2)
        x = Data['NoExp'].to_numpy()
        gpr.fit(x.reshape(-1, 1), y_gp)

        # compute GP predictions
        y_mean_pred, y_std_pred = gpr.predict(x.reshape(-1, 1), return_std=True)
        return y_mean_pred, y_std_pred, m_func

    # function to calculate determination coefficient (R2)
    @staticmethod
    def R(yexp, ytheo):
        ss_res = np.dot((yexp - ytheo), (yexp - ytheo))
        y_mean = np.mean(yexp)
        ne = len(yexp)
        ss_tot = np.dot((yexp - y_mean), (yexp - y_mean))
        myr = 1 - ss_res / ss_tot
        err = np.sqrt(ss_res / ne)
        return myr, err

    # function to print header in file of parameters
    @staticmethod
    def print_header_parameter(f, typeM, description_parameters):
        # Model 3 without API term
        if typeM == 4:
            f.write("_____________________________________________________________________________________________\n")
            f.write("The simulated model has the following shape:\n")
            f.write("BS&Wo = [C1*(2S/AC)^(0.5)]/[(2S/AC)*(1 + (AC/2S)^0.5)]\n")
            f.write('2S/AC = ' + '\u03B2' + '1' + '* T - \u03B2' + '2 * Ee^2 - \u03B2' + '4 * Ql - \u03B2' + '5 * Kp\n')
            f.write(description_parameters)
            f.write("_____________________________________________________________________________________________\n")

        # Model 3 with API term
        if typeM == 6:
            f.write("_____________________________________________________________________________________________\n")
            f.write("The simulated model has the following shape:\n")
            f.write("BS&Wo = [C1*(2S/AC)^(0.5)]/[(2S/AC)*(1 + (AC/2S)^0.5)] + \u03B26 * API\n")
            f.write('2S/AC = ' + '\u03B2' + '1' + '* T - \u03B2' + '2 * Ee^2 - \u03B2' + '4 * Ql - \u03B2' + '5 * Kp\n')
            f.write(description_parameters)
            f.write("_____________________________________________________________________________________________\n")

    # function to print parameter content in file
    @staticmethod
    def print_parameter_in_file(f, i, arr_popt, arr_mas_menos, arr_popt_err, typeM):
        # Model 3 without API term - write parameters
        if typeM == 4:
            if i == 0:
                f.write(
                    "\u03B2" + str(i + 1) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")
            if i == 1:
                f.write(
                    "\u03B2" + str(i + 1) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")
            if i == 2:
                f.write(
                    "\u03B2" + str(i + 2) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")
            if i == 3:
                f.write(
                    "\u03B2" + str(i + 2) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")
            if i == 4:
                f.write("C1: " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] + "\n")

        # Model 3 with API term - write parameters
        if typeM == 6:
            if i == 0:
                f.write(
                    "\u03B2" + str(i + 1) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")
            if i == 1:
                f.write(
                    "\u03B2" + str(i + 1) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")
            if i == 2:
                f.write(
                    "\u03B2" + str(i + 2) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")
            if i == 3:
                f.write(
                    "\u03B2" + str(i + 2) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")
            if i == 4:
                f.write("C1: " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] + "\n")

            if i == 5:
                f.write(
                    "\u03B2" + str(i + 1) + ": " + arr_popt[i] + " " + arr_mas_menos[i] + " " + arr_popt_err[i] +
                    "\n")

    # function to create the results file (txt) with optimized parameters
    def save_file_parameter_by_oil(self, Namefile, popt, popt_err, typeM):
        arr_popt = np.array(popt).astype(str)
        arr_popt_err = np.array(popt_err).astype(str)
        nf = "Parameters of " + Namefile + ".txt"
        arr_mas_menos = np.empty(len(popt), dtype=object)
        index = 0
        description_parameters = "T - Temperature (Celsius degree); Ee - Electric potential gradient (KV/cm); Ql - " \
                                 "inlet volumetric liquid flow\n (L/h); Kp: ratio between oil viscosity and " \
                                 "difference of water and oil density (cm2/s)\nsee more information: 10.1021/acs." \
                                 "energyfuels.7b03602\n"
        while index < len(popt):
            arr_mas_menos[index] = '\u00B1'
            index += 1
        f = open(nf, "w", encoding="utf-8")
        self.print_header_parameter(f, typeM, description_parameters)

        for i in range(len(popt)):
            self.print_parameter_in_file(f, i, arr_popt, arr_mas_menos, arr_popt_err, typeM)
        f.close()

    # function to calculate write parameter results in file
    def print_parameter_by_oil(self, Description, Data, popt, pconv, typeM):
        freedom_degree = len(Data['NoExp']) - len(popt) - 1
        # t-student values. Only valid until 60 experiments
        t_student95 = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
                       1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725,
                       1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697,
                       1.697, 1.697, 1.697, 1.697, 1.697, 1.697, 1.697, 1.697, 1.697, 1.684,
                       1.684, 1.684, 1.684, 1.684, 1.684, 1.684, 1.684, 1.684, 1.684, 1.684,
                       1.684, 1.684, 1.684, 1.684, 1.684, 1.684, 1.684, 1.684, 1.684, 1.671]
        index = 61
        # for number of experiments between 60 and 100
        while index < 100:
            t_student95.append(1.671)
            index += 1
        # for number of experiments between 100 and  140
        while index < 139:
            t_student95.append(1.66)
            index += 1
        # for number of experiments greater than 140 and less than 10 000
        while index < 10000:
            t_student95.append(1.656)
            index += 1
        # calculating parameter error
        self.err_parameter = t_student95[freedom_degree] * np.sqrt(np.diag(pconv))

        self.save_file_parameter_by_oil(Description, popt, self.err_parameter, typeM)

    # print type of parameter in Greek for different models
    @staticmethod
    def Print_type_Greek(i, typeM):

        # Model 3 without API
        if typeM == 4:
            if i == 0:
                str_greek = '\u03B2' + '1'
            if i == 1:
                str_greek = '\u03B2' + '2'
            if i == 2:
                str_greek = '\u03B2' + '4'
            if i == 3:
                str_greek = '\u03B2' + '5'
            if i == 4:
                str_greek = 'C1'

        # Model 3 with API
        if typeM == 6:
            if i == 0:
                str_greek = '\u03B2' + '1'
            if i == 1:
                str_greek = '\u03B2' + '2'
            if i == 2:
                str_greek = '\u03B2' + '4'
            if i == 3:
                str_greek = '\u03B2' + '5'
            if i == 4:
                str_greek = 'C1'
            if i == 5:
                str_greek = '\u03B2' + '6'

        return str_greek

    # convert covariance to correlation matrix
    @staticmethod
    def correlation_from_covariance(covariance):
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation

    @staticmethod
    def print_header_covariance(f, typeM):
        if typeM == 4:
            f.write('-----------------------------------------------------------------------------------------' + '\n')
            f.write('%11s %11s %11s %9s %9s' % ('\u03B2' + '1', '\u03B2' + '2', '\u03B2' + '4', '\u03B2' + '5', 'C1'))
            f.write('\n')

        if typeM == 6:
            f.write('-----------------------------------------------------------------------------------------' + '\n')
            f.write('%11s %11s %11s %11s %9s %9s' % ('\u03B2' + '1', '\u03B2' + '2', '\u03B2' + '4', '\u03B2' + '5',
                                                     'C1', '\u03B2' + '6'))
            f.write('\n')

    def print_covariance_by_oil(self, Description, matrix, typeM, to_console=False):
        if to_console:
            print('----------------------------------------------------------')
            print('Matrix of variance-covariance of parameters of ' + Description + '\n')
        nf = "Matrix of variance-covariance of parameters of " + Description + ".txt"
        f = open(nf, "w", encoding="utf-8")
        f.write('-----------------------------------------------------------------------------------------\n')
        title = 'Variance - covariance Matrix  of the parameters: '
        f.write(title + '\n')
        self.print_header_covariance(f, typeM)
        # Loop over each row of covariance matrix
        write_legend = True
        for i in range(len(matrix)):
            if write_legend:
                f.write('%2.2s' % (self.Print_type_Greek(i, typeM)))
                write_legend = False
            # Loop over each column in the current row
            for j in range(len(matrix[i])):
                # Print element at row i, column j
                if to_console:
                    msg_to_print = '{x:11.4f}'
                    print(msg_to_print.format(x=round(matrix[i][j], 4)), end=' ')
                f.write('%11.7s' % str(round(matrix[i][j], 4)))
            # Print a new line after each row
            print()
            f.write('\n')
            write_legend = True
        if to_console:
            print('----------------------------------------------------------')
        f.write(
            '------------------------------------------------------------------------------------------------' + '\n')
        x = Decimal(str(np.linalg.cond(matrix)))
        f.write('Condition number (K): ' + '{:.2e}'.format(x) + '\n')
        if to_console:
            print('----------------------------------------------------------')
        title = 'Correlation matrix  of the parameters: '
        f.write(title + '\n')
        f.write('-----------------------------------------------------------------------------------------' + '\n')
        self.print_header_covariance(f, typeM)
        R = self.correlation_from_covariance(matrix)
        # Loop over each row of Correlation matrix
        write_legend = True
        for i in range(len(R)):
            if write_legend:
                f.write('%2.2s' % (self.Print_type_Greek(i, typeM)))
                write_legend = False
            # Loop over each column in the current row
            for j in range(len(R[i])):
                # Print element at row i, column j
                if to_console:
                    msg_to_print = '{x:11.4f}'
                    print(msg_to_print.format(x=round(R[i][j], 4)), end=' ')
                f.write('%11.7s' % str(round(R[i][j], 4)))
            # Print a new line after each row
            print()
            f.write('\n')
            write_legend = True
        if to_console:
            print('----------------------------------------------------------')
        f.close()

    # print some statistic in file of the regression
    def print_data_BSW_estimated_by_oil(self, Description, yexp, ytheo, ratio_2SAC, typeM):
        nf = "Fitting results - " + Description + ".txt"
        ss_res = (yexp - ytheo)
        ss_err_2 = np.dot((yexp - ytheo), (yexp - ytheo))
        arr_res = np.array(round(ss_res, 5).astype(str))
        arr_ytheo = np.array(round(ytheo, 5).astype(str))
        arr_yexp = np.array(round(yexp, 5).astype(str))
        arr_2SAC = np.array(round(ratio_2SAC, 5).astype(str))
        P = 0
        N = len(yexp)
        if typeM == 4:
            P = 5  # number of parameters
        if typeM == 6:
            P = 6

        # -Statistic-
        # Determination of Aikarke information corrected
        lnL = 0.5 * (-N * ((math.log(math.e, 2 * math.pi)) + 1 - math.log(math.e, N) + math.log(math.e, ss_err_2)))
        AIC = 2 * P - 2 * lnL
        AICc = AIC + ((2 * P * (P + 1)) / (N - P - 1))
        f = open(nf, "w")
        f.write("No. Exp   BS&Wo(exp)   BS&Wo(simulated)     2S/AC (estimated)     residuals [BS&Wo(exp) - "
                "BS&Wo(theo)]\n")
        for i in range(len(yexp)):
            f.write('%-11s  %-11s %6s %23s %36s\n' % (str(i), arr_yexp[i], arr_ytheo[i], arr_2SAC[i], arr_res[i]))
        R_model, err = self.R(yexp, ytheo)
        chi_square_test_statistic, p_value = self.chi2_test(yexp, ytheo, P + 1)  # coded here
        f.write("____________________________________________________________________________________________" + "\n")
        f.write("Determination coefficient (R2): " + str(round(R_model, 2)) + "\n")
        R2_ajustado = (1 - (1 - R_model) * ((N - 1) / (N - P)))
        f.write("Adjusted determination coefficient (R2): " + str(round(R2_ajustado, 2)) + "\n")
        f.write("Corrected Akaike Information Criterion (AIcc): " + str(round(AICc, 2)) + "\n")
        f.write("____________________________________________________________________________________________\n")
        f.write("Chi-Square Goodness of Fit Test\n")
        f.write("Chi-square estimated: " + str(round(chi_square_test_statistic, 2)) + "\n")
        f.write("P-value: " + str(round(p_value, 2)) + "\n")

        if chi_square_test_statistic < p_value:
            f.write('Acceptation of null hypothesis (Ho) [95% of reliability]: The model  satisfactorily fits the '
                    'experimental data\n')
        else:
            f.write("Acceptation of alternative hypothesis (Ha) [95% of reliability]: Significant differences "
                    "between experimental and estimated results data\n")
        f.write("____________________________________________________________________________________________\n")

        f.close()

    # To define colors in figures
    @staticmethod
    def set_environment_color(axis: object, axis_color, background_color, location_legend):
        axis.spines['left'].set_color(axis_color)
        axis.spines['right'].set_color(axis_color)
        axis.spines['bottom'].set_color(axis_color)
        axis.spines['top'].set_color(axis_color)
        axis.tick_params(axis='both', colors=axis_color)
        axis.set_facecolor(background_color)
        axis.tick_params(bottom=True, left=True)
        legend = axis.legend(loc=location_legend, ncols=2)
        legend.get_frame().set_facecolor('#ffffff')

    # function to plot figures of simulation result and experimental data - filtered or not -
    def my_plot_with_confidence_interval(self, i, Model, Data, label_data, label_model, minX, maxX, minY,
                                         maxY, SaveFig, xlegR, ylegR, showerr, location_legend, plot_not_filtered):
        self.fig[i] = plt.figure()
        self.ax[i] = self.fig[i].add_subplot(111)
        self.ax[i].axis([minX, maxX, minY, maxY])
        self.ax[i].tick_params(colors='black', which='both', labelsize=12, pad=5)
        self.ax[i].plot(Data['NoExp'], Model, linestyle='-', color='k', label=label_model)

        # Plotting data with unfiltered values
        if self.kalman_filtered:
            self.ax[i].scatter(Data['NoExp'], Data['BS&Wo'], s=15, c='b', marker="s", label='BS&Wo Exp (Filtered)')
            if plot_not_filtered:
                self.ax[i].plot(self.data1['NoExp'], self.data1['BS&Wo'], linestyle='--', color='g',
                                label='BS&Wo Exp')
        else:
            self.ax[i].scatter(Data['NoExp'], Data['BS&Wo'], s=15, c='b', marker="s", label='BS&Wo Exp')

        # plot mean prediction and confidence intervals
        if showerr:
            y_mean_pred, y_std_pred, m_func = self.calculate_Confidence_Intervals(Model, Data)
            self.ax[i].plot(Data['NoExp'], y_mean_pred + m_func, linestyle='-', label="Mean prediction")
            self.ax[i].fill_between(Data['NoExp'], y_mean_pred - 1.96 * y_std_pred + m_func,
                                    y_mean_pred + 1.96 * y_std_pred + m_func, alpha=0.5, label=r"Confidence bands"
                                                                                               r" (95%)")
            self.y_mean_pred.append(y_mean_pred + m_func)
            self.y_max_conf.append(y_mean_pred + 1.96 * y_std_pred + m_func)
            self.y_min_conf.append(y_mean_pred - 1.96 * y_std_pred + m_func)

        r_model, err = self.R(Data['BS&Wo'], Model)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)  # bbox features
        self.ax[i].set_xlabel('No. Exp', fontsize=14, fontweight='bold')
        self.ax[i].set_ylabel('BS&Wo', fontsize=14, fontweight='bold')
        self.ax[i].text(xlegR, ylegR + 0.1, u"R\u00b2: {:0.2f}".format(r_model), transform=self.ax[i].transAxes,
                        fontsize=12, verticalalignment='top', bbox=props)
        self.ax[i].text(xlegR, ylegR, u"error: {:0.2f}".format(err), transform=self.ax[i].transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
        self.set_environment_color(self.ax[i], 'black', 'white', location_legend)
        if SaveFig:
            plt.savefig(label_model + ' ' + label_data + 'created-with-BSW class.jpg', dpi=300)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.tight_layout()
        plt.show()

    # function to predict values
    def Predict(self, data=None, plot_exp_data=True, showErr=True):
        if data is None:
            data = self._data_train

        if len(data) != 0:
            self.counter_fig += 1
            # First Plot
            if not self._api:

                # plot results of optimization
                self.my_plot_with_confidence_interval(self.counter_fig, self.BSWo_4(data, *self._p_opt),
                                                      data, 'Exp. data', 'Model - 3 (without API)', 0,
                                                      len(data['BS&Wo']) + 1, 0.0, max(data['BS&Wo']) + 0.5,
                                                      True, 0.1, 0.8, showErr, 'upper center',
                                                      plot_exp_data)

            else:
                self.my_plot_with_confidence_interval(self.counter_fig, self.BSWo_6(data, *self._p_opt),
                                                      data, 'Exp. data', 'Model - 3 (with API)', 0,
                                                      len(data['BS&Wo']) + 1, 0.0, max(data['BS&Wo']) + 0.5,
                                                      True, 0.1, 0.8, showErr, 'upper center',
                                                      plot_exp_data)
        else:
            print('!The data used as parameter in Predict() is empty. No plotting of results. Please use a value '
                  'for cut parameter (e.g. 0.1) in load_data() to obtain an array full')

    # Kernel for sensitivity (Model 3 - without API) Data is not passed as a data_frame, if not as array
    @staticmethod
    def Kernel_BSWo_4_sentivity(data, a1, a2, a4, a5, c):
        sacratio = a1 * data[0] - a2 * np.power(data[1], 2) - a4 * data[2] - a5 * data[3]
        return (c * np.exp(np.sqrt(sacratio))) / (sacratio * (1 + np.sqrt(1 / sacratio)))

    # Kernel for sensitivity (Model 3 - with API) Data is not passed as a data_frame, if not as array
    @staticmethod
    def Kernel_BSWo_6_sentivity(data, a1, a2, a4, a5, c, a6):
        sacratio = a1 * data[0] - a2 * np.power(data[1], 2) - a4 * data[2] - a5 * data[3]
        return ((c * np.exp(np.sqrt(sacratio))) / (sacratio * (1 + np.sqrt(1 / sacratio)))) + a6 * data[4]

    # function to save in file the sensitivity results of models - by input variables -
    def save_sensitivity(self, Si):
        nf = "Sensitivity on input variables - Model 3"
        if self._api:
            nf += "(with API) --- applied to " + self.file_name_without_extension(self.file_name) + ".txt"
        else:
            nf += "(without API) --- applied to " + self.file_name_without_extension(self.file_name) + ".txt"
        f = open(nf, "w", encoding='utf-8')
        f.write('**********************Sensitivity results*********************' + '\n')
        f.write('First order sensitivity (Temperature): ' + str(Si['S1'][0]) + ' \u00B1 ' + str(Si['S1_conf'][0]) +
                '\n')
        f.write('First order sensitivity (Potential gradient): ' + str(Si['S1'][1]) + ' \u00B1 ' + str(Si['S1_conf'][1])
                + '\n')
        f.write('First order sensitivity (volumetric flux): ' + str(Si['S1'][2]) + ' \u00B1 ' + str(Si['S1_conf'][2])
                + '\n')
        f.write('First order sensitivity (Kp): ' + str(Si['S1'][3]) + ' \u00B1 ' + str(Si['S1_conf'][3]) + '\n')
        if self._api:
            f.write('First order sensitivity (API): ' + str(Si['S1'][4]) + ' \u00B1 ' + str(Si['S1_conf'][4]) + '\n')
        f.write('--------------------------------------------------------------' + '\n')
        f.write('[T, Ee] second order sensitivity: ' + str(Si['S2'][0, 1]) + ' \u00B1 ' + str(Si['S2_conf'][0, 1]) +
                '\n')
        f.write('[T, Ql] second order sensitivity: ' + str(Si['S2'][0, 2]) + ' \u00B1 ' + str(Si['S2_conf'][0, 2]) +
                '\n')
        f.write('[T, Kp] second order sensitivity: ' + str(Si['S2'][0, 3]) + ' \u00B1 ' + str(Si['S2_conf'][0, 3]) +
                '\n')
        if self._api:
            f.write('[T, API] second order sensitivity: ' + str(Si['S2'][0, 4]) + ' \u00B1 ' + str(Si['S2_conf'][0, 4])
                    + '\n')
        f.write('[Ee, Ql] second order sensitivity: ' + str(Si['S2'][1, 2]) + ' \u00B1 ' + str(Si['S2_conf'][1, 2]) +
                '\n')
        f.write('[Ee, Kp] second order sensitivity: ' + str(Si['S2'][1, 3]) + ' \u00B1 ' + str(Si['S2_conf'][1, 3]) +
                '\n')
        if self._api:
            f.write('[Ee, API] second order sensitivity: ' + str(Si['S2'][1, 4]) + ' \u00B1 ' + str(Si['S2_conf'][1, 4])
                    + '\n')
        f.write('[Ql, Kp] second order sensitivity: ' + str(Si['S2'][2, 3]) + ' \u00B1 ' + str(Si['S2_conf'][2, 3]) +
                '\n')
        if self._api:
            f.write('[Ql, API] second order sensitivity: ' + str(Si['S2'][2, 4]) + ' \u00B1 ' + str(Si['S2_conf'][2, 4])
                    + '\n')
        f.write('--------------------------------------------------------------' + '\n')
        f.write('Total sensitivity (Temperature): ' + str(Si['ST'][0]) + ' \u00B1 ' + str(Si['ST_conf'][0]) + '\n')
        f.write('Total sensitivity (Potential gradient): ' + str(Si['ST'][1]) + ' \u00B1 ' + str(Si['ST_conf'][1]) +
                '\n')
        f.write('Total sensitivity (Volumetric flux): ' + str(Si['ST'][2]) + ' \u00B1 ' + str(Si['ST_conf'][2]) +
                '\n')
        f.write('Total sensitivity (Kp): ' + str(Si['ST'][3]) + ' \u00B1 ' + str(Si['ST_conf'][3]) + '\n')
        if self._api:
            f.write('Total sensitivity (API): ' + str(Si['ST'][4]) + ' \u00B1 ' + str(Si['ST_conf'][4]) + '\n')

    # Sensitivity analysis on input variables (T, Ee, Ql, Kp) in the model 3
    def sensitivity_analysis_on_input_variables(self, exp_nsample=10, radial=False, to_console=False):

        nsample = 2 ** exp_nsample # the higher exponent, the higher computing cost
        if self._api:
            T_min, Ee_min, Ql_min, Kp_min, API_min = min(self._data['T']), min(self._data['Ee']), min(self._data['Ql']), \
                min(self._data['Kp']), min(self._data['API'])
            T_max, Ee_max, Ql_max, Kp_max, API_max = max(self._data['T']), max(self._data['Ee']), max(self._data['Ql']), \
                max(self._data['Kp']), max(self._data['API'])
            a1 = self._p_opt[0]
            a2 = self._p_opt[1]
            a4 = self._p_opt[2]
            a5 = self._p_opt[3]
            c = self._p_opt[4]
            a6 = self._p_opt[5]
            problem = {
                'num_vars': 5,
                'names': ['T', 'Ee', 'Ql', 'Kp', 'API'],
                'bounds': [[T_min, T_max],
                           [Ee_min, Ee_max],
                           [Ql_min, Ql_max],
                           [Kp_min, Kp_max],
                           [API_min, API_max]]
            }
        else:
            T_min, Ee_min, Ql_min, Kp_min = min(self._data['T']), min(self._data['Ee']), min(self._data['Ql']), \
                min(self._data['Kp'])
            T_max, Ee_max, Ql_max, Kp_max = max(self._data['T']), max(self._data['Ee']), max(self._data['Ql']), \
                max(self._data['Kp'])
            a1 = self._p_opt[0]
            a2 = self._p_opt[1]
            a4 = self._p_opt[2]
            a5 = self._p_opt[3]
            c = self._p_opt[4]
            problem = {
                'num_vars': 4,
                'names': ['T', 'Ee', 'Ql', 'Kp'],
                'bounds': [[T_min, T_max],
                           [Ee_min, Ee_max],
                           [Ql_min, Ql_max],
                           [Kp_min, Kp_max]]
            }

        param_values = sample(problem, nsample)
        Y = np.zeros([param_values.shape[0]])
        for i, X in enumerate(param_values):
            if not self._api:
                Y[i] = self.Kernel_BSWo_4_sentivity(X, a1, a2, a4, a5, c)
            else:
                Y[i] = self.Kernel_BSWo_6_sentivity(X, a1, a2, a4, a5, c, a6)

        if not self._api:
            groups = {"Temperature": ["T"],
                      "Electric field": ["Ee"],
                      "Flux rate": ["Ql"],
                      "Oil Viscosity/density difference of phase": ["Kp"]}
            # To avoid failure with NaNs in the model
            meanY = Y[~np.isnan(Y)].mean()
            Y[np.isnan(Y)] = meanY
            if to_console:
                print('Sensitivity analysis on input variables - Model 3 (without API) -- applied to: ' +
                      self.file_name)
            Si = analyze(problem, Y, print_to_console=to_console)
            if radial:
                self.grouped_radial(Si, ['T', 'Ee', 'Ql', 'Kp'], groups=groups, threshold=0.025)
            else:
                Si.plot()
                plt.suptitle('Sobol sensitivity indices of each variable in the model \n (S1: first-order, '
                             'S2: second-order, ST = S1 + S2: total)', fontsize=14, fontweight='bold')
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
                plt.show()
            self.save_sensitivity(Si)
        else:
            groups = {"Temperature": ["T"],
                      "Electric field": ["Ee"],
                      "Flux rate": ["Ql"],
                      "Oil Viscosity/density difference of phase": ["Kp"],
                      "Oil API": ["API"]}
            meanY = Y[~np.isnan(Y)].mean()
            Y[np.isnan(Y)] = meanY
            if to_console:
                print('Sensitivity analysis on input variables - Model 3 (with API) -- applied to: ' + self.file_name)
            Si = analyze(problem, Y, print_to_console=to_console)
            if radial:
                self.grouped_radial(Si, ['T', 'Ee', 'Ql', 'Kp', 'API'], groups=groups, threshold=0.025)
            else:
                Si.plot()
                plt.suptitle('Sobol sensitivity indices of each variable in the model', fontsize=16, fontweight='bold')
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
                plt.show()
            self.save_sensitivity(Si)
        return

    # function to plot sensitivity on parameters
    def plot_sensitivity_analysis_on_parameters(self, S1s, multiples_plot, problem):
        self.counter_fig += 1
        if not self._api:
            marker_symbol = ['o', 'v', 'D', 'X', '*']
            color_symbol = ['g', 'r', 'b', 'k', 'c']
        else:
            marker_symbol = ['o', 'v', 'D', 'X', '*', 's']
            color_symbol = ['g', 'r', 'b', 'k', 'c', 'm']
        if not multiples_plot:
            for i in range(problem['num_vars']):
                # plot in different windows
                self.fig[i + self.counter_fig] = plt.figure()
                self.ax[i + self.counter_fig] = self.fig[i + self.counter_fig].add_subplot(111)
                self.ax[i + self.counter_fig].scatter(self._data['BS&Wo'], S1s[:, i], marker=marker_symbol[i],
                                                      c=color_symbol[i])
                self.ax[i + self.counter_fig].set_xlabel('BS&Wo')
                self.ax[i + self.counter_fig].set_ylabel(problem["names"][i] + ' (first-order sensitivity index)')
        else:
            # plot in an only window
            self.fig[self.counter_fig] = plt.figure()
            plt.suptitle('First-order (S1) sensitivity index of each parameter in the model', fontsize=14, fontweight='bold')
            self.ax[0] = self.fig[self.counter_fig].add_subplot(321)
            self.ax[1] = self.fig[self.counter_fig].add_subplot(323)
            self.ax[2] = self.fig[self.counter_fig].add_subplot(325)
            self.ax[3] = self.fig[self.counter_fig].add_subplot(222)
            self.ax[4] = self.fig[self.counter_fig].add_subplot(224)

            for i in range(problem['num_vars'] - 1):
                self.ax[i].scatter(self._data['BS&Wo'], S1s[:, i], marker=marker_symbol[i], c=color_symbol[i])
                if i == 2 or i == 4:
                    self.ax[i].set_xlabel('BS&Wo', fontsize=14, fontweight='bold')
                self.ax[i].set_ylabel(problem["names"][i], fontsize=14, fontweight='bold')

        print('Sensitivity analysis completed!')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.tight_layout()
        plt.show()

    # Sensitivity analysis on each parameter  in the Model 3
    def sensitivity_analysis_on_parameters(self, exp_nsample=6, multiples_plot=True):
        print('Working - Wait!')
        nsample = 2 ** exp_nsample  # the higher exponent, the higher computing cost
        if self._api:
            a1_min = self._p_opt[0] - self.err_parameter[0]
            a1_max = self._p_opt[0] + self.err_parameter[0]
            a2_min = self._p_opt[1] - self.err_parameter[1]
            a2_max = self._p_opt[1] + self.err_parameter[1]
            a4_min = self._p_opt[2] - self.err_parameter[2]
            a4_max = self._p_opt[2] + self.err_parameter[2]
            a5_min = self._p_opt[3] - self.err_parameter[3]
            a5_max = self._p_opt[3] + self.err_parameter[3]
            C1_min = self._p_opt[4] - self.err_parameter[4]
            C1_max = self._p_opt[4] + self.err_parameter[4]
            a6_min = self._p_opt[5] - self.err_parameter[5]
            a6_max = self._p_opt[5] + self.err_parameter[5]
            problem = {
                'num_vars': 6,
                'names': [r'$\beta_1$', r'$\beta_2$', r'$\beta_4$', r'$\beta_5$', r'$C_1$', r'$\beta_6$'],
                'bounds': [[a1_min, a1_max],
                           [a2_min, a2_max],
                           [a4_min, a4_max],
                           [a5_min, a5_max],
                           [C1_min, C1_max],
                           [a6_min, a6_max]]
            }
            param_values = sample(problem, nsample)
            print('Working - Wait!')
            y = np.array([self.BSWo_6(self._data, *params) for params in param_values])
            # To avoid failure with NaNs in the model
            meanY = y[~np.isnan(y)].mean()
            y[np.isnan(y)] = meanY
            # analyse the problem of variables with the sobol module
            sobol_indices = [analyze(problem, Y) for Y in y.T]
            S1s = np.array([s['S1'] for s in sobol_indices])
            self.plot_sensitivity_analysis_on_parameters(S1s, multiples_plot, problem)
        else:
            a1_min = self._p_opt[0] - self.err_parameter[0]
            a1_max = self._p_opt[0] + self.err_parameter[0]
            a2_min = self._p_opt[1] - self.err_parameter[1]
            a2_max = self._p_opt[1] + self.err_parameter[1]
            a4_min = self._p_opt[2] - self.err_parameter[2]
            a4_max = self._p_opt[2] + self.err_parameter[2]
            a5_min = self._p_opt[3] - self.err_parameter[3]
            a5_max = self._p_opt[3] + self.err_parameter[3]
            C1_min = self._p_opt[4] - self.err_parameter[4]
            C1_max = self._p_opt[4] + self.err_parameter[4]
            problem = {
                'num_vars': 5,
                'names': [r'$\beta_1$', r'$\beta_2$', r'$\beta_4$', r'$\beta_5$', r'$C_1$'],
                'bounds': [[a1_min, a1_max],
                           [a2_min, a2_max],
                           [a4_min, a4_max],
                           [a5_min, a5_max],
                           [C1_min, C1_max]]
            }
            param_values = sample(problem, nsample)
            print('Working - Wait!')
            y = np.array([self.BSWo_4(self._data, *params) for params in param_values])
            # To avoid failure with NaNs in the model
            meanY = y[~np.isnan(y)].mean()
            y[np.isnan(y)] = meanY
            # analyse the problem of variables with the sobol module
            sobol_indices = [analyze(problem, Y) for Y in y.T]
            S1s = np.array([s['S1'] for s in sobol_indices])
            self.plot_sensitivity_analysis_on_parameters(S1s, multiples_plot, problem)

    # Determine correlation between parameters from covariance
    @staticmethod
    def correlation_from_covariance(covariance):
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation

    # Plot each axis
    @staticmethod
    def plot_axis(ax, data_x, data_y, legend, label_y, label_x, scatter=False, loc=(0.0, 1.04)):
        if scatter:
            if legend != '':
                ax.scatter(data_x, data_y, s=15, c='b', marker="s", label=legend)
                ax.legend(loc=loc, ncol=2)
            else:
                ax.scatter(data_x, data_y, s=15, c='b', marker="s")
        else:
            if legend != '':
                ax.plot(data_x, data_y, linestyle='--', color='g', label=legend)
                ax.legend(loc=loc, ncol=2)
            else:
                ax.plot(data_x, data_y, linestyle='--', color='g')
        ax.set_ylabel(label_y, fontsize=14, fontweight='bold')
        ax.set_xlabel(label_x, fontsize=14, fontweight='bold')

    # Show experimental data without filtered data
    def show_only_experimental_data(self):
        self.counter_fig += 1
        self.fig[self.counter_fig] = plt.figure()
        plt.suptitle('Model variables', fontsize=16, fontweight='bold')
        if self._api:
            ax1 = self.fig[self.counter_fig].add_subplot(321)
            ax2 = self.fig[self.counter_fig].add_subplot(322)
            ax3 = self.fig[self.counter_fig].add_subplot(323)
            ax4 = self.fig[self.counter_fig].add_subplot(324)
            ax5 = self.fig[self.counter_fig].add_subplot(325)
            ax6 = self.fig[self.counter_fig].add_subplot(326)

            self.plot_axis(ax1, self.data1['NoExp'], self.data1['BS&Wo'], 'Exp.', 'BS&Wo', '')
            self.plot_axis(ax2, self.data1['NoExp'], self.data1['Ee'], '', 'Ee', '')
            self.plot_axis(ax3, self.data1['NoExp'], self.data1['T'], '', 'T', '')
            self.plot_axis(ax4, self.data1['NoExp'], self.data1['Kp'], '', 'Kp', '')
            self.plot_axis(ax5, self.data1['NoExp'], self.data1['Ql'], '', 'Ql', 'NoExp')
            self.plot_axis(ax6, self.data1['NoExp'], self.data1['API'], '', 'API', 'NoExp')

        else:
            ax1 = self.fig[self.counter_fig].add_subplot(321)
            ax2 = self.fig[self.counter_fig].add_subplot(323)
            ax3 = self.fig[self.counter_fig].add_subplot(325)
            ax4 = self.fig[self.counter_fig].add_subplot(222)
            ax5 = self.fig[self.counter_fig].add_subplot(224)
            self.plot_axis(ax1, self.data1['NoExp'], self.data1['BS&Wo'], 'BS&Wo (Exp.)', 'BS&Wo', '')
            self.plot_axis(ax2, self.data1['NoExp'], self.data1['Ee'], 'Ee (Exp.)', 'Ee', '')
            self.plot_axis(ax3, self.data1['NoExp'], self.data1['T'], 'T (Exp.)', 'T', '')
            self.plot_axis(ax4, self.data1['NoExp'], self.data1['Kp'], 'Kp (Exp.)', 'Kp', '')
            self.plot_axis(ax5, self.data1['NoExp'], self.data1['Ql'], 'Ql (Exp.)', 'Ql', '')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

    # Plot the variables filtered with kalman filter along with experimental ones
    def show_data_filtered(self):

        if self.kalman_filtered:
            self.counter_fig += 1
            self.fig[self.counter_fig] = plt.figure()
            plt.suptitle('Model variables', fontsize=16, fontweight='bold')
            if self._api:

                ax1 = self.fig[self.counter_fig].add_subplot(321)
                ax2 = self.fig[self.counter_fig].add_subplot(322)
                ax3 = self.fig[self.counter_fig].add_subplot(323)
                ax4 = self.fig[self.counter_fig].add_subplot(324)
                ax5 = self.fig[self.counter_fig].add_subplot(325)
                ax6 = self.fig[self.counter_fig].add_subplot(326)

                self.plot_axis(ax1, self.df_filtered[5], self.df_filtered[0], 'Filtered', 'BS&Wo', 'NoExp',
                               True)
                self.plot_axis(ax1, self.data1['NoExp'], self.data1['BS&Wo'], 'Exp.', 'BS&Wo', '')

                self.plot_axis(ax2, self.df_filtered[5], self.df_filtered[1], '', 'Ee', 'NoExp', True)
                self.plot_axis(ax2, self.data1['NoExp'], self.data1['Ee'], '', 'Ee', '')

                self.plot_axis(ax3, self.df_filtered[5], self.df_filtered[2], '', 'T', 'NoExp', True)
                self.plot_axis(ax3, self.data1['NoExp'], self.data1['T'], '', 'T', '')

                self.plot_axis(ax4, self.df_filtered[5], self.df_filtered[3], '', 'Kp', 'NoExp', True)
                self.plot_axis(ax4, self.data1['NoExp'], self.data1['Kp'], '', 'Kp', '')

                self.plot_axis(ax5, self.df_filtered[5], self.df_filtered[4], '', 'Ql', 'NoExp', True)
                self.plot_axis(ax5, self.data1['NoExp'], self.data1['Ql'], '', 'Ql', 'NoExp')

                self.plot_axis(ax6, self.df_filtered[5], self.df_filtered[6], '', 'API', 'NoExp', True)
                self.plot_axis(ax6, self.data1['NoExp'], self.data1['API'], '', 'API', 'NoExp')

            else:
                ax1 = self.fig[self.counter_fig].add_subplot(321)
                ax2 = self.fig[self.counter_fig].add_subplot(323)
                ax3 = self.fig[self.counter_fig].add_subplot(325)
                ax4 = self.fig[self.counter_fig].add_subplot(222)
                ax5 = self.fig[self.counter_fig].add_subplot(224)
                self.plot_axis(ax1, self.df_filtered[5], self.df_filtered[0], 'Filtered', 'BS&Wo', 'NoExp',
                               True)
                self.plot_axis(ax1, self.data1['NoExp'], self.data1['BS&Wo'], 'Exp.', 'BS&Wo', '')

                self.plot_axis(ax2, self.df_filtered[5], self.df_filtered[1], '', 'Ee', 'NoExp', True)
                self.plot_axis(ax2, self.data1['NoExp'], self.data1['Ee'], '', 'Ee', '')

                self.plot_axis(ax3, self.df_filtered[5], self.df_filtered[2], '', 'T', 'NoExp', True)
                self.plot_axis(ax3, self.data1['NoExp'], self.data1['T'], '', 'T', '')

                self.plot_axis(ax4, self.df_filtered[5], self.df_filtered[3], '', 'Kp', 'NoExp', True)
                self.plot_axis(ax4, self.data1['NoExp'], self.data1['Kp'], '', 'Kp', '')

                self.plot_axis(ax5, self.df_filtered[5], self.df_filtered[4], '', 'Ql', 'NoExp', True)
                self.plot_axis(ax5, self.data1['NoExp'], self.data1['Ql'], '', 'Ql', '')
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.show()
        else:
            print('No data has been filtered. Please activate in the load_data() the option kalman_filtered=True')

    # Print correlation matrix for the parameters using seaborn
    def plot_correlation_matrix_with_heatmap(self, ):
        # Create the heatmap using the `heatmap` function of Seaborn
        self.counter_fig += 1
        self.fig[self.counter_fig] = plt.figure()
        ax1 = self.fig[self.counter_fig].add_subplot(111)
        if not self._api:
            y_axis_labels = x_axis_labels = ['$\mathit{\u03B2}_{1}$', '$\mathit{\u03B2}_{2}$', '$\mathit{\u03B2}_{4}$',
                                             '$\mathit{\u03B2}_{5}$', '$\mathit{C}_{1}$']  # labels for x-axis
        else:
            y_axis_labels = x_axis_labels = ['$\mathit{\u03B2}_{1}$', '$\mathit{\u03B2}_{2}$', '$\mathit{\u03B2}_{4}$',
                                             '$\mathit{\u03B2}_{5}$', '$\mathit{C}_{1}$', '$\mathit{\u03B2}_{6}$']

        # X - Y axis labels
        Matrix_R = self.correlation_from_covariance(self._p_conv)
        mask = np.zeros_like(Matrix_R, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        np.fill_diagonal(mask, False)
        sns.set_style('white')
        ax1 = sns.heatmap(Matrix_R, xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                          cmap='coolwarm', annot=True, annot_kws={"size": 14}, mask=mask)
        ax1.set_title('Parametric correlation Matrix ', fontdict={'fontsize': 16, 'fontweight': 'bold' })
        ax1.set_ylabel('Parameters', fontsize=15, fontweight='bold')
        ax1.set_xlabel('Parameters', fontsize=15, fontweight='bold')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

    # Determine auto vector and auto valor from covariance matrix
    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    # Determine which parameter is printed on the axis
    @staticmethod
    def which_parameter(fila_col):
        if fila_col == 0:
            parameter_in_latex = r'$\beta_1$'
        if fila_col == 1:
            parameter_in_latex = r'$\beta_2$'
        if fila_col == 2:
            parameter_in_latex = r'$\beta_4$'
        if fila_col == 3:
            parameter_in_latex = r'$\beta_5$'
        if fila_col == 4:
            parameter_in_latex = r'$C_1$'
        if fila_col == 5:
            parameter_in_latex = r'$\beta_6$'
        return parameter_in_latex

    # Kernel to print ellipses
    def print_ellipses(self, combination, input_index, sigma, multiples_plot):
        combination_filtered = [e for e in combination if e['row'] == input_index]
        # plot in an only window
        if multiples_plot:
            # Subplots are organized in a Rows x Cols Grid
            # Tot and Cols are known
            Tot = len(combination_filtered)
            Cols = 2
            # Compute Rows required
            Rows = Tot // Cols
            if Tot % Cols != 0:
                Rows += 1
            Position = range(1, Tot + 1)

            self.fig[self.counter_fig] = plt.figure()
            plt.suptitle('Error ellipses of the parameters in the model', fontsize=14, fontweight='bold')

        for i in range(len(combination_filtered)):
            cov1 = combination_filtered[i]['cov']
            xc = combination_filtered[i]['xc']
            yc = combination_filtered[i]['yc']
            vals, vecs = self.eigsorted(cov1)
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 3 * sigma * np.sqrt(vals)
            max_x = xc + w
            min_x = xc - w
            max_y = yc + h
            min_y = yc - h
            if not multiples_plot:
                self.fig[i + self.counter_fig] = plt.figure()
                self.ax[i + self.counter_fig] = self.fig[i + self.counter_fig].add_subplot(111)
                if max_x > max_y:
                    self.ax[i + self.counter_fig].set_xlim(min_x, max_x)
                else:
                    self.ax[i + self.counter_fig].set_ylim(min_y, max_y)
                self.ax[i + self.counter_fig].set_xlabel(self.which_parameter(combination_filtered[i]['row']),
                                                         fontsize=14, fontweight='bold')
                self.ax[i + self.counter_fig].set_ylabel(self.which_parameter(combination_filtered[i]['col']),
                                                         fontsize=14, fontweight='bold')
                ellipse = Ellipse(xy=(xc, yc), width=w, height=h, angle=theta, color='black', fill=True)
                ellipse.set_facecolor('blue')
                self.ax[i + self.counter_fig].add_artist(ellipse)
                self.ax[i + self.counter_fig].scatter(xc, yc)
            else:
                # add every single subplot to the figure with a for loop
                self.ax[i] = self.fig[self.counter_fig].add_subplot(Rows, Cols, Position[i])
                if max_x > max_y:
                    self.ax[i].set_xlim(min_x, max_x)
                else:
                    self.ax[i].set_ylim(min_y, max_y)
                self.ax[i].set_xlabel(self.which_parameter(combination_filtered[i]['row']),
                                      fontsize=14, fontweight='bold')
                self.ax[i].set_ylabel(self.which_parameter(combination_filtered[i]['col']),
                                      fontsize=14, fontweight='bold')
                ellipse = Ellipse(xy=(xc, yc), width=w, height=h, angle=theta, color='black', fill=True)
                ellipse.set_facecolor('blue')
                self.ax[i].add_artist(ellipse)
                self.ax[i].scatter(xc, yc)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.tight_layout()
        plt.show()

    # Print error ellipse for the parameters
    def error_ellipse(self, Bool_parameter, sigma=3, multiples_plot=False):
        combination = []
        for i in range(len(self._p_conv)):
            j = i + 1
            while j != len(self._p_conv):
                # Store the possible combinations between variables
                combination.append({'row': i, 'col': j, 'cov': []})
                j += 1
            # lets create to different covariance matrix
        # init array mcov with zeroes
        cols = rows = 2
        mcov = [[0 for i in range(cols)] for j in range(rows)]
        # Create the multiple combinations of ellipses between parameters and store in combination variable
        for i in range(len(combination)):
            mcov[0][0] = self._p_conv[combination[i]['row']][combination[i]['row']]
            mcov[0][1] = self._p_conv[combination[i]['row']][combination[i]['col']]
            mcov[1][0] = self._p_conv[combination[i]['col']][combination[i]['row']]
            mcov[1][1] = self._p_conv[combination[i]['col']][combination[i]['col']]
            combination[i]['cov'] = mcov
            combination[i]['xc'] = self._p_opt[combination[i]['row']] # ellipses center x
            combination[i]['yc'] = self._p_opt[combination[i]['col']] # ellipses center y
        if Bool_parameter["B1"]:
            input_index = 0
            self.print_ellipses(combination, input_index, sigma, multiples_plot)
        if Bool_parameter["B2"]:
            input_index = 1
            self.print_ellipses(combination, input_index, sigma, multiples_plot)
        if Bool_parameter["B4"]:
            input_index = 2
            self.print_ellipses(combination, input_index, sigma, multiples_plot)
        if Bool_parameter["B5"]:
            input_index = 3
            self.print_ellipses(combination, input_index, sigma, multiples_plot)
        if self._api:
            if Bool_parameter["B6"]:
                input_index = 4
                self.print_ellipses(combination, input_index, sigma, multiples_plot)
        else:
            print('The data does not contain API field')
            exit()




