import math
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_sets
import seiir_compartmental

"""
Runs an extended Kalman filter on Prof Flaxman's SEIIR predictions and
measurement data for New York State from mid-April to 7 July 2020.
"""

# https://github.com/ihmeuw/vivarium_uw_covid/blob/master/src/vivarium_uw_covid/components/seiir_compartmental.py#L11-L67
# https://github.com/ihmeuw/vivarium_uw_covid/

K0 = datetime.date(2020, 4, 12)


def main():
    # set constants
    the_state = 'NY'
    the_county = data_sets.get_fips(the_state, 'Albany')
    constants = {
        'alpha': 0.948786,
        'gamma1': 0.500000,
        'gamma2': 0.662215,
        'sigma': 0.266635,
        'theta': 6.000000
        }
    P_mult = 1
    Q_mult = 1
    Rn_mult = 5*10**-4
    Q = Q_mult * np.eye(5)
    P = P_mult * np.eye(5)
    Rn = Rn_mult * np.eye(5)
    the_title = 'P_init = ' + str(P_mult) + '*Iden, Q = ' + str(Q_mult) + '*Iden, Rn = ' + str(Rn_mult) + '*Iden'

    # generate data
    seiir, beta, fb_data, case_data = get_data_sets(data_sets.STATES[the_state],
                                                    fips=the_county)

    if the_county is None:
        b = None
    else:
        county_pop = data_sets.get_pop_county(the_county)
        state_pop = data_sets.get_pop_state(data_sets.STATES[the_state])
        b = county_pop / state_pop


    # calculate moving averages on the fb and case data
    prop_ma7, case_ma7_all = calc_ma7(fb_data, case_data)
    # ensure the case rate data is the same date range as the prop data
    case_ma7 = case_ma7_all['2020-04-12':'2020-07-07']

    # get starting compartment values for the state level
    x_hat_state_k0, beta_k0 = get_predicts_prior(0, seiir, beta)

    # approximate rho values
    # I2_county = b * I2
    # prop_county = rho1 * I2_county
    # case_county = rho2 * I2_county
    if b is not None:
        x_hat_k0 = b * x_hat_state_k0
        I2_county = x_hat_k0[3, 0]
        rho1 = prop_ma7[0] / I2_county
        rho2 = case_ma7_all.loc['2020-04-12'] / I2_county

    # create empty dictionaries to hold the estimated values
    prop_est = {}
    case_est = {}
    seiir_pred = {}

    k = 0
    while K0 + datetime.timedelta(days=k) <= case_ma7.index[-1]:
        # each cycle of the while loop executes a step

        # get state level compartments
        x_hat_state_k, beta_k = get_predicts_prior(k, seiir, beta)

        # step the state level compartments one step forward (7 days)
        x_hat_state_k1 = step_seiir(x_hat_state_k, constants, beta_k)

        # convert the state level compartments to county level values
        x_hat_k = b * x_hat_state_k
        x_hat_k1 = b * x_hat_state_k1

        # get measurements
        z_k = np.array([[0],
                        [0],
                        [prop_ma7[k]],
                        [case_ma7[k]],
                        [0]])

        # predict step
        P = predict_step(x_hat_k1, P, Q, beta_k, constants)
        print('step -----------')
        print('P:')
        print(P)
        # update step
        x_hat_post, P_post = update_step(x_hat_k, x_hat_k1, P, Rn,
                                         rho1, rho2, z_k)
        print('P_post:')
        print(P_post)
        # store estimated values for proportion and case rate
        prop_est[K0 + datetime.timedelta(days=k+7)] = rho1 * x_hat_post[3, 0]
        case_est[K0 + datetime.timedelta(days=k+7)] = rho2 * x_hat_post[3, 0]
        seiir_pred[K0 + datetime.timedelta(days=k+7)] = rho2 * x_hat_k1[3, 0]

        # update the P and k
        P = P_post
        k += 1

    # create pandas series of the estimated case rate
    predicted_case = pd.Series(case_est)

    predicted_seiir_prior = pd.Series(seiir_pred)

    # plot findings
    fig, ax = plt.subplots(1)
    plt.plot(predicted_case.index, predicted_case, label='case rate predicted 7 days prior')
    plt.plot(case_ma7.index, case_ma7, label='case rate measured moving avg')
    plt.plot(predicted_seiir_prior.index, predicted_seiir_prior, label='IHME rho2*I2 predicted 7 days prior')
    plt.legend()
    plt.grid()
    plt.title('Albany County MA case rate vs. 7 day prediction using Kalman filter estimate')
    fig.suptitle('R = ' + str(Rn_mult) + ' * Identity Matrix')
    plt.show()


# functions to support the Kalman filtering
def get_predicts_prior(k, seiir, beta, k0=K0):
    day = k0 + datetime.timedelta(days=k)
    x_hat = np.array([[seiir['S'].loc[day]],
                      [seiir['E'].loc[day]],
                      [seiir['I1'].loc[day]],
                      [seiir['I2'].loc[day]],
                      [seiir['R'].loc[day]]])

    beta_k = beta.loc[day]

    return x_hat, beta_k


def step_seiir(x_hat, constants, beta_k, days=7):
    s_dict = {'S': x_hat[0, 0],
              'E': x_hat[1, 0],
              'I1': x_hat[2, 0],
              'I2': x_hat[3, 0],
              'R': x_hat[4, 0]}

    s = pd.Series(s_dict)

    for i in range(days):
        infectious = s.loc['I1'] + s.loc['I2']
        s = seiir_compartmental.compartmental_covid_step(s, s.sum(),
                                                         infectious,
                                                         constants['alpha'],
                                                         beta_k,
                                                         constants['gamma1'],
                                                         constants['gamma2'],
                                                         constants['sigma'],
                                                         constants['theta'])
    x_hat_future_prior = np.array([[s.loc['S']],
                                   [s.loc['E']],
                                   [s.loc['I1']],
                                   [s.loc['I2']],
                                   [s.loc['R']]])

    return x_hat_future_prior


def predict_step(x_hat_k1_prior, P, Q, beta_k, constants):
    S = x_hat_k1_prior[0, 0]
    E = x_hat_k1_prior[1, 0]
    I1 = x_hat_k1_prior[2, 0]
    I2 = x_hat_k1_prior[3, 0]
    R = x_hat_k1_prior[4, 0]
    N = S + E + I1 + I2 + R
    alpha = constants['alpha']
    sigma = constants['sigma']
    gamma1 = constants['gamma1']
    gamma2 = constants['gamma2']

    part_f_S = np.array([[-beta_k * math.pow(I1 + I2, alpha) / N],
                         [beta_k * math.pow(I1 + I2, alpha) / N],
                         [0],
                         [0],
                         [0]])

    part_f_E = np.array([[0],
                         [-sigma],
                         [sigma],
                         [0],
                         [0]])

    part_f_I1 = np.array([[-alpha * beta_k * S * math.pow(I1+I2, alpha-1) / N],
                          [alpha * beta_k * S * math.pow(I1+I2, alpha-1) / N],
                          [-gamma1],
                          [gamma1],
                          [0]])

    part_f_I2 = np.array([[-alpha * beta_k * S * math.pow(I1+I2, alpha-1) / N],
                          [alpha * beta_k * S * math.pow(I1+I2, alpha-1) / N],
                          [0],
                          [-gamma2],
                          [gamma2]])

    part_f_R = np.array([[0],
                         [0],
                         [0],
                         [0],
                         [0]])

    # 5x5
    f_jacob = np.concatenate([part_f_S, part_f_E, part_f_I1, part_f_I2,
                              part_f_R], axis=1)

    # 5x5
    P_k1_prior = np.matmul(np.matmul(f_jacob, P), np.transpose(f_jacob)) + Q
    return P_k1_prior


def update_step(x_hat, x_hat_k1, P_k1, Rn, rho1, rho2, z_k):
    # 5x5
    H_new = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, rho1, 0],
                      [0, 0, 0, rho2, 0],
                      [0, 0, 0, 0, 0]])

    # Si = H_new * P_k1 * H_new^T + Rn
    Si = np.matmul(np.matmul(H_new, P_k1), np.transpose(H_new)) + Rn

    # K_new = P_k1 * H_new^T * Si^(-1)
    K_new = np.matmul(np.matmul(P_k1, np.transpose(H_new)), np.linalg.inv(Si))
    y_new = np.matmul(H_new, x_hat)

    # 5x1
    diff = z_k - y_new

    x_hat_k1_post = x_hat_k1 + np.matmul(K_new, diff)

    P_k1_post = P_k1 - np.matmul(np.matmul(K_new, Si), np.transpose(K_new))

    return x_hat_k1_post, P_k1_post


def get_predict_measures():
    predictions = pd.read_csv(
        r'case_vs_symptom/kalman/predicted_measures/predictions.csv',
        header=None, names=['measure_date', 'prediction_date', 'prop', 'case',
                            'prop_pred7', 'case_pred7', 'prop_pred1',
                            'case_pred1'],
        index_col=False, usecols=[0, 4, 5])

    start_d = datetime.timedelta(days=int(predictions['measure_date'].iloc[0]))
    start = datetime.date(2019, 12, 31) + start_d
    end_d = datetime.timedelta(days=int(predictions['measure_date'].iloc[-1]))
    end = datetime.date(2019, 12, 31) + end_d

    measure_rng = pd.date_range(start=start, end=end)

    # code to save:
    # prediction_rng = pd.date_range(start=datetime.date(2019, 12, 31) +
    # datetime.timedelta(days=int(predictions['prediction_date'].iloc[0])),
    # end=datetime.date(2019, 12, 31) +
    # datetime.timedelta(days=int(predictions['prediction_date'].iloc[-1])))

    predictions.set_index(measure_rng, inplace=True)

    return predictions


def get_data_sets(state, fips=None):
    # the post hoc seiir model predictions provided by Prof Flaxman without
    # any kalman filtering:
    seiir = pd.read_csv(r'case_vs_symptom/kalman/ny_proj.csv', header=0,
                        index_col='date', parse_dates=True)
    # beta time series provided by Prof Flaxman:
    beta = pd.read_csv('vivarium_uw_covid/beta_t_ny.csv', header=0,
                       index_col='date', parse_dates=True, squeeze=True)

    fb_data = data_sets.create_symptom_df()

    if fips is None:
        case_data = data_sets.create_case_df_state()
        case_data_geo = case_data.loc[state]['case_rate'].copy()
        fb_data_geo = fb_data.loc[state].groupby('date').sum().copy()

    else:
        case_data = data_sets.create_case_df_county()
        case_data_geo = case_data.loc[fips]['case_rate'].copy()
        fb_data_geo = fb_data.loc[(slice(None), fips), :].copy()

    return seiir, beta, fb_data_geo, case_data_geo


def calc_ma7(fb_data, case_data):
    # the fb_data is a DataFrame while the case_data is a Series
    fb_ma7 = fb_data.rolling(window=7).mean()
    fb_ma7 = fb_ma7.iloc[6:, :]
    prop_ma7 = fb_ma7['num_stl'].div(fb_ma7['n'])

    case_ma7 = case_data.rolling(window=7).mean()
    case_ma7 = case_ma7.iloc[6:]
    return prop_ma7, case_ma7


main()
