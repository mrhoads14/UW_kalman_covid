import math
import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import data_sets
import seiir_compartmental

"""
Runs an extended Kalman filter on Prof Flaxman's SEIIR predictions and
measurement data for New York State from mid-April to 7 July 2020.

S - Susceptible
E - Exposed
I1 - Presymptomatic
I2 - Symptomatic
R - Recovered
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

    # set initial values for Kalman filter parameters
    P_mult = 1
    Q_mult = 1

    # Rn is the R noise matrix; it remains constant thru the stepping of the
    # Kalman filter
    # prior to experiments, Rn_mult = 5*10**-4
    Rn_mult = 5*10**-8

    Rn_22 = 100
    Rn_32 = 1

    Rn_23 = 1
    Rn_33 = 1

    Rn = Rn_mult * np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, Rn_22, Rn_23, 0],
                             [0, 0, Rn_32, Rn_33, 0],
                             [0, 0, 0, 0, 0]])
    Q = Q_mult * np.eye(5)
    P = P_mult * np.eye(5)

    # colors used in plots
    purple = '#33016F'
    gold = '#9E7A27'
    gray = '#797979'

    # generate data
    seiir, fb_data, fb_data_val, case_data = get_data_sets(
        data_sets.STATES[the_state], fips=the_county)

    county_pop, state_pop = data_sets.get_pops(the_county)
    b = county_pop / state_pop

    # calculate moving averages on the fb and case data
    prop_ma7 = calc_fb_ma7(fb_data)
    case_ma7 = case_data.rolling(window=7).mean()
    case_ma7_all = case_ma7.iloc[6:]



    
    # ensure the case rate data is the same date range as the prop data
    case_ma7 = case_ma7_all['2020-04-12':'2020-07-07']

    # get starting compartment values for the state level
    x_hat_state_k0, beta_k0 = get_predicts_prior(K0, seiir)

    # approximate rho values
    # I2_county = b * I2
    # prop_county = rho1 * I2_county
    # case_county = rho2 * I2_county

    x_hat_k0 = b * x_hat_state_k0
    I2_county = x_hat_k0[3, 0]
    rho1 = prop_ma7.loc['2020-04-12'] / I2_county
    rho2 = case_ma7_all.loc['2020-04-12'] / I2_county

    # create empty dictionaries to hold the estimated values
    prop_est = {}
    case_est = {}
    seiir_pred = {}

    # each iteration of the while loop is a step for the Kalman filter
    d = K0
    while d <= prop_ma7.index[-1]:
        # each cycle of the while loop executes a step

        # get state level compartments
        x_hat_state_k, beta_k = get_predicts_prior(d, seiir)

        # step the state level compartments 7 days forward
        x_hat_state_k1 = step_seiir(x_hat_state_k, constants, beta_k)

        # convert the state level compartments to county level values
        x_hat_k = b * x_hat_state_k
        x_hat_k1 = b * x_hat_state_k1

        # get measurements
        z_k = np.array([[0],
                        [0],
                        [prop_ma7.loc[d]],
                        [case_ma7_all.loc[d]],
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
        indexDate = d + datetime.timedelta(days=7)
        prop_est[indexDate] = rho1 * x_hat_post[3, 0]
        case_est[indexDate] = rho2 * x_hat_post[3, 0]
        seiir_pred[indexDate] = b * x_hat_k1[3, 0]

        # update the P and d
        P = P_post
        d += datetime.timedelta(days=1)

    # create pandas series of the estimated case rate
    predicted_case = pd.Series(case_est)
    predicted_seiir_prior = pd.Series(seiir_pred)

    # compute squared error for the overlapping time period
    overlap = pd.date_range(start='2020-04-19', end='2020-07-07')
    overlap_dt = [x.to_pydatetime().date() for x in overlap]

    # error between the measured case rate (smoothed) and the
    # SEIIR model scaled to the county (without any Kalman adjustment)
    seiir_sq_err = 0

    # error between the measured case rate (smoothed) and the predicted
    # output of the Kalman filter
    kalman_sq_err = 0

    for day in overlap_dt:
        seiir_sq_err += (case_ma7_all[day] - predicted_seiir_prior[day])**2
        kalman_sq_err += (predicted_case[day] - case_ma7_all[day])**2

    print('SSE between seiir forecast and case rate:', seiir_sq_err)
    print('SSE between kalman forecast and case rate:', kalman_sq_err)


    # plot findings
    matplotlib.rcParams.update({'font.size': 20})
    plt.style.use('seaborn-whitegrid')
    tick_start = K0
    tick_end = predicted_seiir_prior.index[-1]

    week_interval = pd.date_range(start=tick_start, end=tick_end, freq='W')
    week_interval = [x.to_pydatetime().date() for x in week_interval]

    fig1, ax11 = plt.subplots(1)
    plt.sca(ax11)
    width = 4
    plt.plot(case_ma7.index, case_ma7, label='Case Positive Rate', c=gold,
             linewidth=width)
    plt.plot(predicted_seiir_prior.index, predicted_seiir_prior,
             label='IHME 7-day Forecast', c=gray, linewidth=width)
    plt.ylim(-2, 72)
    plt.xticks(week_interval, rotation=30, ha='right', rotation_mode='anchor')
    plt.ylabel('Number of Cases per Day')
    # plt.xlabel('Date')
    plt.legend(loc='upper left')

    fig2, ax21 = plt.subplots(1)
    plt.sca(ax21)
    plt.plot(case_ma7.index, case_ma7, label='Case Positive Rate', c=gold,
             linewidth=width)
    plt.plot(predicted_seiir_prior.index, predicted_seiir_prior,
             label='IHME 7-day Forecast', c=gray, linewidth=width)
    plt.plot(predicted_case.index, predicted_case, label='Our 7-Day Forecast',
             c=purple, linewidth=width)
    plt.ylim(-2, 72)
    plt.xticks(week_interval, rotation=30, ha='right', rotation_mode='anchor')
    # plt.xlabel('Date')
    plt.ylabel('Number of Cases per Day')
    plt.legend(loc='upper left')

    fig3, ax31 = plt.subplots(1)
    plt.sca(ax31)
    plt.plot(case_ma7.index, case_ma7, label='Case Positive Rate', c=gold,
             linewidth=width)
    plt.plot(predicted_seiir_prior.index, predicted_seiir_prior,
             label='IHME 7-day Forecast', c=gray, linewidth=width)
    plt.plot(predicted_case.index, predicted_case, label='Our 7-Day Forecast',
             c=purple, linewidth=width)
    plt.ylim(-2, 72)
    plt.xticks(week_interval, rotation=30, ha='right', rotation_mode='anchor')
    # plt.xlabel('Date')
    plt.ylabel('Number of Cases per Day')
    plt.legend(loc='upper left')

    ax32 = ax31.twinx()
    plt.sca(ax32)
    plt.plot(case_ma7.index, 100*prop_ma7, c='red', label='Facebook Symptom Rate',
             linewidth=width)
    plt.ylim(-.065, 2.5)
    plt.grid(axis='y', linestyle=':')
    plt.xticks(week_interval, rotation=30, ha='right', rotation_mode='anchor')

    plt.ylabel('Percentage of Positive Symptom Response')
    plt.legend(loc='upper right')


    plt.show()


# functions to support the Kalman filtering
def get_predicts_prior(day, seiir):
    x_hat = np.array([[seiir['S'].loc[day]],
                      [seiir['E'].loc[day]],
                      [seiir['I1'].loc[day]],
                      [seiir['I2'].loc[day]],
                      [seiir['R'].loc[day]]])

    beta_k = seiir['beta_pred'].loc[day]

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
    # P_k1_prior = f_jacob * P * f_jacob^T + Q
    P_k1_prior = np.matmul(np.matmul(f_jacob, P), np.transpose(f_jacob)) + Q
    return P_k1_prior


def update_step(x_hat, x_hat_k1, P_k1, Rn, rho1, rho2, z_k):
    # 5x5
    ep = 10**-10
    H = np.array([[ep, 0, 0, 0, 0],
                  [0, ep, 0, 0, 0],
                  [0, 0, ep, rho1, 0],
                  [0, 0, 0, rho2, 0],
                  [0, 0, 0, 0, ep]])

    # Si = H * P_k1 * H^T + Rn
    Si = np.matmul(np.matmul(H, P_k1), np.transpose(H)) + Rn

    # K_new = P_k1 * H^T * Si^(-1)
    K_new = np.matmul(np.matmul(P_k1, np.transpose(H)), np.linalg.inv(Si))
    y_new = np.matmul(H, x_hat)

    # 5x1
    diff = z_k - y_new

    x_hat_k1_post = x_hat_k1 + np.matmul(K_new, diff)

    # P_k1_post = P_k1 - np.matmul(np.matmul(K_new, Si), np.transpose(K_new))

    # joseph formulation
    joe2 = np.matmul(np.matmul(K_new, Rn), np.transpose(K_new))
    joe1 = np.eye(5) - np.matmul(K_new, H)
    P_k1_post = np.matmul(np.matmul(joe1, P_k1), np.transpose(joe1)) - joe2

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
    seiir = pd.read_csv(r'data/seiir_compartments_post-hoc_ny_state.csv', header=0,
                        index_col='date', parse_dates=True)
    # beta time series provided by Prof Flaxman:
    #beta = pd.read_csv(r'data/beta_t_ny.csv', header=0,
    #                   index_col='date', parse_dates=True, squeeze=True)

    fb_data = data_sets.create_symptom_df()
    fb_data_val = data_sets.create_symptom_df(valid=True)

    if fips is None:
        case_data = data_sets.create_case_df_state()
        case_data_geo = case_data.loc[state]['case_rate'].copy()
        fb_data_geo = fb_data.loc[state].groupby('date').sum().copy()
        fb_data_val_geo = fb_data_val.loc[state].groupby('date').sum().copy()

    else:
        case_data = data_sets.create_case_df_county()
        case_data_geo = case_data.loc[fips]['case_rate'].copy()
        fb_data_geo = fb_data.loc[(slice(None), fips), :].copy()
        fb_data_val_geo = fb_data_val.loc[(slice(None), fips), :].copy()

        # collapse down to a single index column (date)
        fb_data_geo = fb_data_geo.mean(level='date')
        fb_data_val_geo = fb_data_val_geo.mean(level='date')

    return seiir, fb_data_geo, fb_data_val_geo, case_data_geo


def calc_fb_ma7(fb_data):
    """
    Returns a Pandas series
    """
    # the fb_data is a DataFrame while the case_data is a Series
    fb_ma7 = fb_data.rolling(window=7).mean()
    fb_ma7 = fb_ma7.iloc[6:, :]
    prop_ma7 = fb_ma7['num_stl'].div(fb_ma7['n'])

    return prop_ma7




main()
