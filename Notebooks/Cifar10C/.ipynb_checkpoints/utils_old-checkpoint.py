import pandas as pd
import os
import math

domains = ['Train Filter', 'Calib Filter', 'Test Filter (1)', 'All Filter (2)', 'All Filter (3)', 'All Filter (4)', 'All Filter (5)']
METHODS = ["Uncalibrated", 'TS Source', 'TS Oracle',  'HB', 'Isotonic', 'Beta abm', 'Beta am', 'Beta ab', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem',
           'CaliGen Er', 'CaliGen TS Er', 'CaliGen Ensem Er', 'CaliGen ECE', 'CaliGen TS ECE', 'CaliGen Ensem ECE']

MAIN_COLUMNS = ['Domain'] + METHODS


COLUMNS = ['Domain', 'Valid', 'Train', 'Rho Error', 'Rho ECE', 'Uncalibrated', 'TS Source', 'TS Oracle',  'HB', 'Isotonic', 'Beta abm',
           'Beta am', 'Beta ab', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem', 'CaliGen Er', 'CaliGen TS Er', 'CaliGen Ensem Er', 'CaliGen ECE',
           'CaliGen TS ECE', 'CaliGen Ensem ECE']

for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    COLUMNS += [f'CaliGen {r}', f'CaliGen TS {r}', f'CaliGen Ensem {r}']
    
    
def best_rho_calib(model, metric, calib, loss, trainer='4filter_1'):
    df = pd.read_csv(os.path.join('..', '..', 'extra_results', model.lower(), 'cifar10c', loss, 'kfold', 'caligen', calib, trainer, 'kfold_results.csv'))[2:8]
    return df[metric].idxmin() / 10
    

def get_rho_df(df, best_rho_er, best_rho_ece):
    df['Rho Error'] = [best_rho_er] * len(df)
    df['Rho ECE'] = [best_rho_ece] * len(df)
    df['CaliGen Er'] = df[f"CaliGen {best_rho_er}"]
    df['CaliGen TS Er'] = df[f"CaliGen TS {best_rho_er}"]
    df['CaliGen Ensem Er'] = df[f"CaliGen Ensem {best_rho_er}"]
    df['CaliGen ECE'] = df[f"CaliGen {best_rho_ece}"]
    df['CaliGen TS ECE'] = df[f"CaliGen TS {best_rho_ece}"]
    df['CaliGen Ensem ECE'] = df[f"CaliGen Ensem {best_rho_ece}"]
    
    return df[MAIN_COLUMNS]

def get_level_filters(trainer, train_filters, valid_filters, test_filters):
    train_level = []
    valid_level = []
    rest_filters = dict()
    if trainer.split('_')[-1] == 'all':
        for i in range(1, 6):
            for tf, vf in zip(train_filters, valid_filters):
                train_level.append(f"{tf}_{i}")
                valid_level.append(f"{vf}_{i}")
        
        for i in range(1, 6):
            rest_filters[i] = []
            for fil in test_filters:
                rest_filters[i].append(f"{fil}_{i}")
    else:
        for tf, vf in zip(train_filters, valid_filters):
            train_level.append(f"{tf}_{trainer.split('_')[-1]}")
            valid_level.append(f"{vf}_{trainer.split('_')[-1]}")
        levels = [str(i) for i in range(1, 6)]
        levels.remove(trainer.split('_')[-1])
        rest_filters[trainer.split('_')[-1]] = []
        for fil in test_filters:
            rest_filters[trainer.split('_')[-1]].append(f"{fil}_{trainer.split('_')[-1]}")
        
        for i in levels:
            rest_filters[i] = []
            for fil in train_filters + valid_filters + test_filters:
                rest_filters[i].append(f"{fil}_{i}")
    train_level = ["original"] + train_level
    return train_level, valid_level, rest_filters


def get_res_df(res_columns, df, trainer, train_f, valid_f, rest_f):
    level = trainer.split('_')[-1]
    result = pd.DataFrame(columns=res_columns)
    result_std = pd.DataFrame(columns=res_columns)

    result['Train Filter'] = df[df['Domain'].isin(train_f)].mean()
    result_std['Train Filter'] = df[df['Domain'].isin(train_f)].std()
    result['Calib Filter'] = df[df['Domain'].isin(valid_f)].mean()
    result_std['Calib Filter'] = df[df['Domain'].isin(valid_f)].std()
    result[f'Test Filter ({level})'] = df[df['Domain'].isin(rest_f[level])].mean()
    result_std[f'Test Filter ({level})'] = df[df['Domain'].isin(rest_f[level])].std()
    levels = [str(i) for i in range(1, 6)]
    levels.remove(level)
    for i in levels:
        result[f'All Filter ({i})'] = df[df['Domain'].isin(rest_f[i])].mean()
        result_std[f'All Filter ({i})'] = df[df['Domain'].isin(rest_f[i])].std()
    
    result['Average'] = result[res_columns].mean(axis=1)
    #result_std['std'] = result[res_columns].std(axis=1)
    result = result.round(2)
    result_std = result_std.round(2)
    return result, result_std

def get_std_c():
    std['std'] = []
    for i in range(len(result)):
        a = result['Average'].iloc[i]
        to = result['Train Filter'].iloc[i]
        ts = result_std['Train Filter'].iloc[i]
        co = result['Calib Filter'].iloc[i]
        cs = result_std['Calib Filter'].iloc[i]
        ro = result['Test Filter (1)'].iloc[i]
        rs = result_std['Test Filter (1)'].iloc[i]
        ao2 = result['All Filter (2)'].iloc[i]
        as2 = result_std['All Filter (2)'].iloc[i]
        ao3 = result['All Filter (3)'].iloc[i]
        as3 = result_std['All Filter (3)'].iloc[i]
        ao4 = result['All Filter (4)'].iloc[i]
        as4 = result_std['All Filter (4)'].iloc[i]
        ao5 = result['All Filter (5)'].iloc[i]
        as5 = result_std['All Filter (5)'].iloc[i]
        std.append(math.sqrt((((a-to)**2 + ts**2 + (a-co)**2 + cs**2 + (a-ro)**2 + rs**2 + (a-ao2)**2 + as2**2 + (a-ao3)**2 + as3**2 + (a-ao4)**2 + as4**2 + (a-ao5)**2 + as5**2 ) / 7)))
        
def get_std(res_mean, res_std):
    res = []
    for i in range(len(result)):
        a = result['Average'].iloc[i]
        means = []
        stds = []
        for domain in domains:
            means.append(res_mean[domain].iloc[i])
            stds.append(res_std[domain].iloc[i])
        res.append(math.sqrt(sum([(a-m)**2+s**2 for m, s in zip(means, stds)])/len(means)))
    return res
