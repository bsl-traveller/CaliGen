import pandas as pd
import os
import math

domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
METHODS = ["Uncalibrated", 'HB', 'Isotonic', 'CPCS', 'TransCal', 'HB TopLabel', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem', 'Beta abm', 'Beta am', 'Beta ab']

MAIN_COLUMNS = ["Uncalibrated", 'TS Source', 'TS Oracle',  'HB', 'Isotonic', 'Beta abm', 'Beta am', 'Beta ab', 'CPCS', 'TransCal', 'HB TopLabel', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem',
                'CaliGen Er', 'CaliGen TS Er', 'CaliGen Ensem Er', 'CaliGen ECE', 'CaliGen TS ECE', 'CaliGen Ensem ECE']

CALIGEN_METHODS = []
COLUMNS = ['Trainer', 'Calib', 'Domain', 'Valid', 'Train', 'Rho Error', 'Rho ECE', 'Uncalibrated', 'TS Source', 'TS Oracle',  'HB', 'Isotonic', 'Beta abm',
           'Beta am', 'Beta ab', 'CPCS', 'TransCal', 'HB TopLabel', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem', 'CaliGen Er', 'CaliGen TS Er', 'CaliGen Ensem Er', 'CaliGen ECE',
           'CaliGen TS ECE', 'CaliGen Ensem ECE']

for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    COLUMNS += [f'CaliGen {r}', f'CaliGen TS {r}', f'CaliGen Ensem {r}']
    CALIGEN_METHODS += [f'CaliGen {r}']
    
def best_rho_calib(model, metric, calib, loss):
    best_rho = dict()
    
    for i in range(len(domains)-1):
        for j in range(i+1, len(domains)):
            trainer = f"{domains[i]}_{domains[j]}"
            calib_filters = domains.copy()
            calib_filters.remove(domains[i])
            calib_filters.remove(domains[j])
            best_rho[trainer] = dict()
            path = os.path.join('..', 'results', model.lower(), 'DomainNet', loss, 'kfold', 'caligen')
            
            for cf in calib_filters:
                df = pd.read_csv(os.path.join(path, calib, trainer, cf, 'kfold_results.csv'))[2:8]
                best_rho[trainer][cf] = df[metric].idxmin() / 10
            
    return best_rho


def get_res_df(df, best_rho_er, best_rho_ece):
    
    df['Rho Error'] = [0] * len(df)
    df['Rho ECE'] = [0] * len(df)
    df['CaliGen Er'] = [0] * len(df)
    df['CaliGen TS Er'] = [0] * len(df)
    df['CaliGen Ensem Er'] = [0] * len(df)
    df['CaliGen ECE'] = [0] * len(df)
    df['CaliGen TS ECE'] = [0] * len(df)
    df['CaliGen Ensem ECE'] = [0] * len(df)
    
    for i in range(len(domains)-1):
        for j in range(i+1, len(domains)):
            trainer = f"{domains[i]}_{domains[j]}"
            calib_filters = domains.copy()
            calib_filters.remove(domains[i])
            calib_filters.remove(domains[j])
            
            for cf in calib_filters:
                three_filters = domains.copy()
                three_filters.remove(domains[i])
                three_filters.remove(domains[j])
                three_filters.remove(cf)
                l = len(df[(df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))])
                df.loc[((df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))), 'Rho Error'] = [best_rho_er[trainer][cf]] * l
                df.loc[((df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))), 'Rho ECE'] = [best_rho_ece[trainer][cf]] * l


                df.loc[((df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))), 'CaliGen Er'] = df[f"CaliGen {best_rho_er[trainer][cf]}"]
                df.loc[((df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))), 'CaliGen TS Er'] = df[f"CaliGen TS {best_rho_er[trainer][cf]}"]
                df.loc[((df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))), 'CaliGen Ensem Er'] = df[f"CaliGen Ensem {best_rho_er[trainer][cf]}"]
                df.loc[((df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))), 'CaliGen ECE'] = df[f"CaliGen {best_rho_ece[trainer][cf]}"]
                df.loc[((df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))), 'CaliGen TS ECE'] = df[f"CaliGen TS {best_rho_ece[trainer][cf]}"] 
                df.loc[((df['Trainer'] == trainer) & (df['Calib'] == '_'.join(three_filters))), 'CaliGen Ensem ECE'] = df[f"CaliGen Ensem {best_rho_ece[trainer][cf]}"]

    return df[COLUMNS]


def get_std(res_mean, res_std):
    res = []
    for i in range(len(res_std)):
        a = res_mean['Average'].iloc[i]
        means = []
        stds = []
        for domain in domains:
            means.append(res_mean[domain].iloc[i])
            stds.append(res_std[domain].iloc[i])
        res.append(math.sqrt(sum([(a-m)**2+s**2 for m, s in zip(means, stds)])/len(means)))
    return res