import pandas as pd
import os
import math

domains = ['Art', 'Clipart', 'Product', 'RealWorld']
METHODS = ["Uncalibrated", 'HB', 'Isotonic', 'CPCS', 'TransCal', 'HB TopLabel', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem', 'Beta abm', 'Beta am', 'Beta ab']

MAIN_COLUMNS = ["Uncalibrated", 'TS Source', 'TS Oracle',  'HB', 'Isotonic', 'Beta abm', 'Beta am', 'Beta ab', 'CPCS', 'TransCal', 'HB TopLabel', 'TS', 'Cluster NN', 'Cluster LR',
                'Cluster Ensem', 'CaliGen Er', 'CaliGen TS Er', 'CaliGen Ensem Er', 'CaliGen ECE', 'CaliGen TS ECE', 'CaliGen Ensem ECE']
CALIGEN_METHODS = []

COLUMNS = ['Trainer', 'Calib', 'Domain', 'Valid', 'Train', 'Rho Error', 'Rho ECE', 'Uncalibrated', 'TS Source', 'TS Oracle',  'HB', 'Isotonic',
           'Beta abm', 'Beta am', 'Beta ab', 'CPCS', 'TransCal', 'HB TopLabel', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem', 'CaliGen Er', 'CaliGen TS Er', 'CaliGen Ensem Er',
           'CaliGen ECE', 'CaliGen TS ECE', 'CaliGen Ensem ECE']

ABLATION_COLUMNS = ['Trainer', 'Calib', 'Domain', 'Valid', 'Train', 'Uncalibrated', 'CaliGen Er', 'CaliGen 0']

for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    COLUMNS += [f'CaliGen {r}', f'CaliGen TS {r}', f'CaliGen Ensem {r}']
    CALIGEN_METHODS += [f'CaliGen {r}']
    
def best_rho_calib(model, metric, calib, loss, folder='caligen'):
    best_rho = dict()
        
    for domain in domains:
        calib_filters = domains.copy()
        calib_filters.remove(domain)

        best_rho[domain] = dict()
        path = os.path.join('..', 'results', model.lower(), 'OfficeHome', loss, 'kfold', folder)

        for cf in calib_filters:
            two_filters = domains.copy()
            two_filters.remove(domain)
            two_filters.remove(cf)

            df = pd.read_csv(os.path.join(path, calib, domain, '_'.join(two_filters), 'kfold_results.csv'))[2:8]
            best_rho[domain]['_'.join(two_filters)] = df[metric].idxmin() / 10
            
    return best_rho


def get_res_df(df, best_rho_er, best_rho_ece, ablation=False):
    df['Rho Error'] = [0] * len(df)
    df['Rho ECE'] = [0] * len(df)
    df['CaliGen Er'] = [0] * len(df)
    df['CaliGen TS Er'] = [0] * len(df)
    df['CaliGen Ensem Er'] = [0] * len(df)
    df['CaliGen ECE'] = [0] * len(df)
    df['CaliGen TS ECE'] = [0] * len(df)
    df['CaliGen Ensem ECE'] = [0] * len(df)
    for domain in domains:
        calib_filters = domains.copy()
        calib_filters.remove(domain)

        for cf in calib_filters:
            two_filters = domains.copy()
            two_filters.remove(domain)
            two_filters.remove(cf)
            l = len(df[(df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))])
            df.loc[((df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))), 'Rho Error'] = [best_rho_er[domain]['_'.join(two_filters)]] * l
            df.loc[((df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))), 'Rho ECE'] = [best_rho_ece[domain]['_'.join(two_filters)]] * l
                        
            
            df.loc[((df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))), 'CaliGen Er'] = df[f"CaliGen {best_rho_er[domain]['_'.join(two_filters)]}"]
            df.loc[((df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))), 'CaliGen TS Er'] = df[f"CaliGen TS {best_rho_er[domain]['_'.join(two_filters)]}"]
            df.loc[((df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))), 'CaliGen Ensem Er'] = df[f"CaliGen Ensem {best_rho_er[domain]['_'.join(two_filters)]}"]
            df.loc[((df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))), 'CaliGen ECE'] = df[f"CaliGen {best_rho_ece[domain]['_'.join(two_filters)]}"]
            df.loc[((df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))), 'CaliGen TS ECE'] = df[f"CaliGen TS {best_rho_ece[domain]['_'.join(two_filters)]}"] 
            df.loc[((df['Trainer'] == domain) & (df['Calib'] == '_'.join(two_filters))), 'CaliGen Ensem ECE'] = df[f"CaliGen Ensem {best_rho_ece[domain]['_'.join(two_filters)]}"]
            
    if ablation is True:
        return df[ABLATION_COLUMNS]
    else:
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