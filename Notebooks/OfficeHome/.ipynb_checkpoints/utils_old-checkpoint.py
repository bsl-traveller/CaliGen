import pandas as pd
import os
import math


METHODS = ["Uncalibrated", 'HB', 'Isotonic', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem', 'Beta abm', 'Beta am', 'Beta ab']

MAIN_COLUMNS = ["Uncalibrated", 'TS Source', 'TS Oracle',  'HB', 'Isotonic', 'Beta abm', 'Beta am', 'Beta ab', 'TS', 'Cluster NN', 'Cluster LR',
                'Cluster Ensem', 'CaliGen Er', 'CaliGen TS Er', 'CaliGen Ensem Er', 'CaliGen ECE', 'CaliGen TS ECE', 'CaliGen Ensem ECE']


COLUMNS = ['Trainer', 'Calib', 'Domain', 'Valid', 'Train', 'Rho Error', 'Rho ECE', 'Uncalibrated', 'TS Source', 'TS Oracle',  'HB', 'Isotonic',
           'Beta abm', 'Beta am', 'Beta ab', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem', 'CaliGen Er', 'CaliGen TS Er', 'CaliGen Ensem Er',
           'CaliGen ECE', 'CaliGen TS ECE', 'CaliGen Ensem ECE']

ABLATION_COLUMNS = ['Trainer', 'Calib', 'Domain', 'Valid', 'Train', 'Uncalibrated', 'CaliGen Er', 'CaliGen 0']

for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    COLUMNS += [f'CaliGen {r}', f'CaliGen TS {r}', f'CaliGen Ensem {r}']
    
    
def best_rho_calib(metric, calib, loss, folder='caligen'):
    best_rho = dict()
    
    domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    for domain in domains:
        calib_filters = ['Art', 'Clipart', 'Product', 'RealWorld']
        calib_filters.remove(domain)

        best_rho[domain] = dict()
        path = os.path.join('..', '..', 'results', 'OfficeHome', loss, 'kfold', folder)

        for cf in calib_filters:
            two_filters = ['Art', 'Clipart', 'Product', 'RealWorld']
            two_filters.remove(domain)
            two_filters.remove(cf)

            df = pd.read_csv(os.path.join(path, calib, domain, '_'.join(two_filters), 'kfold_results.csv'))[2:8]
            best_rho[domain]['_'.join(two_filters)] = df[metric].idxmin() / 10
            
    return best_rho


def get_res_df(df, best_rho_er, best_rho_ece, ablation=False):
    domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    df['Rho Error'] = [0] * len(df)
    df['Rho ECE'] = [0] * len(df)
    df['CaliGen Er'] = [0] * len(df)
    df['CaliGen TS Er'] = [0] * len(df)
    df['CaliGen Ensem Er'] = [0] * len(df)
    df['CaliGen ECE'] = [0] * len(df)
    df['CaliGen TS ECE'] = [0] * len(df)
    df['CaliGen Ensem ECE'] = [0] * len(df)
    for domain in domains:
        calib_filters = ['Art', 'Clipart', 'Product', 'RealWorld']
        calib_filters.remove(domain)

        for cf in calib_filters:
            two_filters = ['Art', 'Clipart', 'Product', 'RealWorld']
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
        to = res_mean['Art'].iloc[i]
        ts = res_std['Art'].iloc[i]
        co = res_mean['Clipart'].iloc[i]
        cs = res_std['Clipart'].iloc[i]
        ro = res_mean['Product'].iloc[i]
        rs = res_std['Product'].iloc[i]
        ao2 = res_mean['RealWorld'].iloc[i]
        as2 = res_std['RealWorld'].iloc[i]
        res.append(math.sqrt((((a-to)**2 + ts**2 + (a-co)**2 + cs**2 + (a-ro)**2 + rs**2 + (a-ao2)**2 + as2**2 ) / 4)))
    return res