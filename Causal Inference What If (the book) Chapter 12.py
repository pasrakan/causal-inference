from collections import OrderedDict


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats
import matplotlib.pyplot as plt


nhefs_all = pd.read_csv('nhefs.csv')


restriction_cols = [
   'sex', 'age', 'race', 'wt82', 'ht', 'school', 'alcoholpy', 'smokeintensity'
]
missing = nhefs_all[restriction_cols].isnull().any(axis=1)
nhefs = nhefs_all.loc[~missing]

# To avoid 'SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.' in the following codes.
# Copy-on-Write will become the new default in pandas 3.0.
pd.options.mode.copy_on_write = True

nhefs['university'] = (nhefs["education"] == 5).astype('int')
nhefs['inactive'] = (nhefs["active"] == 2).astype('int')
nhefs['no_exercise'] = (nhefs["exercise"] == 2).astype('int')
ave_gain_quit = nhefs[nhefs["qsmk"] == 1]["wt82_71"].mean()
ave_gain_noquit = nhefs[nhefs["qsmk"] == 0]["wt82_71"].mean()


ols = smf.ols("wt82_71~qsmk", data=nhefs).fit()
# print(ols.summary().tables[1])


summaries = OrderedDict((
   ('age', 'mean'),
   ('sex', lambda x: (100 * (x == 0)).mean()),
   ('race', lambda x: (100 * (x == 0)).mean()),
   ('university', lambda x: 100 * x.mean()),
   ('wt71', 'mean'),
   ('smokeintensity', 'mean'),
   ('smokeyrs', 'mean'),
   ('no_exercise', lambda x: 100 * x.mean()),
   ('inactive', lambda x: 100 * x.mean())
))


table = nhefs.groupby('qsmk').agg(summaries)
table.sort_index(ascending=False, inplace=True)
table = table.T


table.index = [
   'Age, years',
   'Men, %',
   'White, %',
   'University education, %',
   'Weight, kg',
   'Cigarettes/day',
   'Years smoking',
   'Little or no exercise, %',
   'Inactive daily life, %'
]


# table.style.format("{:>0.1f}")


edu_dummies = pd.get_dummies(nhefs.education, prefix='edu')
exercise_dummies = pd.get_dummies(nhefs.exercise, prefix='exercise')
active_dummies = pd.get_dummies(nhefs.active, prefix='active')


nhefs = pd.concat(
   [nhefs, edu_dummies, exercise_dummies, active_dummies],
   axis=1
)


def logit_ip_f(y, X, data):
   """
   Returns
   -------
   Numpy array of IP weights


   """
   model = smf.logit(f"{y}~{X}", data=data).fit()
   weights = model.predict()
   indices = data[y].values == 0
   weights[indices] = 1 - weights[indices]
   return weights


X_ip = "sex + race + age + np.power(age,2) + edu_2 + edu_3 + edu_4 + edu_5 + smokeintensity + np.power(smokeintensity,2) + smokeyrs + np.power(smokeyrs,2) + exercise_1 + exercise_2 + active_1 + active_2 + wt71 + np.power(wt71,2)"


denoms = logit_ip_f("qsmk", X_ip, nhefs)
weights = 1 / denoms


wls = smf.wls("wt82_71~qsmk", data=nhefs, weights=weights)
print(wls.fit().summary().tables[1])


clustered = wls.fit(cov_type="cluster", cov_kwds={"groups": nhefs.seqn})
print(clustered.summary().tables[1])


gee = smf.gee("wt82_71~qsmk", data=nhefs, groups=nhefs.seqn, weights=weights)
print(gee.fit().summary().tables[1])


# Check that there is no association between sex and qsmk.
pd.crosstab(nhefs.sex, nhefs.qsmk, weights, aggfunc='sum')

# Stabilized IP weights
qsmk = (nhefs["qsmk"] == 1)
qsmk_mean = qsmk.astype('int').mean()
s_weights = weights
s_weights[qsmk] = qsmk_mean * s_weights[qsmk]
s_weights[~qsmk] = (1-qsmk_mean) * s_weights[~qsmk]

gee = smf.gee("wt82_71~qsmk", data=nhefs, groups=nhefs.seqn, weights=s_weights)
print(gee.fit().summary().tables[1])

# Standardization bootstrap function
# for _ in range(2000):
#    sample = nhefs_all.sample(n=nhefs_all.shape[0], replace=True)
#
#    block2 = sample[common_Xcols + ['zero', 'zero']]
#    block3 = sample[common_Xcols + ['one', 'smokeintensity']]
#
#    uncens = sample.loc[~sample.wt82.isnull()]
#    y = uncens.wt82_71
#    X = uncens[common_Xcols + ['qsmk', 'qsmk_x_smokeintensity']]
#    result = sm.OLS(y, X).fit()
#
#    block2_pred = result.predict(block2)
#    block3_pred = result.predict(block3)
#
#    boot_samples.append(block3_pred.mean() - block2_pred.mean())

