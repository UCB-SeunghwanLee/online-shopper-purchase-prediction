"""Builds the README figures in images/. Run from the project folder: python make_figures.py"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

BLUE, ORANGE, GRID, INK, MUTED = '#2a78d6', '#eb6834', '#e6e5e0', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.dpi': 150})
os.makedirs('images', exist_ok=True)

df = pd.read_csv('data/online_shoppers_intention.csv')
clean = df[df['VisitorType'] != 'Other'].drop_duplicates().copy()
clean['y'] = clean['Revenue'].astype(int)

# ---------- Figure 1: purchase rate by month and by visitor type ----------
order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month = clean.groupby('Month')['y'].agg(['mean', 'count']).reindex(order)
vis = clean.groupby('VisitorType')['y'].mean().rename({'New_Visitor': 'New', 'Returning_Visitor': 'Returning'})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [3, 1]})
a1.bar(month.index, month['mean'] * 100, color=BLUE, width=0.7)
a1.set_title('Purchase rate by month', loc='left', color=INK, fontsize=13)
a1.set_ylabel('Sessions ending in a purchase (%)')
for x, v in zip(month.index, month['mean'] * 100):
    if x in ('Nov', 'May', 'Feb'):
        a1.annotate(f'{v:.0f}%', (x, v), ha='center', va='bottom', xytext=(0, 3),
                    textcoords='offset points', color=INK)
a2.bar(vis.index, vis * 100, color=BLUE, width=0.6)
a2.set_title('By visitor type', loc='left', color=INK, fontsize=13)
for x, v in zip(vis.index, vis * 100):
    a2.annotate(f'{v:.0f}%', (x, v), ha='center', va='bottom', xytext=(0, 3),
                textcoords='offset points', color=INK)
for ax in (a1, a2):
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_ylim(0, 30)
plt.tight_layout()
plt.savefig('images/purchase_rate_month_visitor.png')
plt.close()

# ---------- Models (same setup as 02_modeling.ipynb) ----------
numeric = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
           'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'SpecialDay', 'PageValues']
categorical = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
X = pd.get_dummies(clean[numeric + categorical + ['Weekend']], columns=categorical, drop_first=True, dtype=int)
X['Weekend'] = X['Weekend'].astype(int)
y = clean['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


def models():
    return {
        'OLS (linear probability)': 'ols',
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000)),
        'Decision tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'Random forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced',
                                                random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 scale_pos_weight=pos_weight, random_state=42),
    }


auc, rf_no_pv = {}, None
for fs in ['With PageValues', 'Without PageValues']:
    cols = X.columns if fs == 'With PageValues' else X.columns.drop('PageValues')
    for name, m in models().items():
        if m == 'ols':
            lr = LinearRegression().fit(X_train[cols], y_train, sample_weight=compute_sample_weight('balanced', y_train))
            p = lr.predict(X_test[cols])
        else:
            m.fit(X_train[cols], y_train)
            p = m.predict_proba(X_test[cols])[:, 1]
            if fs == 'Without PageValues' and name == 'Random forest':
                rf_no_pv = (m, cols)
        auc[(name, fs)] = roc_auc_score(y_test, p)

# ---------- Figure 2: AUC with vs. without PageValues (dot plot) ----------
names = list(models())[::-1]
w = [auc[(n, 'With PageValues')] for n in names]
wo = [auc[(n, 'Without PageValues')] for n in names]
fig, ax = plt.subplots(figsize=(9, 4.2))
for i in range(len(names)):
    ax.plot([wo[i], w[i]], [i, i], color=GRID, linewidth=2, zorder=1)
ax.scatter(w, range(len(names)), s=80, color=BLUE, label='With PageValues', zorder=2, edgecolor='white', linewidth=2)
ax.scatter(wo, range(len(names)), s=80, color=ORANGE, label='Without PageValues', zorder=2, edgecolor='white', linewidth=2)
for i in range(len(names)):
    ax.annotate(f'{w[i]:.2f}', (w[i], i), xytext=(9, 0), textcoords='offset points', va='center', color=INK)
    ax.annotate(f'{wo[i]:.2f}', (wo[i], i), xytext=(-9, 0), textcoords='offset points', va='center', ha='right', color=INK)
ax.set_yticks(range(len(names)), names)
ax.set_xlim(0.6, 1.0)
ax.set_xlabel('Test AUC (0.5 = random guessing, 1.0 = perfect)')
ax.set_title('Model performance drops ~0.15 AUC without PageValues', loc='left', color=INK, fontsize=13)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.spines['left'].set_visible(False)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
plt.tight_layout()
plt.savefig('images/model_auc_comparison.png')
plt.close()

# ---------- Figure 3: top features of the best model without PageValues ----------
m, cols = rf_no_pv
r = permutation_importance(m, X_test[cols], y_test, scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(r.importances_mean, index=cols).sort_values().tail(8)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
ax.set_xlabel('Drop in test AUC when the feature is shuffled')
ax.set_title('Top features, random forest without PageValues', loc='left', color=INK, fontsize=13)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('images/top_features_rf.png')
plt.close()

print({k: round(v, 3) for k, v in auc.items()})
