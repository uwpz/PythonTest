# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:01:38 2017

@author: Uwe
"""

# ######################################################################################################################
#  Libraries + Parallel Processing Start
# ######################################################################################################################

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
import dill
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
# from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_validate, RepeatedKFold, learning_curve
# from sklearn.metrics import roc_auc_score
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import ElasticNet
# import xgboost as xgb
# import lightgbm as lgbm

import os
os.getcwd()
# os.chdir("C:/My/Projekte/PythonTest")
# exec(open("./code/0_init.py").read())

sns.set(style="whitegrid")
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)


# ######################################################################################################################-
# Parameters ----
# ######################################################################################################################-

dataloc = "./data/"
plotloc = "./output/"


# ######################################################################################################################
# My Functions
# ######################################################################################################################

def setdiff(a, b):
    return [x for x in a if x not in set(b)]


def union(a, b):
    return a + [x for x in b if x not in set(a)]


def create_values_df(df_, topn):
    return pd.concat([df_[catname].value_counts()[:topn].reset_index().
                     rename(columns={"index": catname, catname: catname + "_c"})
                      for catname in df_.select_dtypes(["object"]).columns.values], axis=1)



# ######################################################################################################################
# ETL
# ######################################################################################################################

# Read data --------------------------------------------------------------------------------------------------------

df_orig = pd.read_csv(dataloc + "titanic.csv")
df_orig.describe()

"""
# Check some stuff
df_orig.describe()
df_orig.describe(include = ["object"])
df_values = create_values_df(df_orig, 10)
print(df_values)
df_orig["survived"].value_counts() / df_orig.shape[0]
"""

# "Save" original data
df = df_orig.copy()


# Read metadata (Project specific) -------------------------------------------------------------------------------------
df_meta = pd.read_excel(dataloc + "datamodel_titanic.xlsx", header=1)

# Check
print(setdiff(df.columns.values, df_meta["variable"]))
print(setdiff(df_meta.loc[df_meta["category"] == "orig", "variable"], df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready", "derive"])]


# Feature engineering -----------------------------------------------------------------------------------------
# df$deck = as.factor(str_sub(df$cabin, 1, 1))
df["deck"] = df["cabin"].str[:1]
df["familysize"] = df["sibsp"] + df["parch"] + 1
df["fare_pp"] = df.groupby("ticket")["fare"].transform("mean")
df[["deck","familysize","fare_pp"]].describe(include = "all")

# Check
print(setdiff(df_meta["variable"], df.columns.values))


# Define target and train/test-fold ----------------------------------------------------------------------------------

# Target
df["target"] = np.where(df.survived == 0, "N", "Y")
df["target_num"] = df.target.map({"N": 0, "Y": 1})
print(df[["target", "target_num"]].describe(include="all"))

# Train/Test fold: usually split by time
df["fold"] = "train"
df.loc[df.sample(frac=0.3, random_state=123).index, "fold"] = "test"
print(df.fold.value_counts())

# Define the id
df["id"] = df.index



# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# Define metric covariates -------------------------------------------------------------------------------------

metr = df_meta_sub.loc[df_meta_sub.type == "metr","variable"].values.tolist()
df[metr] = df[metr].apply(pd.to_numeric)
df[metr].describe()



# Create nominal variables for all metric variables (for linear models) before imputing -------------------------------

metr_binned = [x + "_BINNED_" for x in metr]
df[metr_binned] = df[metr].apply(lambda x: pd.qcut(x, 10).astype(object))

# Convert missings to own level ("(Missing)")
df[metr_binned] = df[metr_binned].fillna("(Missing)")
print(create_values_df(df[metr_binned], 11))

# Remove binned variables with just 1 bin
onebin = [var for var in metr_binned if df[var].nunique() == 1]
metr_binned = setdiff(metr_binned, onebin)




# Missings + Outliers + Skewness ---------------------------------------------------------------------------------

# Remove covariates with too many missings from metr
misspct = df[metr].isnull().mean().round(3) #missing percentage
misspct.sort_values(ascending=False) #view in descending order
remove = misspct[misspct > 0.95].index.values.tolist(); print(remove) #vars to remove
metr = setdiff(metr, remove) #adapt metadata
metr_binned = setdiff(metr_binned, [x + "_BINNED_" for x in remove]) #keep "binned" version in sync

# Check for outliers and skewness
fig, ax = plt.subplots(1,3)
for i in range(len(metr)):
    sns.distplot(df.loc[df.target == "Y", metr[i]].dropna(), color = "red", label = "Y", ax = ax[i])
    sns.distplot(df.loc[df.target == "N", metr[i]].dropna(), color = "blue", label = "N", ax = ax[i])
    ax[i].set_title(metr[i])
    ax[i].set_ylabel("density")
    ax[i].set_xlabel(metr[i] + "(NA: " + str(round(misspct[metr[i]] * 100, 1)) + "%)")
ax[0].legend(title = "Target", loc = "best")
fig.savefig(plotloc + "metr.pdf")
# plt.show()
# plt.subplot_tool()
# plt.close(fig)


# plotnine cannot plot several plots on one page
"""
i=1
nbins = 20
target_name = "target"
color = ["blue","red"]
levs_target = ["N","Y"]
p=(ggplot(data = df, mapping = aes(x = metr[i])) +
      geom_histogram(mapping = aes(y = "..density..", fill = target_name, color = target_name), 
                     stat = stat_bin(bins = nbins), position = "identity", alpha = 0.2) +
      geom_density(mapping = aes(color = target_name)) +
      scale_fill_manual(limits = levs_target[::-1], values = color[::-1], name = target_name) + 
      scale_color_manual(limits = levs_target[::-1], values = color[::-1], name = target_name) +
      labs(title = metr[i],
           x = metr[i] + "(NA: " + str(round(misspct[metr[i]] * 100, 1)) + "%)")
      )
p
plt.show()
plt.close()
"""

# Winsorize
df[metr] = df[metr].apply(lambda x: winsorize(x, (0.01,0.01))) #hint: one might want to plot again before deciding for log-trafo

# Log-Transform
tolog = ["fare"]
df[[x + "_LOG_" for x in tolog]] = df[tolog].apply(lambda x: np.where(x <= 0, np.log(x - np.min(x) + 1), np.log(x)))
metr = [x + "_LOG_" if x in tolog else x for x in metr] #adapt metadata (keep order)
# alternative: metr = list(map(lambda x: x + "_LOG_" if x in tolog else x, metr))



"""
# Final variable information --------------------------------------------------------------------------------------------

# Univariate variable importance
#varimp = filterVarImp(df[metr], df$target, nonpara = TRUE) %>% 
#  mutate(Y = round(ifelse(Y < 0.5, 1 - Y, Y),2)) %>% .$Y
#names(varimp) = metr
#varimp[order(varimp, decreasing = TRUE)]

# Plot 
#plots = get_plot_distr_metr_class(df, metr, missinfo = misspct, varimpinfo = varimp)
#ggsave(paste0(plotloc, "titanic_distr_metr_final.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), width = 18, height = 12)
"""



# Removing variables -------------------------------------------------------------------------------------------

# Remove Self features
remove = ["xxx","xxx"]
metr = setdiff(metr, remove)
metr_binned = setdiff(metr_binned, [x + "_BINNED" for x in remove]) # keep "binned" version in sync

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[metr].describe()
m_corr = abs(df[metr].corr(method = "spearman"))
fig = sns.heatmap(m_corr, annot=True, fmt=".2f", cmap = "Blues").get_figure()
fig.savefig(plotloc + "corr_metr.pdf")
#plt.show()
remove = ["xxx","xxx"]
metr = setdiff(metr, remove)
metr_binned = setdiff(metr_binned, [x + "_BINNED" for x in remove]) # keep "binned" version in sync


"""
# Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
df$fold_test = factor(ifelse(df$fold == "test", "Y", "N"))
(varimp_metr_fold = filterVarImp(df[metr], df$fold_test, nonpara = TRUE) %>% rowMeans() %>%
    .[order(., decreasing = TRUE)] %>% round(2))

# Plot: only variables with with highest importance
metr_toprint = names(varimp_metr_fold)[varimp_metr_fold >= cutoff_varimp]
plots = map(metr_toprint, ~ BoxCore::plot_distr(df[[.]], df$fold_test, ., "fold_test", varimps = varimp_metr_fold,
                                                colors = c("blue","red")))
ggsave(paste0(plotloc, TYPE, "_distr_metr_final_folddependency.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2),
       width = 18, height = 12)
"""



# Missing indicator and imputation (must be done at the end of all processing)-----------------------------------------

miss = np.array(metr)[df[metr].isnull().any().values].tolist()
# alternative: [x for x in metr if df[x].isnull().any()]
df[["MISS_" + x for x in miss]] = pd.DataFrame(np.where(df[miss].isnull(), "miss", "no_miss"))
df[["MISS_" + x for x in miss]].describe()

# Impute missings with randomly sampled value (or median, see below)
df[miss] = df[miss].fillna(df[miss].median())
# df[miss] = df[miss].apply(lambda x: np.where(x.isnull(), np.random.choice(x[x.notnull()], len(x.isnull())), x))
df[miss].isnull().sum()




#######################################################################################################################-
#|||| Nominal variables: Explore and adapt ||||----
#######################################################################################################################-

# Define categorical covariates -------------------------------------------------------------------------------------

# Nominal variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["nomi","ordi"]), "variable"].values.tolist()
df[cate] = df[cate].astype(object)
df[cate].describe()

# Merge categorical variable (keep order)
cate = union(cate,["MISS_" + x for x in miss])



# Handling factor values ----------------------------------------------------------------------------------------------

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 10
levinfo = df[cate].apply(lambda x: x.unique().size).sort_values(ascending = False)  # number of levels
toomany = levinfo[levinfo > topn_toomany].index.values.tolist(); print(toomany)
toomany = setdiff(toomany, ["xxx","xxx"])  # set exception for important variables
df[[x + "_ENCODED" for x in toomany]] = df[toomany]



## Convert categorical variables
# Convert "standard" features: map missings to own level
df[cate].fillna("(Missing)", inplace = True)
df[cate].describe()

# Convert toomany features: lump levels and map missings to own level
df[toomany] = df[toomany].apply(lambda x:
        x.replace(np.setdiff1d(x.value_counts()[topn_toomany:].index.values, "(Missing)"), "_OTHER_"))


"""
# Univariate variable importance
(varimp_cate = filterVarImp(df[cate], df$target, nonpara = TRUE) %>% rowMeans() %>%
    .[order(., decreasing = TRUE)] %>% round(2))
"""



# Check
fig, ax = plt.subplots(3,4, figsize=(18, 16))
for i in range(len(cate)):
    df_tmp = pd.DataFrame({"h": df.groupby(cate[i])["target_num"].mean(),
                           "w": df.groupby(cate[i]).size()}).reset_index()
    df_tmp["w"] = df_tmp["w"]/max(df_tmp["w"])
    axact = ax.flat[i]
    sns.barplot(df_tmp.h, df_tmp[cate[i]], orient = "h", color = "coral", ax = axact)
    axact.set_xlabel("Proportion Target = Y")
    axact.axvline(np.mean(df.target_num), ls = "dotted", color = "black")
    for bar,width in zip(axact.patches, df_tmp.w):
        bar.set_height(width)
plt.subplots_adjust(wspace = 1)
# plt.show()
fig.savefig(plotloc + "cate.pdf", dpi = 600)




# Removing variables ----------------------------------------------------------------------------------------------

# Remove leakage variables
cate = setdiff(cate, ["boat"]); toomany = setdiff(toomany, ["boat"])

"""
# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot = BoxCore::plot_corr(df[setdiff(cate, paste0("MISS_",miss))], "nomi", cutoff = cutoff_switch)
ggsave(paste0(plotloc,TYPE,"_corr_cate.pdf"), plot, width = 9, height = 9)
if (TYPE %in% c("REGR","MULTICLASS")) {
  plot = BoxCore::plot_corr(df[ paste0("MISS_",miss)], "nomi", cutoff = cutoff_switch)
  ggsave(paste0(plotloc,TYPE,"_corr_cate_MISS.pdf"), plot, width = 9, height = 9)
  cate = setdiff(cate, c("MISS_BsmtFin_SF_2","MISS_BsmtFin_SF_1","MISS_second_Flr_SF","MISS_Misc_Val_LOG_",
                         "MISS_Mas_Vnr_Area","MISS_Garage_Yr_Blt","MISS_Garage_Area","MISS_Total_Bsmt_SF"))
}
"""


"""
# Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance
(varimp_cate_fold = filterVarImp(df[cate], df$fold_test, nonpara = TRUE) %>% rowMeans() %>%
   .[order(., decreasing = TRUE)] %>% round(2))

# Plot (Hint: one might want to filter just on variable importance with highest importance)
cate_toprint = names(varimp_cate_fold)[varimp_cate_fold >= cutoff_varimp]
plots = map(cate_toprint, ~ BoxCore::plot_distr(df[[.]], df$fold_test, ., "fold_test", varimps = varimp_cate_fold,
                                                colors = c("blue","red")))
ggsave(paste0(plotloc,TYPE,"_distr_cate_folddependency.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3),
       width = 18, height = 12)
"""




#######################################################################################################################-
#|||| Prepare final data ||||----
#######################################################################################################################-

# Define final features ----------------------------------------------------------------------------------------

features_notree = metr + cate
features = metr + cate + [x + "_ENCODED" for x in toomany]
features_binned = metr_binned + setdiff(cate, ["MISS_" + x for x in miss])  # do not need indicators if binned variables

# Check
setdiff(features_notree, df.columns.values.tolist())
setdiff(features, df.columns.values.tolist())
setdiff(features_binned, df.columns.values.tolist())



# Save image ----------------------------------------------------------------------------------------------------------
del df_orig
dill.dump_session("1_explore.pkl")



