#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
#load("1_explore.rdata")
import dill
dill.load_session("1_explore.pkl")

# Load libraries and functions
#source("./code/0_init.R")
exec(open("./code/0_init.py").read())



# Initialize parallel processing
#closeAllConnections() #reset
#Sys.getenv("NUMBER_OF_PROCESSORS") 
#cl = makeCluster(4)
#registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster



# Undersample data --------------------------------------------------------------------------------------------

# Just take data from train fold
#summary(df[df$fold == "train", "target"])
df.loc[df.fold == "train","target"].describe()
#df.train = c()
#for (i in 1:2) {
#  i.samp = which(df$fold == "train" & df$target == levels(df$target)[i])
#  set.seed(i*123)
#  df.train = bind_rows(df.train, df[sample(i.samp, min(1000, length(i.samp))),]) #take all but 1000 at most
#}
df_train = df.loc[df.fold == "train",].groupby("target").apply(lambda x: x.sample(min(1000, x.shape[0])))
#summary(df.train$target)
df_train.target.value_counts()
# Define prior base probabilities (needed to correctly switch probabilities of undersampled data)
#b_all = mean(df %>% filter(fold == "train") %>% .$target_num)
#b_sample = mean(df.train$target_num)
b_all = df.loc[df.fold == "train","target_num"].mean()
b_sample = df_train.target_num.mean()

# Define test data
#df.test = df %>% filter(fold == "test")
df_test = df[df.fold == "test"]



#######################################################################################################################-
#|||| Test an algorithm (and determine parameter grid) ||||----
#######################################################################################################################-

## Validation information
metric = "roc_auc"

# Possible controls
#set.seed(999)
#ctrl_cv = trainControl(method = "repeatedcv", number = 4, repeats = 1, returnResamp = "final",
#                       summaryFunction = mysummary_class, classProbs = TRUE) #NOT USED
#l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
#ctrl_index_fff = trainControl(method = "cv", number = 1, index = l.index, returnResamp = "final",
#                              summaryFunction = mysummary_class, classProbs = TRUE, 
#                              indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
#fit = train(df.train[predictors], df.train$target, 
#            trControl = ctrl_index_fff, metric = metric, 
#            method = "rf", 
#            tuneGrid = expand.grid(mtry = seq(1,length(predictors),3)), 
#            ntree = 200) #use the Dots (...) for explicitly specifiying randomForest parameter
fit = GridSearchCV(RandomForestClassifier(warm_start = True), 
                  [{"n_estimators": [10,20,50,100,200], 
                    "max_features": [x for x in range(1, len(predictors), 3)]}], 
                  cv = ShuffleSplit(1, 0.2, random_state = 999), 
                  refit = False, 
                  scoring = metric, 
                  n_jobs = 4
                  #use_warm_start = ["n_estimators"]}
      ).fit(pd.get_dummies(df_train[predictors]), df_train.target_num)
# Plot
fit.best_params_
df_fitres = pd.DataFrame(fit.cv_results_["params"])
df_fitres["mean_test_score"] = fit.cv_results_["mean_test_score"]
df_fitres.pivot_table("mean_test_score", index = "n_estimators", columns = "max_features").plot.line()


#fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
#            trControl = ctrl_index_fff, metric = metric, 
#            method = "glmnet", 
#            tuneGrid = expand.grid(alpha = c(0,0.2,0.4,0.6,0.8,1), lambda = 2^(seq(-3, -10, -1))),
#            #tuneLength = 20, 
#            preProc = c("center","scale")) 
#plot(fit, ylim = c(0.49,0.51))
# -> keep alpha=1 to have a full Lasso
fit = GridSearchCV(ElasticNet(normalize = True, warm_start = True), 
                  [{"alpha": [2**x for x in range(-6,-15,-1)], 
                    "l1_ratio": [0,0.2,0.4,0.6,0.8,1]}], 
                  cv = ShuffleSplit(1, 0.2, random_state = 999), 
                  refit = False, 
                  scoring = metric, 
                  n_jobs = 4
      ).fit(pd.get_dummies(df_train[predictors]), df_train.target_num)
fit.best_params_
df_fitres = pd.DataFrame(fit.cv_results_["params"])
df_fitres["mean_test_score"] = fit.cv_results_["mean_test_score"]
df_fitres.pivot_table("mean_test_score", index = "alpha", columns = "l1_ratio").plot()
# -> keep l1_ratio=1 to have a full Lasso    


#fit = train(as.data.frame(df.train[predictors]), df.train$target, 
#            trControl = ctrl_index_fff, metric = metric, 
#            method = "gbm", 
#            tuneGrid = expand.grid(n.trees = seq(100,1100,100), interaction.depth = c(3,6,9), 
#                                   shrinkage = c(0.1,0.01), n.minobsinnode = c(5,10)), 
#            verbose = FALSE)
#plot(fit)
fit = GridSearchCV(GradientBoostingClassifier(warm_start = True), 
                  [{"n_estimators": [x for x in range(100,1100,200)], 
                    "max_depth": [3,6,9],
                    "learning_rate": [0.01,0.02],
                    "min_samples_leaf": [5,10]}], 
                  cv = ShuffleSplit(1, 0.2, random_state = 999), 
                  refit = False, 
                  scoring = metric, 
                  n_jobs = 1
      ).fit(pd.get_dummies(df_train[predictors]), df_train.target_num)
fit.best_params_
df_fitres = pd.DataFrame(fit.cv_results_["params"])
df_fitres["mean_test_score"] = fit.cv_results_["mean_test_score"]

# -> keep to the recommended values: interaction.depth = 6, shrinkage = 0.01, n.minobsinnode = 10

#fit = train(formula, data = df.train[c("target",predictors)],
#            trControl = ctrl_index_fff, metric = metric, 
#            method = "xgbTree", 
#            tuneGrid = expand.grid(nrounds = seq(100,1100,200), max_depth = c(3,6,9), 
#                                   eta = c(0.1,0.01), gamma = 0, colsample_bytree = c(0.5,0.7), 
#                                   min_child_weight = c(5,10), subsample = c(0.5,0.7)))
#plot(fit)
fit = GridSearchCV(xgb.XGBClassifier(n_jobs = 4), 
                  [{"n_estimators": [x for x in range(100,1100,200)], 
                    "max_depth": [3,6,9],
                    "learning_rate": [0.01,0.02],
                    "min_child_weight": [5,10]}], 
                  cv = ShuffleSplit(1, 0.2, random_state = 999), 
                  refit = False, 
                  scoring = metric, 
                  n_jobs = 1
      ).fit(pd.get_dummies(df_train[predictors]), df_train.target_num)
fit.best_params_
df_fitres = pd.DataFrame(fit.cv_results_["params"])
df_fitres["mean_test_score"] = fit.cv_results_["mean_test_score"]

fit = GridSearchCV(lgbm.LGBMClassifier(n_jobs = 4), 
                  [{"n_estimators": [x for x in range(100,1100,200)], 
                    "max_depth": [3,6,9],
                    "learning_rate": [0.01,0.02],
                    "min_child_weight": [5,10]}], 
                  cv = ShuffleSplit(1, 0.2, random_state = 999), 
                  refit = False, 
                  scoring = metric, 
                  n_jobs = 1
      ).fit(pd.get_dummies(df_train[predictors]), df_train.target_num)
## -> max_depth = 3, shrinkage = 0.01, colsample_bytree = subsample = 0.7, n.minobsinnode = 5
#

#fit = train(formula, data = as.data.frame(df.train[c("target",predictors)]),
#            trControl = ctrl_index_fff, metric = metric, 
#            method = lgbm, 
#            # tuneGrid = expand.grid(num_rounds = c(50,100,200), num_leaves = 10,  
#            #                        learning_rate = .1, feature_fraction = .7,  
#            #                        min_data_in_leaf = 5, bagging_fraction = .7),
#            tuneGrid = expand.grid(num_rounds = seq(100,1100,100), num_leaves = c(10,20),
#                                   learning_rate = c(0.1,0.01), feature_fraction = c(0.5,0.7),
#                                   min_data_in_leaf = c(5,10), bagging_fraction = c(0.5,0.7)),
#            max_depth = 3,
#            verbose = 0) 
#plot(fit)
## -> numLeaves = 20, learning_rate = 0.01, feature_fraction = example_fraction = 0.7, minSplit = 10


# Score 
clf = RandomForestClassifier(n_estimators = 50, max_features = 3)
fit = clf.fit(pd.get_dummies(df_train[predictors]), df_train.target_num)
pred = fit.predict_proba(pd.get_dummies(df_test[predictors]))[:,1]
pred.mean()
roc_auc_score(df_test.target_num, pred)
cross_val_score(fit, pd.get_dummies(df_train[predictors]), df_train.target_num, 
                cv = 5, scoring = metric, n_jobs = 5)




# ## Special plotting
# fit
# plot(fit, ylim = c(0.85,0.93))
# varImp(fit) 
# # unique(fit$results$lambda)
# 
# skip = function() {
#   
#   y = metric
#   
#   # xgboost
#   x = "nrounds"; color = "as.factor(max_depth)"; linetype = "as.factor(eta)"; 
#   shape = "as.factor(min_child_weight)"; facet = "min_child_weight ~ subsample + colsample_bytree"
#   
#   # ms_boosttree
#   x = "numTrees"; color = "as.factor(numLeaves)"; linetype = "as.factor(learningRate)";
#   shape = "as.factor(minSplit)"; facet = "minSplit ~ exampleFraction + featureFraction"
#   
#   # lgbm
#   x = "num_rounds"; color = "as.factor(num_leaves)"; linetype = "as.factor(learning_rate)";  
#   shape = "as.factor(min_data_in_leaf)"; facet = "min_data_in_leaf ~ bagging_fraction + feature_fraction"
#   
#   # Plot tuning result with ggplot
#   fit$results %>% 
#     ggplot(aes_string(x = x, y = y, colour = color)) +
#     geom_line(aes_string(linetype = linetype, dummy = shape)) +
#     geom_point(aes_string(shape = shape)) +
#     #geom_errorbar(mapping = aes_string(ymin = "auc - aucSD", ymax = "auc + aucSD", linetype = linetype, width = 100)) +
#     facet_grid(as.formula(paste0("~",facet)), labeller = label_both) 
#   
# }

df_fitres["min_child_weight__learning_rate"] = (df_fitres.min_child_weight.astype("str") + "_" + 
                                                df_fitres.learning_rate.astype("str"))
sns.factorplot(x, "mean_test_score", "min_child_weight__learning_rate", df_fitres, col = "max_depth",
               palette = ["C0","C0","C1","C1"], markers = ["o","x","o","x"], linestyles = ["-",":","-",":"],
               legend_out = False)

from plotnine import *
factors = ["min_child_weight","learning_rate","max_depth"]
df_fitres[factors] = df_fitres[factors].astype("str")
#  x = "nrounds"; color = "as.factor(max_depth)"; linetype = "as.factor(eta)"; 
#   shape = "as.factor(min_child_weight)"; facet = "min_child_weight ~ subsample + colsample_bytree"
x = "n_estimators"; color = "min_child_weight"; linetype = "learning_rate"; 
shape = "learning_rate"; column = "max_depth"
(ggplot(df_fitres, aes(x = x, y = "mean_test_score", colour = color))
     + geom_line(aes(linetype = linetype))
     + geom_point(aes(shape = shape))
     + facet_grid(". ~ max_depth"))




# #######################################################################################################################-
# #|||| Compare algorithms ||||----
# #######################################################################################################################-
# 
# # Data to compare on
# df.comp = df.train
# 
# 
# 
# #---- Simulation function ---------------------------------------------------------------------------------------
# 
# perfcomp = function(method, nsim = 5) { 
#   
#   result = NULL
#   
#   for (sim in 1:nsim) {
#     
#     # Hold out a k*100% set
#     set.seed(sim*999)
#     k = 0.2
#     i.holdout = sample(1:nrow(df.comp), floor(k*nrow(df.comp)))
#     df.holdout = df.comp[i.holdout,]
#     df.train = df.comp[-i.holdout,]    
#     
#     # Control for train
#     set.seed(999)
#     l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
#     ctrl_index = trainControl(method = "cv", number = 1, index = l.index, returnResamp = "final",
#                               summaryFunction = mysummary_class, classProbs = TRUE)
#     
#     
#     ## Fit data
#     fit = NULL
#     
#     if (method == "glmnet") {  
#       fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
#                   trControl = ctrl_index, metric = metric, 
#                   method = "glmnet", 
#                   tuneGrid = expand.grid(alpha = 1, lambda = 2^(seq(-3, -10, -1))),
#                   #tuneLength = 20, 
#                   preProc = c("center","scale")) 
#     }     
#     
#     if (method == "glm") {      
#       fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
#                   trControl = ctrl_index, metric = metric, 
#                   method = "glm", 
#                   tuneLength = 1,
#                   preProc = c("center","scale"))
#     }
#     
#     if (method == "rpart") {      
#       fit = train( df.train[predictors], df.train$target, 
#                    trControl = ctrl_index, metric = metric, 
#                    method = "rpart",
#                    tuneGrid = expand.grid(cp = 2^(seq(-20, -2, 2))) )
#     }
#     
#     if (method == "rf") {      
#       fit = train(df.train[predictors], df.train$target, 
#                   trControl = ctrl_index, metric = metric, 
#                   method = "rf", 
#                   tuneGrid = expand.grid(mtry = c(4,5,6,7)), 
#                   ntree = 500) 
#     }
#     
#     if (method == "gbm") { 
#       fit = train(as.data.frame(df.train[predictors]), df.train$target, 
#                   trControl = ctrl_index, metric = metric, 
#                   method = "gbm", 
#                   tuneGrid = expand.grid(n.trees = seq(100,1100,100), interaction.depth = 6, 
#                                          shrinkage = 0.01, n.minobsinnode = 10), 
#                   verbose = FALSE)
#     }
#     
#     if (method == "xgbTree") { 
#       fit = train(formula, data = df.train[c("target",predictors)],
#                   trControl = ctrl_index, metric = metric, 
#                   method = "xgbTree", 
#                   tuneGrid = expand.grid(nrounds = seq(100,1100,200), max_depth = 3, 
#                                          eta = 0.01, gamma = 0, colsample_bytree = 0.7, 
#                                          min_child_weight = 5, subsample = 0.7))
#     }
#     
#     if (method == "ms_boosttree") { 
#       fit = train(df.train[,predictors], df.train$target, 
#                   trControl = ctrl_index, metric = metric, 
#                   method = ms_boosttree, 
#                   tuneGrid = expand.grid(numTrees = seq(400,1000,200), numLeaves = 10,  
#                                          learningRate = 0.01, featureFraction = 0.7,  
#                                          minSplit = 10, exampleFraction = 0.7),
#                   verbose = 0) 
#     }
#     
#     if (method == "ms_forest") { 
#       fit = train(df.train[,predictors], df.train$target, 
#                   trControl = ctrl_index, metric = metric, 
#                   method = ms_forest, 
#                   tuneGrid = expand.grid(numTrees = c(100,300,500), splitFraction = 0.3),
#                   verbose = 0) 
#       plot(fit)
#     }
#     
#     
#     
#     ## Get metrics
#     
#     # Calculate holdout performance
#     if (method %in% c("glmnet","glm")) {
#       yhat_holdout = predict(fit, df.holdout[predictors_binned], type = "prob")[[2]] 
#     } else  {
#       yhat_holdout = predict(fit, df.holdout[predictors], type = "prob")[[2]] 
#     }
#     perf_holdout = mysummary_class(data.frame(y = df.holdout$target, yhat = yhat_holdout))
# 
#     # Put all together
#     result = rbind(result, data.frame(sim = sim, method = method, t(perf_holdout)))
#   }   
#   result
# }
# 
# 
# 
# 
# #---- Simulate --------------------------------------------------------------------------------------------
# 
# df.result = as.data.frame(c())
# nsim = 5
# df.result = bind_rows(df.result, perfcomp(method = "glmnet", nsim = nsim) )   
# df.result = bind_rows(df.result, perfcomp(method = "glm", nsim = nsim) )     
# df.result = bind_rows(df.result, perfcomp(method = "rpart", nsim = nsim))      
# df.result = bind_rows(df.result, perfcomp(method = "rf", nsim = nsim))        
# df.result = bind_rows(df.result, perfcomp(method = "gbm", nsim = nsim))       
# df.result = bind_rows(df.result, perfcomp(method = "xgbTree", nsim = nsim))       
# df.result = bind_rows(df.result, perfcomp(method = "ms_boosttree", nsim = nsim))       
# df.result = bind_rows(df.result, perfcomp(method = "ms_forest", nsim = nsim))       
# df.result$sim = as.factor(df.result$sim)
# df.result$method = factor(df.result$method, levels = unique(df.result$method))
# 
# =============================================================================

cvresults = cross_validate(
      GridSearchCV(RandomForestClassifier(warm_start = True), 
                  [{"n_estimators": [10,20,50,100,200], 
                    "max_features": [x for x in range(1, len(predictors), 3)]}], 
                  cv = ShuffleSplit(1, 0.2, random_state = 999), 
                  refit = True, 
                  scoring = metric, 
                  n_jobs = 1),
      pd.get_dummies(df_train[predictors]), df_train.target_num,
      cv = RepeatedKFold(5, 2, random_state = 42),
      n_jobs = 4,
      return_train_score = True)
cvresults




#---- Plot simulation --------------------------------------------------------------------------------------------

# =============================================================================
# p = ggplot(df.result, aes_string(x = "method", y = metric)) + 
#   geom_boxplot() + 
#   geom_point(aes(color = sim), shape = 15) +
#   geom_line(aes(color = sim, group = sim), linetype = 2) +
#   scale_x_discrete(limits = rev(levels(df.result$method))) +
#   coord_flip() +
#   labs(title = "Model Comparison") +
#   theme_bw() + theme(plot.title = element_text(hjust = 0.5))
# p  
# ggsave(paste0(plotloc, "model_comparison.pdf"), p, width = 12, height = 8)
# 
# =============================================================================



# =============================================================================
# #######################################################################################################################-
# #|||| Check number of trainings records needed for winner algorithm ||||----
# #######################################################################################################################-
# 
# skip = function() {
#   # For testing on smaller data
#   df.train = df.train[sample(1:nrow(df.train), 5000),]
#   df.test = df.test[sample(1:nrow(df.test), 5000),]
# }
# 
# 
# 
# 
# #---- Loop over training chunks --------------------------------------------------------------------------------------
# 
# chunks_pct = c(seq(1,10,1), seq(20,100,10))
# 
# df.obsneed = c()  
# df.obsneed = foreach(i = 1:length(chunks_pct), .combine = bind_rows, .packages = c("caret")) %do% #NO dopar for xgboost!
# { 
#   #i = 1
#   
#   ## Sample chunk
#   set.seed(chunks_pct[i])
#   i.train = sample(1:nrow(df.train), floor(chunks_pct[i]/100 * nrow(df.train)))
#   print(length(i.train))
#   
#   
#   
#   ## Fit on chunk
#   ctrl_none = trainControl(method = "none", classProbs = TRUE)
#   tmp = Sys.time()
#   fit = train(formula, data = df.train[i.train,c("target",predictors)],
#               trControl = ctrl_none, metric = metric, 
#               method = "xgbTree", 
#               tuneGrid = expand.grid(nrounds = 500, max_depth = 3, 
#                                      eta = 0.01, gamma = 0, colsample_bytree = 0.7, 
#                                      min_child_weight = 5, subsample = 0.7))
#   
#   print(Sys.time() - tmp)
#   
#   
#   
#   ## Score (needs rescale to prior probs)
#   # Train data 
#   y_train = df.train[i.train,]$target
#   (b_sample = (summary(y_train) / length(y_train))[[2]]) #new b_sample
#   yhat_train = prob_samp2full(predict(fit, df.train[i.train,predictors], type = "prob")[[2]],
#                               b_sample, b_all)
#   
#   # Test data 
#   y_test = df.test$target
#   l.split = split(1:nrow(df.test), (1:nrow(df.test)) %/% 50000)
#   yhat_test = foreach(j = 1:length(l.split), .combine = c) %do% {
#     # Scoring in chunks due to high memory consumption of xgboost
#     yhat = predict(fit, df.test[l.split[[j]],predictors], type = "prob")[[2]]
#     gc()
#     yhat
#   }
#   yhat_test = prob_samp2full(yhat_test, b_sample, b_all)
#   
#   # Bind together
#   res = rbind(cbind(data.frame("fold" = "train", "numtrainobs" = length(i.train)),
#                     t(mysummary_class(data.frame(y = y_train, yhat = yhat_train)))),
#               cbind(data.frame("fold" = "test", "numtrainobs" = length(i.train)),
#                     t(mysummary_class(data.frame(y = y_test, yhat = yhat_test)))))
# 
#   
#   
#   ## Garbage collection and output
#   gc()
#   res
# }
# #save(df.obsneed, file = "df.obsneed.RData")
# 
# 
# 
# 
# #---- Plot results --------------------------------------------------------------------------------------
# 
# p = ggplot(df.obsneed, aes_string("numtrainobs", metric, color = "fold")) +
#   geom_line() +
#   geom_point() +
#   scale_color_manual(values = c("#F8766D", "#00BFC4")) 
# p
# ggsave(paste0(plotloc,"learningCurve.pdf"), p, width = 8, height = 6)
# 
# =============================================================================



