import xgboost as xgb
import torch
import os

'''CableInspectAD'''
train = torch.load(os.path.join(os.getcwd(), "ObjectTensors", f"cablesBRF_featurevectors_train.pt"), weights_only=True).to(torch.float32)
test = torch.load(os.path.join(os.getcwd(), "ObjectTensors", f"cablesBRF_featurevectors_test.pt"), weights_only=True).to(torch.float32)

labels_normal_train = torch.zeros((1692,))
labels_anomalies_train = torch.ones((781,))
labels_train = torch.cat([labels_normal_train, labels_anomalies_train], dim=0)

labels_normal_test = torch.zeros((467,))
labels_anomalies_test = torch.ones((195,))
labels_test = torch.cat([labels_normal_test, labels_anomalies_test], dim=0)


'''MVTEC'''
# train = torch.load(os.path.join(os.getcwd(), "ObjectTensors", f"MVTECBRF_featurevectors_train.pt"), weights_only=True).to(torch.float32)
# test = torch.load(os.path.join(os.getcwd(), "ObjectTensors", f"MVTECBRF_featurevectors_test.pt"), weights_only=True).to(torch.float32)

# # Load the indexes for the images which are anomalous. 0 (normal) or 1 (anomalous)
# anomalytest_indexes = torch.load(os.path.join(os.getcwd(), "ObjectTensors", f"MVTEC3Dtest_anomaliesindex.pt"), weights_only=True).to(torch.float32)

# anomalytest_indexes = anomalytest_indexes.to(torch.bool)
# normal = test[~anomalytest_indexes]
# anomalies = test[anomalytest_indexes]

## Get a random subset of anomalies, 10%
# num_samples = int(0.1 * len(anomalies))   # → 94

# index_to_get = int(len(anomalies) / num_samples)

# keep = torch.ones(len(anomalies), dtype=torch.bool)
# for i, _ in enumerate(keep):
#     if i % index_to_get == 0:
#         keep[i] = False
        
# for_train = anomalies[~keep]
# for_test = anomalies[keep]

# train = torch.cat([train, for_train], dim=0)
# labels_normal_train = torch.zeros((2656,))
# labels_anomalies_train = torch.ones((95,))
# labels_train = torch.cat([labels_normal_train, labels_anomalies_train], dim=0)

# test = torch.cat([normal, for_test], dim=0)
# labels_normal_test = torch.zeros((249,))
# labels_anomalies_test = torch.ones((853,))
# labels_test = torch.cat([labels_normal_test, labels_anomalies_test], dim=0)

"""Training Boosted Random Forest"""
params = {"n_estimators": 50, 
          "num_parallel_tree": 200, 
          "learning_rate": 0.3, 
          "max_depth": 5, 
          "device": 'cuda', 
          "objective": "binary:logistic", 
          "colsample_bytree": 0.8,
          "colsample_bynode": 0.6,
          "subsample": 0.6,
          "reg_alpha": 5.0, 
          "eval_metric": "auc",
          "min_child_weight": 10,
          "early_stopping_rounds": 200
}

model = xgb.XGBClassifier(**params, n_jobs=-1)

model.fit(
    train, labels_train,
    eval_set=[(test, labels_test)],
    verbose=True
)

model.save_model(os.path.join(os.getcwd(), "ObjectTensors", f"MVTECBRF_model.pt"))