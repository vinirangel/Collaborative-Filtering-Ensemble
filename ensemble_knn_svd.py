from surprise import Dataset, Reader
from surprise import SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
import numpy as np

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

# Train SVD model
svd = SVD()
svd.fit(trainset)

# Train KNN model
sim_options = {
    'name': 'cosine',
    'user_based': False  # item-based
}
knn = KNNBasic(sim_options=sim_options)
knn.fit(trainset)

# Get predictions from both models
preds_svd = svd.test(testset)
preds_knn = knn.test(testset)

# Create a dictionary of predictions for easy lookup
preds_svd_dict = {(pred.uid, pred.iid): pred.est for pred in preds_svd}
preds_knn_dict = {(pred.uid, pred.iid): pred.est for pred in preds_knn}

# Ensemble predictions (average)
ensemble_preds = []
for (uid, iid, true_r) in testset:
    est_svd = preds_svd_dict[(uid, iid)]
    est_knn = preds_knn_dict[(uid, iid)]
    ensemble_est = (est_svd + est_knn) / 2
    ensemble_preds.append((uid, iid, true_r, ensemble_est))


# Calculate RMSE and MAE manually
true_ratings = [true_r for (_, _, true_r, _) in ensemble_preds]
ensemble_ratings = [est for (_, _, _, est) in ensemble_preds]

rmse_val = np.sqrt(np.mean([(true - pred) ** 2 for true, pred in zip(true_ratings, ensemble_ratings)]))
mae_val = np.mean([abs(true - pred) for true, pred in zip(true_ratings, ensemble_ratings)])

print(f"Ensemble RMSE: {rmse_val:.4f}")
print(f"Ensemble MAE:  {mae_val:.4f}")
