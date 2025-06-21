from surprise import Dataset, SVD, KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load data
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train models
svd = SVD()
svd.fit(trainset)

knn = KNNWithMeans()
knn.fit(trainset)

# Predict
pred_svd = svd.test(testset)
pred_knn = knn.test(testset)

# Ensemble average predictions
ensemble_predictions = []
for p1, p2 in zip(pred_svd, pred_knn):
    avg_est = (p1.est + p2.est) / 2
    ensemble_predictions.append(
        type(p1)(p1.uid, p1.iid, p1.r_ui, avg_est, p1.details)
    )

# Evaluate
rmse = accuracy.rmse(ensemble_predictions)
mae = accuracy.mae(ensemble_predictions)

print(f"Ensemble RMSE: {rmse:.4f}")
print(f"Ensemble MAE: {mae:.4f}")