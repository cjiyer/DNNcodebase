# Example usage with scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

pipeline = RecommendationPipeline(
    model=RandomForestClassifier(),
    data_preprocessor=StandardScaler()
)

# Train
pipeline.train(train_data=X_train, train_labels=y_train)

# Evaluate
metrics = {'accuracy': accuracy_score}
results = pipeline.evaluate(test_data=X_test, test_labels=y_test, metrics=metrics)
print(results)

# Serve
recommendations = pipeline.serve(input_data=X_new)
print(recommendations)
