class RecommendationPipeline:
    def __init__(self, model=None, data_preprocessor=None):
        """
        Initialize the recommendation pipeline.
        :param model: Any ML model that implements fit and predict methods.
        :param data_preprocessor: Optional preprocessing function or object.
        """
        self.model = model
        self.data_preprocessor = data_preprocessor
        self.is_trained = False

    def train(self, train_data: Any, train_labels: Any, **kwargs):
        """
        Trains the recommendation model.
        :param train_data: The training features/data.
        :param train_labels: The training labels/targets.
        :param kwargs: Additional keyword arguments for model.fit.
        """
        if self.data_preprocessor:
            train_data = self.data_preprocessor.fit_transform(train_data)
        self.model.fit(train_data, train_labels, **kwargs)
        self.is_trained = True

    def evaluate(self, test_data: Any, test_labels: Any, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluates the model on test data.
        :param test_data: The test features/data.
        :param test_labels: The test labels/targets.
        :param metrics: A dictionary of metric functions {'name': func(y_true, y_pred)}
        :return: Dictionary of metric results.
        """
        if self.data_preprocessor:
            test_data = self.data_preprocessor.transform(test_data)
        predictions = self.model.predict(test_data)
        results = {}
        if metrics:
            for name, func in metrics.items():
                results[name] = func(test_labels, predictions)
        return results

    def serve(self, input_data: Any) -> Any:
        """
        Serves recommendations for new input data.
        :param input_data: The data for which to generate recommendations.
        :return: Model predictions/recommendations.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before serving recommendations.")
        if self.data_preprocessor:
            input_data = self.data_preprocessor.transform(input_data)
        return self.model.predict(input_data)
