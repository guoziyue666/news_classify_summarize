class NewsService:
    def __init__(self, predictor, summary_predictor):
        self.predictor = predictor
        self.summary_predictor = summary_predictor

    def predict(self, text):
        return self.predictor.predict(text), self.summary_predictor.predict(text)
