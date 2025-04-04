class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1, estimated_error=1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimated_error = estimated_error
        self.posteri_estimate = 0
        self.posteri_error_estimate = 1

    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_noise

        # Measurement update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_noise)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate
