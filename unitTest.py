import unittest
from fastapi.testclient import TestClient
from main import app,load_Model

class TestBackend(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_model_prediction(self):
        # Provide sample input features
        data = {"sepalLength": 5.1, "sepalWidth": 3.5, "petalLength": 1.4, "petalWidth": 0.2}
        response = self.client.post("/", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Prediction", response.json())

    def test_valid_input(self):
        # Provide valid input data
        data = {"sepalLength": 5.1, "sepalWidth": 3.5, "petalLength": 1.4, "petalWidth": 0.2}
        response = self.client.post("/", json=data)
        self.assertEqual(response.status_code, 200)

    def test_invalid_input_handling(self):
        # Provide invalid input data
        data = {"sepalLength": "invalid", "sepalWidth": 3.5, "petalLength": 1.4, "petalWidth": 0.2}
        response = self.client.post("/", json=data)
        self.assertNotEqual(response.status_code, 200)

    def test_model_loading(self):
        # Check if the model object exists
        model = load_Model()
        self.assertIsNotNone(model)

    def test_endpoint_availability(self):
        # Test availability of the endpoint
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()