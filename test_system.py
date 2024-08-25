from Quiz_1.system import ML_System_Classification
import unittest

class TestMLSystemClassification(unittest.TestCase):
    def test_classification_system(self):
        path = "/code/Python/Quiz_1/"  # Asegúrate de que la ruta sea correcta y accesible
        sistema = ML_System_Classification(path)
        resultado = sistema.ML_Flow_regression()
        
        # Verificar que el sistema se ejecutó correctamente
        self.assertTrue(resultado["success"], "Modelo ejecutado correctamente")
        
        # Verificar que el accuracy sea al menos del 70%
        self.assertGreaterEqual(resultado["accuracy"], 70, "The model accuracy should be above 0.7")

if __name__ == "__main__":
    unittest.main()
