import unittest
import json


class JsonTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_results = []

    def addSuccess(self, test):
        super().addSuccess(test)
        test_name = test._testMethodName
        test_number = getattr(test, "test_number", None)
        self.test_results.append(
            {"function_name": test_name, "test_number": test_number, "status": "OK"}
        )

    def addError(self, test, err):
        super().addError(test, err)
        test_name = test._testMethodName
        test_number = getattr(test, "test_number", None)
        self.test_results.append(
            {
                "function_name": test_name,
                "test_number": test_number,
                "message": str(err),
                "status": "ERROR",
            }
        )

    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_name = test._testMethodName
        test_number = getattr(test, "test_number", None)
        self.test_results.append(
            {
                "function_name": test_name,
                "test_number": test_number,
                "message": str(err),
                "status": "FAIL",
            }
        )


class JsonTestRunner(unittest.TextTestRunner):
    def __init__(self, game_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_name = game_name
        self.resultclass = JsonTestResult

    def run(self, test):
        result = super().run(test)
        with open(f"{self.game_name}_test_results.json", "w") as f:
            json.dump(result.test_results, f, indent=4)
        return result
