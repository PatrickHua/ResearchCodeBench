import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "factorsim"))
from factorized_pomdp import Function, StateVariable, GameRep

class TestFactorizedPOMDP(unittest.TestCase):
    def test_core_classes(self):
        """Test the core Function class functionality"""
        # Test function initialization
        func = Function(
            name="test_func",
            description="Test function",
            implementation="def test_func(state_manager):\n    pass",
            relevant_state_names=["test_state"]
        )
        
        # Test properties
        self.assertEqual(func.name, "test_func")
        self.assertEqual(func.description, "Test function")
        self.assertEqual(func.relevant_state_names, ["test_state"])
        
        # Test is_relevant method
        test_states = [StateVariable("test_state", "value", "str", "test")]
        self.assertTrue(func.is_relevant(test_states))
        
        # Test sanity_check method
        self.assertTrue(func.sanity_check())

    def test_state_management(self):
        """Test the StateVariable class functionality"""
        # Test initialization
        state = StateVariable(
            name="test_state",
            value="test_value",
            variable_type="str",
            description="Test state",
            dont_clean=True
        )
        
        # Test properties
        self.assertEqual(state.name, "test_state")
        self.assertEqual(state.variable_type, "str")
        self.assertEqual(state.description, "Test state")
        self.assertTrue(state.dont_clean)
        
        # Test value property
        self.assertEqual(state.value, '"""test_value"""')
        
        # Test string representation
        str_rep = str(state)
        self.assertIn("Test state", str_rep)
        self.assertIn("self.test_state", str_rep)

    def test_game_representation(self):
        """Test the GameRep class initialization"""
        # Test initialization with default values
        game = GameRep()
        
        # Test properties
        self.assertEqual(game.num_tokens, 0)
        self.assertEqual(game.MAX_RETRIES, 10)
        self.assertEqual(game.debug_mode, False)
        self.assertEqual(game.query_idx, 0)
        self.assertEqual(game.num_api_calls, 0)
        
        # Test collections
        self.assertIsInstance(game.queries, list)
        self.assertIsInstance(game.states, list)
        self.assertIsInstance(game.input_logics, list)
        self.assertIsInstance(game.logics, list)
        self.assertIsInstance(game.renders, list)

    def test_core_game_logic(self):
        """Test the core game logic methods"""
        # Create a minimal game instance for testing
        game = GameRep(debug_mode=True)
        
        # Test query processing structure
        query = "test query"
        game.queries.append({"query": query})
        
        # Test state management
        test_state = StateVariable("test", "value", "str", "test")
        game.states.append(test_state)
        
        # Test function management
        test_func = Function("test", "test", "def test(state_manager):\n    pass")
        game.input_logics.append(test_func)
        
        # Verify collections
        self.assertEqual(len(game.queries), 1)
        self.assertEqual(len(game.states), 4)  # 3 default + 1 test
        self.assertEqual(len(game.input_logics), 1)

    def test_code_generation(self):
        """Test the code generation methods"""
        # Create a minimal game instance for testing
        game = GameRep()
        
        # Add test state
        test_state = StateVariable("test", "value", "str", "test")
        game.states.append(test_state)
        
        # Test state manager code generation
        state_manager_code = game.get_state_manager_code()
        self.assertIn("class StateManager:", state_manager_code)
        self.assertIn("def __init__(self):", state_manager_code)
        
        # Test function definition generation
        test_func = Function("test", "test", "def test(state_manager):\n    pass")
        function_def = game.get_function_def([test_func])
        self.assertIn("def test(state_manager):", function_def)
        self.assertIn("test", function_def)

if __name__ == '__main__':
    unittest.main()


