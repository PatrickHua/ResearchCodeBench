# decorators.py
ALLOWED_CATEGORIES = {"Init", "Reset", "Movement", "Collision", "Mechanics"}

# Init (logic) - all initialization events, including generation of new assets that is not conditionally triggered
# Restart (logic) - all events triggered by a restart or a reset
# Movement (logic) - events triggered by any user-input triggered movements
# Collision (logic) - events triggered by collision between game components (including walls and the screen boundaries)
# Mechanics (logic) - events triggered by the game meeting a certain state/condition (if all of the above does not apply it's this)


def category(value):
    if value not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"Invalid category {value}. Allowed categories are {', '.join(ALLOWED_CATEGORIES)}"
        )

    def decorator(func):
        func.category = value
        return func

    return decorator


def print_category_of_test_methods(test_case_class):
    for attr_name in dir(test_case_class):
        attr_value = getattr(test_case_class, attr_name)
        if callable(attr_value) and hasattr(attr_value, "category"):
            print(f"Method {attr_name} category: {attr_value.category}")


if __name__ == "__main__":
    import sys
    import importlib
    from glob import glob

    for game_name in glob("*"):
        if game_name != "__pycache__" and "decorator" not in game_name:
            print(game_name)
            sys.path.append(game_name)
            import unit_test, game

            importlib.reload(game)
            importlib.reload(unit_test)
            from unit_test import Test

            print_category_of_test_methods(Test)
            sys.path.remove(game_name)
            print()
