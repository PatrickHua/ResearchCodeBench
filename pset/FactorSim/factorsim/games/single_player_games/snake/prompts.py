iterative_prompts = """
Create a snake character represented as a series of connected pixels or blocks. Initially, the snake should be a single block (i.e. the head) that moves in a specific direction within the game window.
Allow the player to control the snake's movement using arrow keys. The snake should be able to turn left or right, but it should not be able to move directly backward. Eg. if its moving downwards it cannot move up.
The movement of the snake should be continuous in the current direction until the player provides new input. Ensure that the snake moves one grid unit at a time.
Implement a basic food system where one food item appears randomly on the screen.
When the snake consumes the food by moving over or colliding with it, the snake's length increases, and the player earns points. It recieves a positive reward, +1, for each food the head comes in contact with. While getting -1 for each terminal state it reaches.
If the head of the snake comes in contact with any of the walls or its own body, the game should end.
Incorporate a scoring system, displaying the current score on the screen during gameplay. The score should increase each time the snake consumes food.
Ensure that the game has no predefined end, allowing the snake to continue growing and the difficulty to increase over time. New food items should appear after the snake consumes one.
Provide an option for the player to restart the game after it ends. Display a "Restart" option on the game over screen to allow the player to play again.
"""