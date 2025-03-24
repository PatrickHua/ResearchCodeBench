iterative_prompts = """
Create a catcher character, represented as a rectangle, positioned at the bottom and the middle of the screen.
Allow the player to control the catcher's horizontal movement using the left and right arrow keys on the keyboard.
There should always be exactly one ball on the screen at all times. The ball should be visually distinct and easily recognizable.
Make the ball move downwards at a steady pace towards the catcher. The speed can be constant or increase gradually as the game progresses.
Detect collisions between the catcher and the ball. When the catcher catches a ball, increment the player's score, spawn a new ball, and display this score in the top-left corner of the screen.
Give the player a 3 lives. Each time a ball is missed by the catcher and reaches the bottom of the screen, decrease the player's life count by one.
End the game when the player's lives reach zero. Display a "Game Over!" message and temporarily halt gameplay but dont terminate the game.
Provide an option for the player to click the screen to restart the game after the "Game Over" screen is displayed.
Continuously generate new balls after each catch or miss, ensuring endless gameplay. Optionally, increase the game's difficulty gradually by speeding up the ball's fall or reducing the size of the catcher as the player's score increases.
"""