iterative_prompts = """
Create a paddle character, represented as a horizontal rectangle, that is controlled by the player using the left and right arrow keys. The paddle moves only along the horizontal axis at the bottom of the game window.
Design a ball character as a small circle that initially rests on the paddle. When the game starts, allow the ball to move in a random direction upon a player's mouse click or key press.
Implement  gravity, causing the ball to fall downward. If the ball hits the paddle, ensure it bounces off at an angle dependent on the point of contact.
Introduce bricks scattered across the top of the game window, forming a wall. Each brick should be breakable, and when hit by the ball, it should disappear. 
Implement collision detection to determine when the ball hits a brick, the paddle, or the walls of the game window. When the ball hits a brick, increase the player's score.
Create a scoring system where the player earns points for each brick destroyed. Display the current score at the top-left corner of the screen during gameplay.
Ensure that the game has no predefined end and that new levels or rows of bricks continue to appear, maintaining consistent difficulty as the game progresses.
Ensure the game ends if the ball touches the bottom of the game window. When the game ends, display a "Game Over!" message.
Allow the player to restart the game after it ends by providing a "Restart" option on the game over screen.
"""