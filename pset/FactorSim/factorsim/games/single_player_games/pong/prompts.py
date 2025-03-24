iterative_prompts = """Create a paddle character for the human player, represented as a rectangle, positioned on the left side of the screen. 
Allow the human player to control the paddle's vertical movement using the up and down arrow keys. The paddle has a little velocity added to it to allow smooth movements.
Implement a paddle character for the CPU opponent, also represented as a rectangle, positioned on the opposite side of the screen.
Introduce a ball that moves across the screen with a speed. The ball should bounce off the paddles and the top and bottom walls of the game window.
If the ball goes off the left or right side of the screen, reset its position to the center and its direction.
The CPU to control its paddle's vertical movement to autonomously track the ball.
Detect collisions between the ball and the paddles. When the ball collides with a paddle, make it bounce off in the opposite direction.
Implement a scoring system that rewards the human player for returning the ball back (i.e. making contact with the bal).
Display the current score at the top of the screen. Ensure the game has no predefined end, allowing for continuous play. 
Display a "Game Over!" message when one player reaches a score of 5 and provide an option for the player to restart the game after the "Game Over" screen is displayed.
"""