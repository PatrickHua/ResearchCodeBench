iterative_prompts = """
Create a copter character represented as a large white square that remains fixed horizontally but can ascend and descend vertically within the game window.
Introduce gravity mechanics, causing the copter to continuously descend slowly. Enable the player to ascend when the player clicks the mouse, allowing it to momentarily counteract gravity and rise upwards.
Create obstacles in the shape of a cavern. Construct the cavern using a series of vertically aligned rectangular barriers positioned at both the bottom and the top of the screen. Ensure the adjacent obstacles are of similar length to maintain a consistently smooth "tunnel" effect.
Implement collision detection to detect when the copter collides with obstacles or the boundaries of the game window, triggering the end of the game upon collision.
Display a "Game Over!" message prominently when the game ends due to a collision, halting all movement within the game and prompting the player to restart.
Create a scoring system that rewards the player based on how far the copter travels through the maze without colliding with obstacles.
Show the current score in the top left area of the screen.
Ensure the game has no predefined end and that new obstacles continue to generate, maintaining consistent difficulty as the game progresses
Allow the player to start a new game after a collision.
"""