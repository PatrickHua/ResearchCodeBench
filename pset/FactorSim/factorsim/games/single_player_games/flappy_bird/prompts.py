iterative_prompts = """
Create a bird character, visually represented as a simple rectangle within the game window.
Introduce gravity, causing the bird to continuously fall slowly.
Allow the bird to 'jump' or accelerate upwards in response to a player's mouse click, temporarily overcoming gravity.
Periodically spawn pairs of vertical pipes moving from right to left across the screen. Each pair should have a gap for the bird to pass through, and their heights should vary randomly.
If the bird makes contact with the ground, pipes or goes above the top of the screen the game is over.
Implement the following scoring system: for each pipe it passes through it gains a positive reward of +1. Each time a terminal state is reached it receives a negative reward of -1.
When the game ends, display a "Game Over!" messagea and stop all the motion of the game.
Show the current score in the top-left corner of the screen during gameplay.
Ensure the game has no predefined end and that new pipes continue to generate, maintaining consistent difficulty as the game progresses.
"""