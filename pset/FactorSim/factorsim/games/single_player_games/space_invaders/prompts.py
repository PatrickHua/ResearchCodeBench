iterative_prompts = """
Create a spaceship character, visually represented as a simple rectangle, positioned at the bottom and center of the game window. The spaceship should be movable horizontally across the bottom of the screen.
Allow the player to control the spaceship's horizontal movement using the left and right arrow keys on the keyboard and shoot bullets using the spacebar.
Periodically spawn alien ships with random x-coordinates at the top of the screen. The alien ships should be in a swarm formation, with each ship spaced evenly apart from the others.
Ensure that only one swarm of alien ships is present on the screen at any given time.
Make the alien ships move sideways at a steady pace towards the spaceship. 
On reaching the edge of the screen, the alien ships should move down a fixed distance and then reverse direction.
When the spaceship successfully shoots down an alien ship, increment the player's score and display this score in the top-left corner of the screen.
If an alien ship reaches the bottom of the screen without being shot down, end the game and display a "Game Over!" message and temporarily halt gameplay.
Provide an option for the player to restart the game after the "Game Over" screen is displayed.
Continuously generate new alien ships after each successful shot or miss, ensuring endless gameplay.
"""