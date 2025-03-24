iterative_prompts="""
Write code for a 2d game viewed from the side where the character can walk and jump. Let the character move left or right using the 'a' and 'd' keys. Let the character jump using the 'w' key.
Create a level by arranging 5 stationary platforms above the ground. Make sure the character's jump can reach the platform height.
Let the character stand on the ground or platforms but fall otherwise. Start the player on the ground.
Add a princess character that the character must rescue by reaching her position. Place the princess on one of the platforms.
Implement fireballs that fall from random places from the top of the screen. Do not let the fireballs move through the platforms. These fireballs serve as obstacles that the player must avoid.
Touching a fireball should deduct 25 points from the player's score and cause them to lose a life. The game ends if the player loses fifteen lives.
Scatter ten coins randomly around the game window that the player can collect for 5 points each.
Award the player 50 points for rescuing the princess. Move the princess to a random platform when the player rescues her.
Display the current score and remaining lives in the game window.
"""