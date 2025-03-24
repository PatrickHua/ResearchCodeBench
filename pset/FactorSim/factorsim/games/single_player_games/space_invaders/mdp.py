import pygame
import sys


class StateManager:
    def __init__(self):
        # Initialize state variables directly
        self.screen_width = 800
        self.screen_height = 600
        self.player_position = {"x": 400, "y": 550}
        self.player_speed = 20
        self.alien_positions = []
        self.bullet_position = {"x": 0, "y": 0}
        self.bullet_speed = 15
        self.score = 0
        self.game_over = False

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()


def render(state_manager):
    state_manager.screen.fill((0, 0, 0))  # Fill the screen with black

    pygame.draw.rect(
        state_manager.screen,
        (255, 255, 255),
        (
            state_manager.player_position["x"],
            state_manager.player_position["y"],
            50,
            20,
        ),
    )

    # Render aliens
    for alien in state_manager.alien_positions:
        pygame.draw.rect(
            state_manager.screen,
            (0, 255, 0),
            (
                alien["x"],
                alien["y"],
                30,
                30,
            ),
        )

    # Render bullet
    if state_manager.bullet_position["y"] > 0:
        pygame.draw.rect(
            state_manager.screen,
            (0, 255, 255),
            (
                state_manager.bullet_position["x"],
                state_manager.bullet_position["y"],
                5,
                10,
            ),
        )

    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {state_manager.score}", True, (255, 255, 255))
    state_manager.screen.blit(score_text, (10, 10))

    pygame.display.flip()
    state_manager.clock.tick(60)


def player_movement(state_manager, event):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            state_manager.player_position["x"] -= state_manager.player_speed
        elif event.key == pygame.K_RIGHT:
            state_manager.player_position["x"] += state_manager.player_speed

    # Ensure the player stays within the screen boundaries
    state_manager.player_position["x"] = max(
        0,
        min(
            state_manager.screen_width - 50,
            state_manager.player_position["x"],
        ),
    )


def spawn_aliens(state_manager):
    if len(state_manager.alien_positions) == 0:
        for i in range(5):
            for j in range(5):
                new_alien = {
                    "x": 50 + i * 100,
                    "y": 50 + j * 50,
                    "direction": 1,  # 1 for right, -1 for left
                }
                state_manager.alien_positions.append(new_alien)

    # Determine if the entire group needs to change direction and descend
    change_direction = False
    for alien in state_manager.alien_positions:
        if alien["x"] <= 0 or alien["x"] >= state_manager.screen_width - 30:
            change_direction = True
            break

    if change_direction:
        for alien in state_manager.alien_positions:
            alien["direction"] *= -1
            alien["y"] += 20

    for alien in state_manager.alien_positions:
        alien["x"] += 0.5 * alien["direction"]  # Move side by side
        # alien["y"] += 0.5  # Descend slowly

    if all(
        alien["y"] > state_manager.screen_height
        for alien in state_manager.alien_positions
    ):
        # Create a new group of aliens
        state_manager.alien_positions = []
        for i in range(5):
            for j in range(5):
                new_alien = {
                    "x": 50 + i * 100,
                    "y": 50 + j * 50,
                    "direction": 1,
                }
                state_manager.alien_positions.append(new_alien)


def bullet_movement(state_manager, event):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_SPACE and state_manager.bullet_position["y"] == 0:
            state_manager.bullet_position["x"] = state_manager.player_position["x"] + 25
            state_manager.bullet_position["y"] = state_manager.player_position["y"]
    if state_manager.bullet_position["y"] > 0:
        state_manager.bullet_position["y"] -= state_manager.bullet_speed
    else:
        state_manager.bullet_position["y"] = 0

    for alien in state_manager.alien_positions:
        if (
            alien["x"] < state_manager.bullet_position["x"] < alien["x"] + 30
            and alien["y"] < state_manager.bullet_position["y"] < alien["y"] + 30
        ):
            state_manager.alien_positions.remove(alien)
            state_manager.bullet_position["y"] = 0
            state_manager.score += 1


def check_game_over(state_manager):

    for alien in state_manager.alien_positions:
        if alien["y"] >= state_manager.screen_height:
            state_manager.game_over = True
            break

    if state_manager.game_over:
        font = pygame.font.Font(None, 72)
        game_over_text = font.render("Game Over", True, (255, 255, 255))
        state_manager.screen.blit(game_over_text, (250, 250))
        pygame.display.flip()
        pygame.time.wait(2000)
        return False
    return True


class Game:
    def __init__(self):
        self.state_manager = StateManager()
        self.state_manager.screen = pygame.display.set_mode(
            (self.state_manager.screen_width, self.state_manager.screen_height)
        )

    def run(self, event):

        state_manager = self.state_manager

        if event.type == pygame.QUIT:
            return False
        player_movement(state_manager, event)

        spawn_aliens(state_manager)

        bullet_movement(state_manager, event)

        render(state_manager)

        return check_game_over(state_manager)


if __name__ == "__main__":
    game = Game()
    pygame.init()
    global event
    running = True
    while running:
        event = pygame.event.poll()
        running = game.run(event)
    pygame.quit()
