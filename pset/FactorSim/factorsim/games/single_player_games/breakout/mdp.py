import pygame
import sys
import random


class StateManager:
    def __init__(self):
        # Initialize state variables directly
        self.screen_width = 600
        self.screen_height = 800
        self.paddle = {"x": 250, "y": 700, "width": 100, "height": 10}
        self.grid_size = 20
        self.ball = {"x": 300, "y": 400, "radius": 10, "velocity": {"x": 5, "y": 5}}
        self.bricks = []
        self.score = 0
        self.game_over = False

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Initialize bricks after all states have been set
        self.bricks = [
            {
                "x": col * (self.grid_size * 2 + 5),  # Adjusted for spacing
                "y": row * (self.grid_size + 5),
                "width": self.grid_size * 2,
                "height": self.grid_size,
            }
            for row in range(5)
            for col in range(self.screen_width // (self.grid_size * 2))
        ]


def render(state_manager):
    state_manager.screen.fill((255, 255, 255))  # Fill the screen with white

    pygame.draw.rect(
        state_manager.screen,
        (0, 0, 255),
        (
            state_manager.paddle["x"],
            state_manager.paddle["y"],
            state_manager.paddle["width"],
            state_manager.paddle["height"],
        ),
    )

    pygame.draw.circle(
        state_manager.screen,
        (255, 0, 0),
        (
            state_manager.ball["x"],
            state_manager.ball["y"],
        ),
        state_manager.ball["radius"],
    )

    # Render bricks
    for brick in state_manager.bricks:
        pygame.draw.rect(
            state_manager.screen,
            (0, 255, 0),
            (
                brick["x"],
                brick["y"],
                brick["width"],
                brick["height"],
            ),
        )

    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {state_manager.score}", True, (0, 0, 0))
    state_manager.screen.blit(score_text, (10, 10))

    pygame.display.flip()
    state_manager.clock.tick(60)


def move_paddle_logic(state_manager, event):
    """Move the paddle based on the event (left or right key press)"""
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT and state_manager.paddle["x"] > 0:
            state_manager.paddle["x"] -= 40
        elif (
            event.key == pygame.K_RIGHT
            and state_manager.paddle["x"]
            < state_manager.screen_width - state_manager.paddle["width"]
        ):
            state_manager.paddle["x"] += 40


def ball_logic(state_manager):
    """Move the ball and handle collisions"""
    state_manager.ball["x"] += state_manager.ball["velocity"]["x"]
    state_manager.ball["y"] += state_manager.ball["velocity"]["y"]

    # Check for collisions with walls
    if (
        state_manager.ball["x"] <= 0
        or state_manager.ball["x"] >= state_manager.screen_width
    ):
        state_manager.ball["velocity"]["x"] *= -1

    if state_manager.ball["y"] <= 0:
        state_manager.ball["velocity"]["y"] *= -1

    # Check for collision with paddle
    paddle_rect = pygame.Rect(
        state_manager.paddle["x"],
        state_manager.paddle["y"],
        state_manager.paddle["width"],
        state_manager.paddle["height"],
    )
    ball_rect = pygame.Rect(
        state_manager.ball["x"] - state_manager.ball["radius"],
        state_manager.ball["y"] - state_manager.ball["radius"],
        2 * state_manager.ball["radius"],
        2 * state_manager.ball["radius"],
    )

    if ball_rect.colliderect(paddle_rect):
        state_manager.ball["velocity"]["y"] *= -1

    # Check for collision with bricks
    for brick in state_manager.bricks:
        brick_rect = pygame.Rect(
            brick["x"],
            brick["y"],
            brick["width"],
            brick["height"],
        )

        if ball_rect.colliderect(brick_rect):
            state_manager.bricks.remove(brick)
            state_manager.ball["velocity"]["y"] *= -1
            state_manager.score += 10

    # Check if the ball goes below the paddle


def game_over_logic(state_manager):
    """Handle game over logic"""
    if state_manager.ball["y"] >= state_manager.paddle["y"]:
        state_manager.game_over = True
    if state_manager.game_over:

        font = pygame.font.Font(None, 36)
        game_over_text = font.render("Game Over", True, (0, 0, 0))
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
        move_paddle_logic(self.state_manager, event)
        ball_logic(self.state_manager)
        render(self.state_manager)
        render(state_manager)
        return game_over_logic(state_manager)


if __name__ == "__main__":
    game = Game()
    pygame.init()
    global event
    running = True
    while running:
        event = pygame.event.poll()
        running = game.run(event)
    pygame.quit()