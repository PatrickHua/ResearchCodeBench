old_decompose_prompt = """Your task is to decompose the given game specification into a set of executable steps.
The game has the following implelementation:
```python
{existing_code}
```

The steps should be in the form of a list of separable parts of the game that can be (that is, different steps should share as little state variables as possible).
You don't have to give initialization-related steps but do not leave out important details.
Do not make the steps too detailed. The steps should be at a relatively high level of abstraction (e.g. "implement gravity" instead of "add a variable to keep track of the y position of the balloon and update it every frame").
You are encouraged to rearrange the given instructions so that each step contains (1) at most one input event handler, (2) a state transitional logic, and (3) and a rendering function.

Make sure that different steps are as disjoint as possible (as in they can be implemented separately without affecting each other)

Give your response in the following format:
```json
{{
    "steps": [
        "step 1: describe the first step",
        "step 2: describe the second step",
        "step 3: describe the third step",
        ...
    ]
}}
```

For example:
```json
{{
    "steps": [
        "implement gravity so that the balloon is by default always falling down",
        "implement the ability to move the balloon left and right with arrowkeys",
        ...
    ]
}}  

The specification of the game is: 
{game_specification}


"""

high_level_decompose_prompt = """
Given an unstructured game description, decompose the game's specification into a set of steps or modules. Each step or module should contain at most one input event handling, one state transitional logic, and one rendering logic.
If we model the game as a Markov Decision Process (MDP), the steps, or modules of the game, should share as little state variables as possible.

Please provide the response in the following format:
```json
{{
    "steps": [
        "Step 1: Describe the first module/step",
        "Step 2: Describe the second module/step",
        "Step 3: Describe the third module/step",
        ...
    ],
    "explanations": "To make sure the decomposition don't miss any important game mechanics, explain where each module/step fits in the game's logic."
}}
```

For example:
```json
{{
    "steps": [
        "Step 1: Introduce a balloon asset rendered as a blue rectangle. Implement gravity so that the balloon is by default always falling down. Allow users to move the balloon with arrow keys.",
        "Step 2: Implement a scoring system that rewards the human player for returning the ball back (i.e. making contact with the ball). Display the current score at the top of the screen. Display a 'Game Over!' message when one player reaches a score of 5 and provide an option for the player to restart the game after the 'Game Over' screen is displayed."
        ...
        "Step 5: Implement a 'Game Over' screen that is displayed when the player reaches a score of 5. Allow the player to restart the game from this screen."
    ],
    "explanations": "The balloon asset is the main character of the game. Its rendering and movement are handled in the first module. The scoring system is implemented in the second module to keep track of the player's progress. The 'Game Over' logic is implemented in the last module to provide a clear end to the game."
}}

The unstructured specification of the game is:
{game_specification}


Please provide the structured steps in the format of the JSON above. 
- Ensure that each step is a separable part of the game that can be implemented as independently as possible.
- You most likely don't need to decompose the game into more than 5 steps. However, the most important thing is to ensure that all the steps accurately describe the game's implementation.
- The most important thing is to make sure that the decomposition does not miss any logic step (e.g., the balloon should not be able to go off the screen).
- Note that the order of the steps is the order that these modules will be called in the game loop. Ensure that the game described can be implemented sequentially. For example, the reset position logic should be implemented after the collision detection logic.
"""

state_change_prompt = """The game designer is building a single-player game in Pygame by modeling this game as a Markov Decision Process (MDP).
Your task is to identify and compile a list of relevant state variables to implement a specific feature requested by the game designer.
The game already has the following state space implementation:
```python
import pygame
import sys
import random


{state_manager_code}

    # new variables will be added here:
    # variable_description
    self.{{variable_name}} = {{variable_type}}({{value}})
```

Please provide the state variables in the following format within a JSON object:
```json
{{
    "relevant_state_variables": [
        {{
            "variable_name": "Name of the variable",
        }},
        ...
    ],
    "new_state_variables": [
        {{
            "variable_description": "Description of the variable",
            "variable_name": "Name of the variable",
            "variable_type": "Type of the variable, one of {{int, float, str, bool, tuple, list}}",
            "variable_value": "Value of the variable, e.g. 100, 0.5, 'balloon', tuple((255, 0, 255)), True, [10, 50], [{{'x': 100, 'y': 200}}]",
        }},
        ...
    ]
}}
```

The game designer's request is: {query}.

Here are the dos and don'ts for this request:
- The list "relevant_state_variables" should contain the names of the existing state variables that are relevant to the game designer's request.
- Please return a single list of state variables that contains both existing variables that you think are relevant and new state variables.
- A software engineer will later implement this request by implementing a function that takes these variables as input, so ensure all the variables needed to implement the request are included.
- It is okay to include variables that don't end up being used in the implementation because redundant state variables will be filtered out later.
- Please provide all rendering variables (e.g., size, color) if there are components to be rendered. Color should never be white since the background is white.
- Don't provide Sprite, Surface, or Rect variables. We will handle these variables later.
- Don't introduce variables using existing variables (e.g., self.bird_size = self.pipe_size/2), all state variables should be independent of each other.
- Always provide a default value even if a state variable should be chosen randomly. The randomness will be implemented later.
- "variable_value" should never to empty like []. Always provide a non-empty default value so the software engineer can infer how the variable can be accessed.
- Do not hallucinate external image files (e.g., .png, .jpg) or sound effects(e.g., mp3).
- Prioritize reusing the existing state variables as much as possible. For example, if we have "position_x" and "position_y" of a character, do not give me another variable "positions" in a list format.
- Additionally, you may add new state variables to the list "new_state_variables" if necessary. Please only create new state variables if necessary.
"""

decompose_query_prompt = """The game designer is building a single-player game in Pygame by modeling this game as a Markov Decision Process (MDP).
There are three types of functions/modules in this game: input event handling, state transition, and UI rendering. 
- input event handling: functions that detect user input and update the state variables accordingly.
- state transition: functions that update the state variables according to the game logic.
- UI rendering: functions that render the state variables as UI components on the screen.

Given a specific feature requested by the game designer, your task is to decide how to implement this feature by decomposing it into the three types of functions/modules mentioned above.

The game already has the following implementation:
```python
import pygame
import sys
import random


{state_manager_code}

# all the input event handling functions
{relevant_input_def}

# all the state transitional functions
{relevant_logic_def}

# all the UI rendering functions that govern how state variables are rendered as UI components
{relevant_render_def}

def main():
    state_manager = StateManager()
    running = True
    while running:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = False
        # call all the input event handling functions
        # call all the state transitional functions
        # call all the rendering functions
    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    main()
```

Please provide the answer in the following format within a JSON object:
```json
{{
    "input_logic": {{
        "description": "the detailed description of what this function should achieve.",
        "function_name": "the name of the function to be added",
    }},
    "state_transition": {{
        "description": "the detailed description of what this function should achieve",
        "function_name": "the name of the function to be added",
    }},
    "ui_rendering": {{
        "description": "the detailed description of what this function should achieve",
        "function_name": "the name of the function to be added",
    }},
}}
```

The game designer's request is: {query}.

Here are the dos and don'ts for this request:
- Only give output that pertains to the particular request from the game designer. Do not add things that are not requested. For example, if the game designer asks to "introduce an obstacle", do not add additional logics such as "allow the human player to control the obstacle with arrow keys".
- The initialization of the state variables is already implemented. You should not include steps that initialize or prepare the state variables.
- Be detailed and specific about what the function should achieve. For example, do not give instructions such as "description": "This function should handle input events relevant to the game.".
- The resulting JSON should have three keys: "input_logic", "state_transition", and "ui_rendering". Each key should have two keys: "description" and "function_name". The "description" key should have a string value that describes what the function should achieve. The "function_name" key should have a string value that is the name of the function to be added. If the function already exists, this new function will be used in place of the old one.
- If any particular type of functions is not needed, please leave it as an empty string. It is okay to have empty strings if the function has already been implemented.
- The state variables are already updated according to the game designer's request. You should not include steps that update the state variables.
"""

input_logic_add_prompt = """The game designer is building a single-player game in Pygame by modeling this game as a Markov Decision Process (MDP). Your task is to detect key/mouse input and update the state variables accordingly according to a feature requested by the game designer.
The game has the following implementation already:
```python
import pygame
import sys
import random


{state_manager_code}

# existing input event handling functions
{existing_implementation}
# the new logic function will be here
# if the function is already implemented, it will be replaced with the new implementation

def main():
    state_manager = StateManager()
    running = True
    while running:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = False
        # {{function_description}}
        {{function_name}}(state_manager, event)
    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    main()
```

To ensure the implementation is correct, please also implmeent an unit test for the function. Please implement the following request from the game designer and return your answer in the following format:
```json
{{
    "function_name": "{function_name}",
    "function_description": "{function_description}",
    "function_implementation": "the pygame implementation of the function, including the first line of the function definition",
    "unit_test": "the unit test code for the function"
}}
```

Here are the dos and don'ts for this request:
- Note that the implementation of the function shuold only have two arguments (i.e. state_manager and event).
- The function implementation should involve checking user input with event (i.e. event.type and event.key).
- Minimize the number of functions added while meeting the game designer's requirements. However, make sure to always give the full implementation of the function.
- Include only the essential details requested by the game designer. Do not add things that are not requested.
- Please use the state variables defined in the state manager. Do not introduce new state variables.
- Only KEYDOWN events will be detected. Do not rely on KEYUP events.
- Check for index out of bounds errors with any lists or arrays being used.
- Check for divide-by-zero errors.
- Do not leave any code incomplete. Do not leave placeholder values. Do not provide demonstration code implementation. Be sure all code is fully implemented.
"""

logic_add_prompt = """The game designer is building a single-player game in Pygame by modeling this game as a Markov Decision Process (MDP). Your task is to define and code new state transition functions according to the feature requested by the game designer.
The game has the following implementation already:
```python
import pygame
import sys
import random


{state_manager_code}

# existing state transition functions
{existing_implementation}
# the new function will be here
# if the function is already implemented, it will be replaced with the new implementation


def main():
    state_manager = StateManager()
    running = True
    while running:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = False
        # {{function_description}}
        {{function_name}}(state_manager)
    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    main()
```

To ensure the implementation is correct, please also implmeent an unit test for the function. Please implement the following request from the game designer and return your answer in the following format:
```json
{{
    "function_name": "{function_name}",
    "function_description": "{function_description}",
    "function_implementation": "the pygame implementation of the function, including the first line of the function definition",
    "unit_test": "the unit test code for the function"
}}
```

Here are the dos and don'ts for this request:
- Only implement things that pertain to updating the state variables. Other aspects of the game like input event handling and UI components will be handled separately.
- Include only the essential details requested by the game designer. Do not add things that are not requested.
- These state transition functions will be called in every iteration of the main game loop. If you want to add a conditional logic to the function, please implement it in the function itself.
- Note that this new function will be added to the end of the list of state transition functions.
- Please use the state variables defined in the state manager. Do not introduce new state variables.
- Check for index out of bounds errors with any lists or arrays being used.
- Check for divide-by-zero errors.
- Do not leave any code incomplete. Do not leave placeholder values. Do not provide demonstration code implementation. Be sure all code is fully implemented.
"""

ui_add_prompt = """The game designer is building a single-player game in Pygame by modeling this game as a Markov Decision Process (MDP). Your task is to add rendering functions that decide how state variables are rendered as UI components on the screen, according to the feature requested by the game designer.
The game has the following implementation already:
```python
import pygame
import sys
import random


{state_manager_code}

# existing rendering functions
{render_code}
# the new function will be here
# if the function is already implemented, it will be replaced with the new implementation

def main():
    state_manager = StateManager()
    clock = pygame.time.Clock()
    running = True
    while running:
        action = pygame.event.poll()
        if action.type == pygame.QUIT:
            running = False

        # all the code for state transitional logics
        # omitted for brevity

        # Fill the screen with white
        state_manager.screen.fill((255, 255, 255))
        # all the code for rendering states as UI components
        # {{function_description}}
        {{function_name}}(state_manager)
        pygame.display.flip()
        state_manager.clock.tick(state_manager.fps)

    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("")
    main()
```

To ensure the implementation is correct, please also implmeent an unit test for the function. Please implement the following request from the game designer and return your answer in the following format:
```json
{{
    "function_name": "{function_name}",
    "function_description": "{function_description}",
    "function_implementation": "the pygame implementation of the function, including the first line of the function definition",
    "unit_test": "the unit test code for the function"
}}
```

Here are the dos and don'ts for this request:
- Only implement things that pertain to how state variables are rendered as UI components on the screen. Other aspects like input event handling and state transition will be handled separately.
- Please make sure that all of the state variables remain unchanged in the rendering functions.
- Include only the essential details requested by the game designer. Do not add things that are not requested.
- These rendering functions will be called in every iteration of the main game loop. If you want to add a conditional logic to the function, please implement it in the function itself.
- Note that the background color of the screen is white so white UI components will not be visible. Do not fill the screen with white again in the rendering functions.
- Note that the new function will be added to the end of the list of rendering functions.
- Please use the state variables defined in the state manager. Do not introduce new state variables.
- Check for index out of bounds errors with any lists or arrays being used.
- Check for divide-by-zero errors.
- Do not call pygame.display.set_mode in UI functions. Only call it once outside of any UI function that is called multiple times.
- Do not leave any code incomplete. Do not leave placeholder values. Do not provide demonstration code implementation. Be sure all code is fully implemented.
"""
