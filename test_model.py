import game.wrapped_flappy_bird as game
import pygame
from pygame.locals import *
import sys
import joblib
import numpy as np

from study import collect_parametrs

def main(path_to_weights):
    action_terminal = game.GameState()

    #Указываем путь к весам
    agent = joblib.load(path_to_weights)

    while True:
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        proba = agent.predict_proba([collect_parametrs(action_terminal)]).squeeze()
        # action = np.random.choice(2, p=proba)
        action = np.argmax(proba)

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        if action == 0:
            input_actions = [1, 0]
        else:
            input_actions = [0, 1]

        action_terminal.frame_step(input_actions,60)


if __name__ == '__main__':
    '''
        Файл показывающий работу модели
    '''
    path_to_weights = "weights/mlp_agent_100_final.pkl"
    main(path_to_weights)
