import game.wrapped_flappy_bird as game
import pygame
from pygame.locals import *
import sys

from sklearn.neural_network import MLPClassifier
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import joblib


N_ACTIONS = 2
N_AGENTS = 60
START_WEIGHTS_PATH=None
START_WEIGHTS_PATH = "mlp_agent_100_prom_save_2.pkl"




def collect_parametrs(action_terminal):
    '''
        Эта функция собирает из текущей среды все необходимые параметры
    '''
    parametrs = []
    parametrs.append(action_terminal.playery)

    playerMidPos = action_terminal.playerx + game.PLAYER_WIDTH / 2
    for i in range(len(action_terminal.upperPipes)):
        pipeMidPos = action_terminal.upperPipes[i]['x'] + game.PIPE_WIDTH / 2
        if pipeMidPos <= playerMidPos:
            continue
        parametrs.append(action_terminal.upperPipes[i]['x'])
        parametrs.append(action_terminal.upperPipes[i]['y'])
        parametrs.append(action_terminal.upperPipes[i]['y'] + game.PIPEGAPSIZE)
        break

    parametrs.append(action_terminal.playerVelY)
    return parametrs

def select_elites(states_batch, actions_batch, rewards_batch, percentile = 70):
    '''
        Эта функция отбирает элитные сессии из батча
    '''

    elite_states = []
    elite_actions = []
    reward_threshold = np.percentile(rewards_batch, percentile)
    for session_states, session_actions, session_reward in zip(states_batch, actions_batch, rewards_batch):
        if session_reward < reward_threshold:
            continue
        elite_states.extend(session_states)
        elite_actions.extend(session_actions)

    return elite_states, elite_actions

def main():
    '''
        Обучение модели
    '''

    action_terminal = game.GameState()

    #Инициализируем модель случайно, либо загружаем готовые веса
    if START_WEIGHTS_PATH is None:
        agent = MLPClassifier(
            hidden_layer_sizes=(80, 80),
            activation="logistic",
            warm_start=True,
            max_iter=1,
        )

        agent.fit([collect_parametrs(action_terminal)] * N_ACTIONS, range(N_ACTIONS))
    else:
        agent = joblib.load(START_WEIGHTS_PATH)

    max_reward=0

    # Обучаем модель
    for epoch in range(1000):
        states_batch = []
        actions_batch = []
        rewards_batch = []
        for i in range(N_AGENTS):

            states=[]
            actions=[]
            sum_reward = 0
            while True:
                # Получаем предсказание
                proba = agent.predict_proba([collect_parametrs(action_terminal)]).squeeze()
                # Выбираем случайный вариант в соответствии с вероятностями
                action = np.random.choice(N_ACTIONS, p=proba)

                if action == 0:
                    input_actions = [1, 0]
                else:
                    input_actions = [0, 1]

                states.append(collect_parametrs(action_terminal))
                actions.append(action)

                # if i ==N_AGENTS-1:
                #     _, reward, terminal = action_terminal.frame_step(input_actions,30)
                # else:
                _, reward, terminal = action_terminal.frame_step(input_actions,None)

                sum_reward+=reward

                if terminal:
                    break

            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(sum_reward)

        elite_states,elite_actions = select_elites(states_batch,actions_batch,rewards_batch)

        #Сохраняем веса модели, которая показала наилучший результат среди пройденных батчей
        if max_reward<max(rewards_batch):
            joblib.dump(agent, "mlp_agent.pkl")
            max_reward = max(rewards_batch)
        #Тренируем модель на элитных сессиях
        agent.fit(elite_states,elite_actions)
        print(epoch,f'max = {max(rewards_batch)}, mean = {np.mean(rewards_batch)}')

if __name__ == '__main__':
    '''
        Обучение модели со случайностью (ответ выбирается как np.random.choice(N_ACTIONS, p=proba))
        Параметры влияющие на работу этого алгоритма находятся сверху файла после import

        N_ACTIONS - количество действий, для данной игры не изменяется
        N_AGENTS - количество сессий в одной эпохе
        START_WEIGHTS_PATH - путь к начальным весам агента
    '''
    main()

