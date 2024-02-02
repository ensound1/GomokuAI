#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
from tqdm import tqdm
import gc
import wandb
import os
from game_classes import GomokuBoard

print("ACTIVE GPUS: ", tf.config.list_physical_devices("GPU"))

def softmax(x, temperature=0.005):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

def choose_move_by_probability(predictions, temperature = 0.05):
    """ Sceglie una mossa basata sulla probabilità calcolata dalle predizioni. """
    # Trasforma le stime per favorire i valori più bassi
    transformed_predictions = - predictions
    # Calcola la distribuzione di probabilità
    probabilities = softmax(transformed_predictions, temperature)
    # Esegui sampling basato sulla distribuzione di probabilità
    move_index = np.random.choice(len(predictions), p=probabilities)
    return move_index

def monte_carlo_value_estimation(games_data, gamma=0.9):
    # Inizializza il valore di ogni stato della griglia come una lista vuota
    state_values = {}
    double_counted = 0
    
    def flip(board):
        flipped_board = [[cell if cell == " " else ("O" if cell == "X" else "X") for cell in row] for row in board]
        return flipped_board

    for game_data, winner in games_data:
        i = 0
        for board, player in reversed(game_data):
            # Se il giocatore corrente ha vinto, il reward è +1, altrimenti -1
            if player == winner:
                reward = 1
            elif not winner == None:
                reward = -1
            else:
                reward = 0
            G = reward * gamma**(i)
            if player == "X":
                # Aggiorna il valore del board utilizzando l'approccio Monte Carlo
                state = tuple(map(tuple, board))  # Rappresenta la griglia come una tupla hashable
                if state in state_values:
                    state_values[state].append(G)
                else:
                    state_values[state] = [G]
            else:
                # Aggiorna il valore del board utilizzando l'approccio Monte Carlo
                state = tuple(map(tuple, flip(board)))  # Rappresenta la griglia come una tupla hashable
                if state in state_values:
                    state_values[state].append(G)
                else:
                    state_values[state] = [G]
            i+=1

    # Calcola la media dei valori per ciascuno stato
    tot = 0
    double_counted = 0
    for state, values in state_values.items():
        state_values[state] = sum(values) / len(values)
        double_counted +=len(values)-1
        tot += len(values)
    print("Double counted: ", double_counted, " ", np.round(double_counted/tot*100), "%")
    
    X = []
    y = []
    for state, value in state_values.items():
        X.append(state)
        y.append(value)
    return X, y

def fight(modelA, modelB, n_games = 500, temperature = 0.05, random_prob = 0, random_init=1, verbose = 1):
    games_data = [[[], None] for _ in range(n_games)]
    games = [GomokuBoard() for _ in range(n_games)]
    unfinished_games = set(range(n_games))

    pbar = tqdm(total=n_games) if verbose == 1 else None
    iteration = 0
    while unfinished_games:
        batch_moves = []
        batch_positions = []
        game_indices = []

        #Iterate over a copy of unfinished_games (indices)
        for game_index in list(unfinished_games):
            game = games[game_index]

            winner =  game.is_last_winnning()
            if winner:
                games_data[game_index][1] = winner
                games_data[game_index][0].append((np.copy(game.board), game.current_player))
                unfinished_games.remove(game_index)
                continue
            
            if iteration<random_init or random_prob >= np.random.random():
                 row, col = random.randint(0, game.size), random.randint(0, game.size)
                 while not game.is_valid_move(row, col):
                      row, col = random.randint(0, game.size), random.randint(0, game.size)
                 game.make_move(row, col)
                 games_data[game_index][0].append((np.copy(game.board), game.current_player))  # Salva lo stato della griglia e il giocatore corrente
                 continue
            
            #if random move 
            for row in range(game.size):
                for col in range(game.size):
                    if game.board[row][col] == " ":
                        temp_board = np.array(game.board)
                        temp_board[row][col] = "X" if game.player_to_move == "X" else "O"
                        batch_moves.append(temp_board)
                        batch_positions.append((row, col))
                        game_indices.append(game_index)

        if batch_moves:
            if game.current_player == "X":
                prepared_input = GomokuBoard.prepare_input(modelA, batch_moves, "O")
                predictions = modelA.predict(prepared_input, verbose=0)
            else:
                prepared_input = GomokuBoard.prepare_input(modelB, batch_moves, "X")
                predictions = modelB.predict(prepared_input, verbose=0)
                
            preds = []
            pos = []
            current_game_index = game_indices[0]
            for i, pred in enumerate(predictions):
                pred = pred[0]
                if current_game_index == game_indices[i]:
                    preds.append(pred)
                    pos.append(batch_positions[i])
                else:
                    best_move_idx = choose_move_by_probability(np.array(preds), temperature)
                    best_row, best_col = pos[best_move_idx]
                    if not games[current_game_index].make_move(best_row, best_col):
                        print("Invalid intermidiate move!")
                    games_data[current_game_index][0].append((np.copy(games[current_game_index].board), games[current_game_index].current_player))
                    preds = [pred]
                    current_game_index = game_indices[i]
                    pos = [batch_positions[i]]
            best_move_idx = choose_move_by_probability(np.array(preds), temperature)
            best_row, best_col = pos[best_move_idx]
            if not games[current_game_index].make_move(best_row, best_col):
                print("Invalid intermidiate move!")
            games_data[current_game_index][0].append((np.copy(games[current_game_index].board), games[current_game_index].current_player))
        if pbar is not None:
            pbar.n = n_games - len(unfinished_games)
            pbar.refresh()
        iteration += 1
        del batch_moves, batch_positions
        

    if pbar is not None:
        pbar.close()
        
    del games
    return games_data

early_stopping = EarlyStopping(
    monitor='val_loss',  # or 'val_accuracy' if you're focusing on accuracy
    min_delta=0.001,     # minimum change to qualify as an improvement
    patience=2,         # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
)

save_file= "DISPREZZO"
game_size = 15
steps = 100
tests = 5
n_games = 500
n_games_fight = 100
fight_mode = True
baseline1 = 10
baseline2 = 50
baseline3 = 100

# Lista tutti i file nella directory corrente
files = os.listdir(".")

# Filtra i file che corrispondono al pattern del modello
model_files = [f for f in files if f.startswith("DISPREZZOv")]

# Assicurati che ci siano file di modello
if model_files:
    # Estrai le versioni e trova il massimo
    versions = [int(f.split('v')[1]) for f in model_files]
    max_version = max(versions)

    # Carica il modello con la versione più alta
    model = tf.keras.models.load_model("DISPREZZOv" + str(max_version))
else:
    print("Nessun file di modello trovato nella directory corrente.")

import psutil
def print_mem():
    memory_info = psutil.virtual_memory()
    available_memory = memory_info.available / (1024 * 1024)  # Converti in MB
    print(f"Memoria disponibile: {available_memory:.2f} MB")
    
def main():
    wandb.init(project="InfiniteTicTacToe", entity="ensound")
    games_data = []
    for step in range(max_version+1,max_version+steps+1):
        print("Step", step)
        print_mem()
        wandb_metrics = {}  # Dictionary to collect metrics
        #Creo i dati
        games_data += fight(model, model, n_games, temperature = 0.05, random_prob = 0.05, random_init = 1)
        mean_length = np.average([len(games_data[i][0]) for i in range(n_games)])
        print("Lunghezza media di una partita: ", mean_length)
        gamma = 0.9
        print("Gamma: ", np.round(gamma*100), "%")
        print("Vittorie di X: ", np.average([1  if games_data[i][1] == "X" else 0 for i in range(n_games)]))
        X, y = monte_carlo_value_estimation(games_data, gamma)
        del games_data
        gc.collect()
        joblib.dump(X, "dis/X"+str(step))
        joblib.dump(y, "dis/y"+str(step)) 
        print("Number of training examples: ", len(X))
        X = GomokuBoard.prepare_input(model, X, )
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle = False)
        history = model.fit(X_train, y_train, epochs=100, batch_size=2048, validation_data=(X_val, y_val), callbacks=[early_stopping])
        wandb_metrics['loss'] = history.history['loss'][-1]
        wandb_metrics['val_loss'] = history.history['val_loss'][-1]
        wandb_metrics['games'] = step * n_games
        wandb_metrics['mean_length'] = mean_length
        wandb_metrics['step'] = step
        
        #Salvo sempre e comunque...
        model.save(save_file+"v" + str(step))
        games_data = []
        if not step%tests and fight_mode:
            #testo contro i precedenti
            if step > baseline1:
                baseline1_model = tf.keras.models.load_model(save_file+"v"+str(baseline1))                
                baseline1_fight_data = fight(baseline1_model, model, n_games_fight)
                baseline1_wins = np.average([1  if baseline1_fight_data[i][1] == "X" else 0 for i in range(n_games_fight)])
                print("Vittorie sulla baseline a iter",baseline1,": ", baseline1_wins)
                games_data+=baseline1_fight_data
                wandb_metrics['baseline' + str(baseline1)] = baseline1_wins
            if step > baseline2:
                baseline2_model = tf.keras.models.load_model(save_file+"v"+str(baseline2))                
                baseline2_fight_data = fight(baseline2_model,model, n_games_fight)
                baseline2_wins = np.average([1  if baseline2_fight_data[i][1] == "X" else 0 for i in range(n_games_fight)])
                print("Vittorie sulla baseline a iter ",baseline2,": ", baseline2_wins)
                games_data+=baseline2_fight_data
                wandb_metrics['baseline' + str(baseline2)] = baseline2_wins
            if step > baseline3:
                baseline3_model = tf.keras.models.load_model(save_file+"v"+str(baseline3))                
                baseline3_fight_data = fight(baseline3_model,model,  n_games_fight)
                baseline3_wins = np.average([1  if baseline3_fight_data[i][1] == "X" else 0 for i in range(n_games_fight)])
                print("Vittorie sulla baseline a iter ",baseline3,": ", baseline3_wins)
                games_data+=baseline3_fight_data
                wandb_metrics['baseline' + str(baseline3)] = baseline3_wins

        # Make a single call to wandb.log
        wandb.log(wandb_metrics)
            

if __name__ == "__main__":
    #cProfile.run('main()', sort='cumtime')
    main()





