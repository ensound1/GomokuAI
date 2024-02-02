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

# Configurazione TensorFlow per evitare l'allocazione completa della memoria GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print("ACTIVE GPUS: ", tf.config.list_physical_devices("GPU"))


class InfiniteTicTacToe:
    def __init__(self, size=15):
        self.size = size  # Dimensione della griglia di gioco
        self.board = [[" " for _ in range(self.size)] for _ in range(self.size)]  # Inizializzazione della griglia
        self.board_values = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.current_player_value = 1
        self.current_player = "X"  # Il giocatore corrente (X o O)
        self.last_row, self.last_col = None, None
        self.last_last_row, self.last_last_col = None, None

    def print_board(self):
        for row in self.board:
            print("|".join(row))
            print("-" * (self.size * 2 - 1))

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            winner = self.check_winner()
            self.current_player = "O" if self.current_player == "X" else "X"
            self.board_values[row][col] = self.current_player_value
            self.current_player_value = -1 if self.current_player_value == 1 else 1
            self.last_last_row, self.last_last_col = self.last_row, self.last_col
            self.last_row, self.last_col = row, col
            return True, winner
        return False, None


    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == " "

    def check_winner(self):
        if self.last_row == None:
            return None
        player = self.board[self.last_row][self.last_col]
        if player == " ":
            return None

        # Controlla la riga, la colonna e le diagonali dell'ultima mossa
        if (self.check_line_winner(self.last_row, self.last_col, 1, 0) or  # Orizzontale
            self.check_line_winner(self.last_row, self.last_col, 0, 1) or  # Verticale
            self.check_line_winner(self.last_row, self.last_col, 1, 1) or  # Diagonale principale
            self.check_line_winner(self.last_row, self.last_col, 1, -1)):  # Diagonale secondaria
            return player

        return None
        
    def check_line_winner(self, start_row, start_col, delta_row, delta_col):
        player = self.board[start_row][start_col]
        count = 1  # Inizia con 1 perché include l'ultima mossa

        # Controlla in una direzione
        for i in range(1, 5):
            new_row = start_row + i * delta_row
            new_col = start_col + i * delta_col
            if 0 <= new_row < self.size and 0 <= new_col < self.size and self.board[new_row][new_col] == player:
                count += 1
            else:
                break

        # Controlla nella direzione opposta
        for i in range(1, 5):
            new_row = start_row - i * delta_row
            new_col = start_col - i * delta_col
            if 0 <= new_row < self.size and 0 <= new_col < self.size and self.board[new_row][new_col] == player:
                count += 1
            else:
                break
        return count >= 5  # Restituisce True se una riga di 5 è stata completata, altrimenti False
                
    def complete_row_of_5(self):
        if self.last_last_row is None:
            return None

        # Limita il controllo alle celle che potrebbero formare una riga di 5
        for delta_row in range(-4, 5):
            for delta_col in range(-4, 5):
                row = self.last_last_row + delta_row
                col = self.last_last_col + delta_col
    
                if 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == " ":
                    # Simula una mossa e controlla se forma una riga di 5
                    self.board[row][col] = self.current_player
                    last_row_ = self.last_row 
                    last_col_ = self.last_col
                    self.last_row = row
                    self.last_col = col
                    if self.check_winner() == self.current_player:
                        self.board[row][col] = " "  # Ripristina la griglia
                        self.last_row = last_row_
                        self.last_col = last_col_
                        return row, col  # Restituisci le coordinate della casella da riempire
                    self.board[row][col] = " "  # Ripristina la griglia
                    self.last_row = last_row_
                    self.last_col = last_col_
    
        return None
       
    def get_possible_moves(self):
        possible_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == " ":
                    possible_moves.append((row, col))
        return possible_moves


    def is_game_over(self):
        if self.check_winner() is not None:
            return True
        for row in self.board:
            if " " in row:
                return False
        return True  # Draw
          
    def get_winner(self):
        winner = self.check_winner()
        if winner is not None:
            return winner
        if all(all(cell != " " for cell in row) for row in self.board):
            return "Draw"
        return None

    def get_move_from_state(self, old_state, new_state):
        for row in range(self.size):
            for col in range(self.size):
                if old_state[row][col] != new_state[row][col]:
                    return row, col
        return None


# In[13]:


def softmax(x, temperature=0.005):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

def choose_move_by_probability(predictions, temperature = 0.005):
    """ Sceglie una mossa basata sulla probabilità calcolata dalle predizioni. """
    # Trasforma le stime per favorire i valori più bassi
    transformed_predictions = - predictions
    # Calcola la distribuzione di probabilità
    probabilities = softmax(transformed_predictions, temperature)
    # Esegui sampling basato sulla distribuzione di probabilità
    move_index = np.random.choice(len(predictions), p=probabilities)
    return move_index


# In[14]:


def generate_by_mixed_strategy(n_games, model, game_size, temperature = 1, fraction = 0.1, pad = 3):
    games_data = [[[], None] for _ in range(n_games)]
    games = [InfiniteTicTacToe(game_size) for _ in range(n_games)]
    unfinished_games = set(range(n_games))


    def prepare_input(batch_moves):
        input_shape = model.input_shape  # Assumendo che input_shape sia accessibile
        if input_shape == (None, game_size, game_size, 2):
            return double_grids(batch_moves)
        return batch_moves

    with tqdm(total=n_games) as pbar:
        iteration = 0
        while unfinished_games:
            batch_moves = []
            batch_positions = []
            game_indices = []
    
            # Itera su una copia del set unfinished_games
            for game_index in list(unfinished_games):
                game = games[game_index]

                winner = game.check_winner()
                if winner in ["X", "O"]:
                    games_data[game_index][1] = winner
                    games_data[game_index][0].append((np.copy(game.board), game.current_player))
                    unfinished_games.remove(game_index)
                    continue
                else:
                    if iteration<pad or fraction >= np.random.random():
                        row, col = random.randint(0, game.size - 1), random.randint(0, game.size - 1)
                        while not game.is_valid_move(row, col):
                            row, col = random.randint(0, game.size - 1), random.randint(0, game.size - 1)
                        game.make_move(row, col)
                        games_data[game_index][0].append((np.copy(game.board), game.current_player))  # Salva lo stato della griglia e il giocatore corrente
                        continue
                        
                    for row in range(game.size):
                        for col in range(game.size):
                            if game.board[row][col] == " ":
                                temp_board = np.copy(game.board_values)
                                if game.current_player == "X":
                                    temp_board = -temp_board
                                temp_board[row][col] = -1
                                temp_board = temp_board.reshape(1, game.size, game.size, 1)
                                batch_moves.append(temp_board)
                                batch_positions.append((row, col))
                                game_indices.append(game_index)
            if batch_moves:
                prepared_input = prepare_input(np.concatenate(batch_moves, axis=0))
                predictions = model.predict(prepared_input, verbose=0)

                current_game_index = game_indices[0]
                index_of_first_prediction_for_current_game = 0
                for i, pred in enumerate(predictions):
                    if current_game_index != game_indices[i]:
                        best_move_idx = choose_move_by_probability(predictions[index_of_first_prediction_for_current_game:i].flatten(), temperature)
                        best_row, best_col = batch_positions[index_of_first_prediction_for_current_game+best_move_idx]
                        if not games[current_game_index].make_move(best_row, best_col):
                            print("Mossa intermedia non valida!")
                        games_data[current_game_index][0].append((np.copy(games[current_game_index].board), games[current_game_index].current_player))
                        index_of_first_prediction_for_current_game = i
                        current_game_index = game_indices[i]
                #Devo chiudere l'ultima partita!
                best_move_idx = choose_move_by_probability(predictions[index_of_first_prediction_for_current_game:].flatten(), temperature)
                best_row, best_col = batch_positions[index_of_first_prediction_for_current_game+best_move_idx]
                if not games[current_game_index].make_move(best_row, best_col):
                    print("Mossa intermedia non valida!")
                games_data[current_game_index][0].append((np.copy(games[current_game_index].board), games[current_game_index].current_player))
            pbar.n = n_games - len(unfinished_games)
            pbar.refresh()
            iteration += 1
    return games_data


# In[15]:


def flip(board):
    flipped_board = [[cell if cell == " " else ("O" if cell == "X" else "X") for cell in row] for row in board]
    return flipped_board
# Supponiamo che tu abbia già il tuo dataset di giochi come games_data
# games_data è una lista di tuple, dove ogni tupla contiene (game_data, winner)

def monte_carlo_value_estimation(games_data, gamma=1):
    # Inizializza il valore di ogni stato della griglia come una lista vuota
    state_values = {}
    double_counted = 0

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
    print("Doppioni: ", double_counted, " ", np.round(double_counted/tot*100), "%")
    return state_values


# Calcola i valori utilizzando l'approccio Monte Carlo
#state_values = monte_carlo_value_estimation(games_data)
def convert_board_to_matrix(board):
    matrix = []
    for row in board:
        matrix_row = []
        for cell in row:
            if cell == "X":
                matrix_row.append(1)
            elif cell == "O":
                matrix_row.append(-1)
            else:
                matrix_row.append(0)
        matrix.append(matrix_row)
    return matrix

def prepare_data(state_values):
    X = []
    y = []

    for state, value in state_values.items():
        matrix = convert_board_to_matrix(state)
        X.append(matrix)
        y.append(value)

    # Convertendo le liste in array numpy
    X = np.array(X)
    y = np.array(y)

    # Reshaping X per aggiungere un canale extra (necessario per l'input nel modello CNN)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    return X, y


# In[16]:


wandb.init(project="InfiniteTicTacToe", entity="ensound")


# In[17]:


def double_grids(grids):
    # Creazione di un nuovo array numpy con la dimensione desiderata
    new_grids = np.zeros((grids.shape[0], grids.shape[1], grids.shape[2], 2))

    # Impostare 1 nel primo canale dove c'erano 1
    new_grids[:,:,:,0] = (grids == 1).squeeze()

    # Impostare 1 nel secondo canale dove c'erano -1
    new_grids[:,:,:,1] = (grids == -1).squeeze()

    return new_grids


# In[18]:


def fight(modelA, modelB, n_games = 1, temperature = 0.01, verbose = 1):
    games_data = [[[], None] for _ in range(n_games)]
    games = [InfiniteTicTacToe(game_size) for _ in range(n_games)]
    unfinished_games = set(range(n_games))

    def prepare_input(model, batch_moves):
        input_shape = model.input_shape  # Assumendo che input_shape sia accessibile
        if input_shape == (None, 15, 15, 2):
            return double_grids(batch_moves)
        return batch_moves

    pbar = tqdm(total=n_games) if verbose == 1 else None
    while unfinished_games:
        
        batch_moves = []
        batch_positions = []
        game_indices = []

        # Itera su una copia del set unfinished_games
        for game_index in list(unfinished_games):
            game = games[game_index]

            if game.check_winner():
                games_data[game_index][1] = "X" if  game.current_player == "O" else "O"
                unfinished_games.remove(game_index)
                continue
                
            for row in range(game.size):
                for col in range(game.size):
                    if game.board[row][col] == " ":
                        temp_board = convert_board_to_matrix(game.board)
                        temp_board = np.array(temp_board)
                        if game.current_player == "X":
                            temp_board = -temp_board
                        temp_board[row][col] = -1
                        temp_board = temp_board.reshape(1, game.size, game.size, 1)
                        batch_moves.append(temp_board)
                        batch_positions.append((row, col))
                        game_indices.append(game_index)

        if batch_moves:
            if game.current_player == "X":
                prepared_input = prepare_input(modelA, np.concatenate(batch_moves, axis=0))
                predictions = modelA.predict(prepared_input, verbose=0)
            else:
                prepared_input = prepare_input(modelB, np.concatenate(batch_moves, axis=0))
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
                        print("Mossa intermedia non valida!")
                    games_data[current_game_index][0].append((np.copy(games[current_game_index].board), games[current_game_index].current_player))
                    preds = [pred]
                    current_game_index = game_indices[i]
                    pos = [batch_positions[i]]
            best_move_idx = choose_move_by_probability(np.array(preds), temperature)
            best_row, best_col = pos[best_move_idx]
            if not games[current_game_index].make_move(best_row, best_col):
                print("Mossa intermedia non valida!")
            games_data[current_game_index][0].append((np.copy(games[current_game_index].board), games[current_game_index].current_player))
        if pbar is not None:
            pbar.n = n_games - len(unfinished_games)
            pbar.refresh()
        del batch_moves
    if pbar is not None:
        pbar.close()

    return games_data


# In[19]:


early_stopping = EarlyStopping(
    monitor='val_loss',  # or 'val_accuracy' if you're focusing on accuracy
    min_delta=0.001,     # minimum change to qualify as an improvement
    patience=2,         # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
)
#online learning, cioè ne gioco 100 e poi alleno e così via... vediamo che succede. per testare lo faccio combattere contro tutti i modelli precedenti
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
def main():
    games_data = None
    for step in range(max_version+1,max_version+steps+1):
        memory_info = psutil.virtual_memory()
        available_memory = memory_info.available / (1024 * 1024)  # Converti in MB
        print(f"Memoria residua disponibile: {available_memory:.2f} MB")
        wandb_metrics = {}  # Dictionary to collect metrics
        print("Step", step)
        #Creo i dati
        if games_data is None:
            games_data = generate_by_mixed_strategy(n_games, model, game_size, temperature = 0.01, fraction = 0.05, pad = 1)
            #games_data += generate_by_mixed_strategy(n_games, model, game_size, temperature = 0.01, fraction = 0.05, pad = 10)
        else:
            games_data += generate_by_mixed_strategy(n_games, model, game_size, temperature = 0.01, fraction = 0.05, pad = 1)
        mean_length = np.average([len(games_data[i][0]) for i in range(n_games)])
        print("Lunghezza media di una partita: ", mean_length)
        #gamma = (0.05)**(1/mean_length)
        gamma = 0.9
        print("Gamma: ", np.round(gamma*100), "%")
        print("Vittorie di X: ", np.average([1  if games_data[i][1] == "X" else 0 for i in range(n_games)]))
        state_values = monte_carlo_value_estimation(games_data, gamma)
        del games_data
        gc.collect()
        print("Costruisco dataset e lo salvo")
        X, y = prepare_data(state_values)
        joblib.dump(X, "dis/X"+str(step))
        joblib.dump(y, "dis/y"+str(step)) 
        print("Preparo la matrice doppia")
        X = double_grids(X)
        #print("Aumento il dataset "+ str(len(X)) +"-->", end = " ")
        #X, y = augment(X, y)
        print("Numero di training examples: ", len(X))
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
    try:
        main()
    except e:
        print(e)





