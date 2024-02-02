import numpy as np

class GomokuBoard:
    board_size = 15
    def __init__(self, board = None, player_to_move = "X", board_size=15):
        if board is None:
            self.board = [[" " for _ in range(GomokuBoard.board_size)] for _ in range(GomokuBoard.board_size)]
        else:
            self.board = board
            GomokuBoard.board_size = len(board[0])
        self.player_to_move = player_to_move
        self.history_of_moves = []  #non so come recuperarle dalla board
        self.winner = None

    def print_board(self):
        for row in self.board:
            print("|".join(row))
            print("-" * (self.board_size * 2 - 1))

    def make_move(self, row, col, player=None):
        if player is None:
            player = self.player_to_move
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            print(player, "played", (row,col))
            self.player_to_move = "O" if player == "X" else "X"
            self.history_of_moves.append((row,col))
            return True
        print("MOSSA NON VALIDA")
        return False

    def is_valid_move(self, row, col):
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == " "

    def is_last_winning(self):
        if self.history_of_moves == []:
            return False
        # Controlla la riga, la colonna e le diagonali dell'ultima mossa
        if (self.check_line_winner(self.history_of_moves[-1][0], self.history_of_moves[-1][1], 1, 0) or  # Orizzontale
            self.check_line_winner(self.history_of_moves[-1][0], self.history_of_moves[-1][1], 0, 1) or  # Verticale
            self.check_line_winner(self.history_of_moves[-1][0], self.history_of_moves[-1][1], 1, 1) or  # Diagonale principale
            self.check_line_winner(self.history_of_moves[-1][0], self.history_of_moves[-1][1], 1, -1)):  # Diagonale secondaria
            return "X" if self.player_to_move == "O" else "O"
        return False
        
    def check_line_winner(self, start_row, start_col, delta_row, delta_col):
        player = self.board[start_row][start_col]
        count = 1  # Inizia con 1 perché include l'ultima mossa

        # Controlla in una direzione
        for i in range(1, 5):
            new_row = start_row + i * delta_row
            new_col = start_col + i * delta_col
            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size and self.board[new_row][new_col] == player:
                count += 1
            else:
                break

        # Controlla nella direzione opposta
        for i in range(1, 5):
            new_row = start_row - i * delta_row
            new_col = start_col - i * delta_col
            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size and self.board[new_row][new_col] == player:
                count += 1
            else:
                break
        return count >= 5  # Restituisce True se una riga di 5 è stata completata, altrimenti False
       
    def get_possible_moves(self):
        possible_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == " ":
                    possible_moves.append((row, col))
        return possible_moves
        
    def get_possible_positions(self):
        possible_positions = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == " ":
                    temp_board = np.copy(np.array(self.board))
                    temp_board[row, col] = self.player_to_move
                    possible_positions.append(temp_board)
        return possible_positions
    
    @staticmethod
    def get_player_to_move(board):
        tot_moves = sum([sum([1 for pos in row if pos != " "]) for row in board])
        return "X" if tot_moves%2 == 0 else "O"

    @staticmethod
    def board_to_single_channel_values(boards):
        # Calcola la dimensione dell'output basandoti sul numero di boards
        board_values = np.zeros((len(boards), GomokuBoard.board_size, GomokuBoard.board_size, 1))

        for b_index, board in enumerate(boards):
            for row in range(GomokuBoard.board_size):
                for col in range(GomokuBoard.board_size):
                    perspective = GomokuBoard.get_player_to_move(board)
                    if board[row][col] == perspective:
                        board_values[b_index, row, col, 0] = 1
                    elif board[row][col] == ("O" if perspective == "X" else "X"):
                        board_values[b_index, row, col, 0] = -1

        return board_values

    @staticmethod
    def board_to_double_channel_values(boards):
        # Calcola la dimensione dell'output basandoti sul numero di boards
        boards_count = len(boards)
        board_values = np.zeros((boards_count, GomokuBoard.board_size, GomokuBoard.board_size, 2))

        for b_index, board in enumerate(boards):
            for row in range(GomokuBoard.board_size):
                for col in range(GomokuBoard.board_size):
                    perspective = GomokuBoard.get_player_to_move(board)
                    if board[row][col] == perspective:
                        board_values[b_index, row, col, 0] = 1  # Canale per le pedine del giocatore
                    elif board[row][col] == ("O" if perspective == "X" else "X"):
                        board_values[b_index, row, col, 1] = 1  # Canale per le pedine dell'avversario

        return board_values

    @staticmethod
    def prepare_input(model, batch_positions):
        input_shape = model.input_shape  # Assumendo che input_shape sia accessibile
        if input_shape == (None, 15, 15, 1):
            return GomokuBoard.board_to_single_channel_values(batch_positions)
        elif input_shape == (None, 15, 15, 2):
            return GomokuBoard.board_to_double_channel_values(batch_positions)
        else:
            print(f"Input shape not supported: {input_shape}")

class Player:
    def __init__(self, gomoku_board, symbol = "X"):
        self.gomoku_board  = gomoku_board
        self.symbol = symbol
    def make_move(self, row, col):
        if not self.gomoku_board.make_move(row, col, self.symbol):
            print("Invalid move by player")
            return False
        return True
    
class AIPlayer:
    def __init__(self,gomoku_board, model = None, temperature = 0.05, difficulty = 0, symbol = "O"):
        self.gomoku_board  = gomoku_board
        self.symbol = symbol
        self.model = model
        self.temperature = temperature
        self.difficulty = difficulty

    def choose_move_by_value(self, moves, values, temperature = 0.05):
        # Calcola la distribuzione di probabilità
        probabilities = self.softmax(values, temperature)
        # Esegui sampling basato sulla distribuzione di probabilità
        move_index = np.random.choice(len(moves), p=probabilities)
        # Restituisci la mossa corrispondente a quell'indice
        return moves[move_index]
        
    def make_move(self, row, col):
        if not self.gomoku_board.make_move(row, col, self.symbol):
            print("Invalid move by AI")
            return False

    def softmax(self, x, temperature=0.05):
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum(axis=0)
        
                
    def think_and_move(self, temperature=None, model=None, difficulty=None):
        if temperature is None:
            temperature = self.temperature
        if model is None:
            model = self.model
        if difficulty is None:
            difficulty = self.difficulty
        moves = self.gomoku_board.get_possible_moves()
        input_ = GomokuBoard.prepare_input(model, self.gomoku_board.get_possible_positions())
        position_values = -model.predict(input_).flatten() #il modello prende in input la posizione del giocatore che deve muovere, ciè l'altro se simulo la mia mossa
        move = self.choose_move_by_value(moves, position_values, temperature)
        if not self.gomoku_board.make_move(*move, self.symbol):
            print("Invalid move by AI")
            return None
        return move