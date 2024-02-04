from game_classes import GomokuBoard, Player, AIPlayer
from tensorflow import keras as k
import tkinter as tk
from tkinter import messagebox, Toplevel, Label, ttk
from PIL import Image, ImageTk
import os

def load_image(file_path, size=(60, 60)):
    image = Image.open(file_path)
    image = image.resize(size, Image.Resampling.LANCZOS)  # Aggiornato qui
    return ImageTk.PhotoImage(image)

def draw_move(canvas, row, col, player):
    x = col * cell_size + cell_size // 2
    y = row * cell_size + cell_size // 2

    if player == 'X':
        canvas.create_image(x, y, image=x_image)
    else:
        canvas.create_image(x, y, image=o_image)
        
def draw_grid(canvas, size, cell_size):
    for i in range(size):
        # Disegna le linee verticali
        canvas.create_line(i*cell_size, 0, i*cell_size, size*cell_size, fill = "black")
        # Disegna le linee orizzontali
        canvas.create_line(0, i*cell_size, size*cell_size, i*cell_size, fill = "black")

        
def on_canvas_click(event):
    col = event.x // cell_size
    row = event.y // cell_size

    if player.make_move(row, col) and not winner_announced:
        
        draw_move(game_canvas, row, col, 'X')
        draw_grid(game_canvas, game_size, cell_size)
        
        if board.is_last_winning():
            announce_winner("player")
            return
    
        # AI fa la sua mossa
        #ai_row, ai_col = get_move_by_policy(game, model, temperature)
        ai_row, ai_col = ai_player.think_and_move()
        draw_move(game_canvas, ai_row, ai_col, 'O')
        draw_grid(game_canvas, game_size, cell_size)
    
        if board.is_last_winning():
            announce_winner("AI")
            return
        else:
            # Aggiorna il valore della posizione dopo la mossa dell'AI
            value = board.value_of_position(model)[0][0]
            print("The new position value is", value)
            update_position_value_display(value)
            
def update_position_value_display(value):
    position_value_label.config(text=f"Position value estimate: {value:.3f}")

def update_wins_display():
    wins_label.config(text=f"Games won by player: {user_wins}/{total_games}")


def reset_game():
    global winner_announced, board, text_ids
    winner_announced = False
    board.reset()
    game_canvas.delete("all")
    text_ids = {}
    draw_grid(game_canvas, game_size, cell_size)
    update_position_value_display(0.0)
    update_wins_display()  # Aggiungi questa linea

def close_popup_and_reset(popup):
    popup.destroy()
    reset_game()
            
def announce_winner(player):
    global winner_announced, user_wins, total_games
    winner_announced = True
    total_games += 1  # Aggiorna il totale delle partite ogni volta che una partita finisce

    if player == "player":
        user_wins += 1  # Aggiorna le vittorie dell'utente se l'utente vince

    update_wins_display()  # Aggiorna il display delle vittorie

    # Crea una finestra popup
    popup = Toplevel(root)
    popup.title("Game Over")
    popup.geometry("300x120")

    # Posiziona la finestra popup al centro dello schermo
    x = root.winfo_x()
    y = root.winfo_y()
    popup.geometry("+%d+%d" % (x +200, y + 200))

    # Aggiungi un messaggio al popup
    message = f"The {player} wins"
    label = Label(popup, text=message, font=("Helvetica", 20, "bold"))
    label.pack(expand=True)

    # Aggiungi un bottone per chiudere il popup e inziziare una nuova partita
    close_button = tk.Button(popup, text="Close", command=lambda: close_popup_and_reset(popup))
    close_button.pack()

# Funzione per elencare le directory dentro "Models"
def find_models(directory="Models"):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def load_model_from_selection(event):
    global model
    model_name = model_selection_combobox.get()
    model_path = os.path.join("Models", model_name)
    model = k.models.load_model(model_path)
    print(f"Model {model_name} loaded.")
    model.summary()  # Mostra il sommario del modello se desiderato



model_name = "Models\DISPREZZOv318"
model= k.models.load_model(model_name)
print("Loading", model_name)
model.summary()
temperature = 0.005
board = GomokuBoard()
player = Player(board)
ai_player = AIPlayer(board, model)


winner_announced = False
user_wins = 0
total_games = 0
game_size = 15
cell_size = 50
text_ids = {}

def init_gui():
    global position_value_label, game_canvas, root, x_image, o_image, wins_label , model_selection_combobox 

    root = tk.Tk()
    root.title("Gomoku")
    x_image = load_image("Local\\ximage.png", size=(cell_size, cell_size))
    o_image = load_image("Local\\oimage.png", size=(cell_size, cell_size))

    # Invece di una sidebar, usa una frame per il valore della posizione nella parte superiore
    top_frame = tk.Frame(root, height=10, bg="lightgray")  # Puoi regolare l'altezza secondo necessità
    top_frame.pack(fill='x', side='top', expand=False)
    
    # Aggiungi qui il combobox per la selezione del modello
    model_names = find_models()
    model_selection_combobox = ttk.Combobox(top_frame, values=model_names)
    model_selection_combobox.pack(side='left', padx=10)
    model_selection_combobox.bind("<<ComboboxSelected>>", load_model_from_selection)
    model_selection_combobox.set("DISPREZZOv318")

    
    # Aggiungi un label per le vittorie dell'utente
    wins_label = tk.Label(top_frame, text="Games won by player: 0/0", bg="lightgray")
    wins_label.pack(side='left', padx=10)  # Posizionalo a sinistra

    position_value_label = tk.Label(top_frame, text="Position value estimate: 0", bg="lightgray")
    position_value_label.pack(side = "right", pady=5)

    # Il canvas viene posizionato sotto il frame superiore
    game_canvas = tk.Canvas(root, width=game_size*cell_size, height=game_size*cell_size, bg="white")
    game_canvas.bind("<Button-1>", on_canvas_click)
    draw_grid(game_canvas, game_size, cell_size)
    game_canvas.pack()

init_gui()
root.mainloop()
