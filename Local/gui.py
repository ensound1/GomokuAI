from game_classes import GomokuBoard, Player, AIPlayer
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox, Toplevel, Label
from PIL import Image, ImageTk

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

def reset_game():
    global winner_announced, board, text_ids
    winner_announced = False
    board.reset()
    game_canvas.delete("all")
    text_ids = {}
    draw_grid(game_canvas, game_size, cell_size)

def close_popup_and_reset(popup):
    popup.destroy()
    reset_game()
            
def announce_winner(player):
    global winner_announced
    winner_announced = True

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
    close_button = tk.Button(popup, text="Chiudi", command=lambda: close_popup_and_reset(popup))
    close_button.pack()



model_name = "Local\DISPREZZOv210"
model= tf.keras.models.load_model(model_name)
print("Carico", model_name)
model.summary()
temperature = 0.005
board = GomokuBoard()
player = Player(board)
ai_player = AIPlayer(board, model)


winner_announced = False
game_size = 15
cell_size = 50
text_ids = {}

root = tk.Tk()
root.title("Gomoku")
x_image = load_image("Local\\ximage.png", size=(cell_size, cell_size))
o_image = load_image("Local\\oimage.png", size=(cell_size, cell_size))

game_canvas = tk.Canvas(root, width=game_size*cell_size, height=game_size*cell_size, bg ="white")
game_canvas.bind("<Button-1>", on_canvas_click)

#show_probabilities(game,model)

draw_grid(game_canvas, game_size, cell_size)

game_canvas.pack()

root.mainloop()