from flask import Flask, request, jsonify, render_template
from game_classes import GomokuBoard, AIPlayer, Player
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Carica il modello di TensorFlow
model = load_model("Models/DISPREZZOv100")

# Inizializza il gioco Gomoku
board = GomokuBoard()
player = Player(board)
ai_player = AIPlayer(board, model=model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    row, col = data['row'], data['col']
    if not player.make_move(row, col):
        print("Attempted invalid move!")
        return jsonify({"error": "Invalid move", "row": None, "col": None, "winner": None})
    if board.is_last_winning():
        print("The player has won!")
        return jsonify({"row":row, "col":col, "winner": "player"})
    else:
        ai_row, ai_col = ai_player.think_and_move()
        if board.is_last_winning():
            return jsonify({"row":row, "col":col, "ai_row": ai_row, "ai_col": ai_col, "winner": "AI"})
        else:
            return jsonify({"row":row, "col":col, "ai_row": ai_row, "ai_col": ai_col, "winner": None})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
