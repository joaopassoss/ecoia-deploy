import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, render_template, request, jsonify
from ai_manager import EcoBrain  # Importa nossa classe de IA

app = Flask(__name__)

# Configurações de Pasta
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inicializa a IA (Carrega na memória ao iniciar o app)
# Isso evita recarregar o modelo a cada clique do usuário
brain = EcoBrain()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Arquivo vazio'}), 400

    if file:
        # Salva o arquivo enviado
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_target.jpg')
        file.save(filepath)

        try:
            # CHAMA NOSSA CLASSE DE IA
            resultado = brain.analyze(filepath)
            
            # Remove o arquivo temporário para economizar espaço
            # os.remove(filepath) 
            
            return jsonify(resultado)
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return jsonify({'error': 'Erro ao processar imagem'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)