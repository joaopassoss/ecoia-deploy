import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
# --- CONFIGURAÇÃO ---
MODEL_PATH = 'meu_modelo_ecoia.h5' # Deve ter o mesmo nome gerado no treino
CONFIDENCE_THRESHOLD = 0.70        # Confiança mínima (70%)

# LISTA DAS 12 CLASSES (Geralmente em ordem alfabética)
MY_CLASSES = [
    'battery',      # 0
    'biological',   # 1
    'brown-glass',  # 2
    'cardboard',    # 3
    'clothes',      # 4
    'green-glass',  # 5
    'metal',        # 6
    'paper',        # 7
    'plastic',      # 8
    'shoes',        # 9
    'trash',        # 10
    'white-glass'   # 11
]

class EcoBrain:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            print(f"Carregando modelo {MODEL_PATH}...")
            self.model = load_model(MODEL_PATH)
            print("Modelo carregado com sucesso!")
        else:
            print(f"AVISO: Arquivo '{MODEL_PATH}' não encontrado.")
            print("Você precisa rodar o 'python train_model.py' primeiro.")
            self.model = None

    def prepare_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    def get_info(self, class_name):
        name = class_name.lower()

        if 'battery' in name:
            return {"label": "Pilha / Bateria", "color": "Laranja", "desc": "PERIGO: Lixo tóxico. NUNCA no lixo comum. Leve a pontos de coleta."}
        
        if 'biological' in name:
            return {"label": "Orgânico", "color": "Marrom", "desc": "Lixeira MARROM. Restos de comida. Ideal para compostagem."}
        
        if 'glass' in name:
            return {"label": "Vidro", "color": "Verde", "desc": "Lixeira VERDE. Se quebrado, proteja com papelão para não ferir o coletor."}
        
        if 'cardboard' in name or 'paper' in name:
            return {"label": "Papel / Papelão", "color": "Azul", "desc": "Lixeira AZUL. Deve estar limpo e seco."}
        
        if 'clothes' in name or 'shoes' in name:
            return {"label": "Roupas / Calçados", "color": "Cinza / Doação", "desc": "Se bom estado: DOE. Se ruim: Lixo comum."}
        
        if 'metal' in name:
            return {"label": "Metal", "color": "Amarela", "desc": "Lixeira AMARELA. Amasse para ocupar menos espaço."}
        
        if 'plastic' in name:
            return {"label": "Plástico", "color": "Vermelha", "desc": "Lixeira VERMELHA. Lave levemente para tirar resíduos."}
        
        if 'trash' in name:
            return {"label": "Lixo Comum", "color": "Cinza", "desc": "Provavelmente não reciclável. Use a lixeira cinza."}

        return {"label": "Desconhecido", "color": "Cinza", "desc": "Não identificado com clareza."}

    def analyze(self, img_path):
        if self.model is None:
            return {"label": "Erro", "desc": "Modelo não carregado no servidor."}

        try:
            processed = self.prepare_image(img_path)
            preds = self.model.predict(processed)
            idx = np.argmax(preds[0])
            conf = preds[0][idx]
            
            # Validação de segurança do índice
            if idx >= len(MY_CLASSES):
                 return {"label": "Erro", "desc": "Classes incompatíveis."}

            class_name = MY_CLASSES[idx]
            
            if conf < CONFIDENCE_THRESHOLD:
                return {
                    "label": "Incerto",
                    "conf": f"{int(conf*100)}%",
                    "desc": "Tente tirar a foto mais de perto ou com mais luz."
                }

            info = self.get_info(class_name)
            
            return {
                "label": info['label'],
                "conf": f"{int(conf*100)}%",
                "desc": info['desc'],
                "color_code": info['color']
            }
        except Exception as e:
            print(f"Erro na análise: {e}")
            return {"label": "Erro", "desc": "Falha ao processar imagem."}