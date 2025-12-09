import os

# Desativa logs chatos do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# --- CONFIGURAÇÃO DA CHAVE ---
# ⚠️ COLE SUA CHAVE AQUI DENTRO DAS ASPAS:
GEMINI_API_KEY = "AIzaSyDD1KYwEw2LmqidnIoENkVr63BqoAfMihY" 

# --- TENTATIVA DE IMPORTAÇÃO COM LOG DE ERRO ---
print("-" * 50)
print("TENTANDO IMPORTAR GOOGLE GENERATIVE AI...")
try:
    import google.generativeai as genai
    print("✅ SUCESSO: Biblioteca importada!")
except ImportError as e:
    genai = None
    print(f"❌ ERRO DE IMPORTAÇÃO: {e}")
except Exception as e:
    genai = None
    print(f"❌ OUTRO ERRO ESTRANHO: {e}")
print("-" * 50)

# --- CONFIGURAÇÃO ---
MODEL_PATH = 'meu_modelo_ecoia.h5'
CONFIDENCE_THRESHOLD = 0.60
gemini_model = None

def setup_gemini():
    """ Tenta conectar ao Gemini buscando modelos disponíveis automaticamente """
    global gemini_model
    
    if not genai:
        return "Erro: Falha na importação da biblioteca (veja o terminal)."
    
    if "SUA_API_KEY" in GEMINI_API_KEY:
        return "Erro: Você esqueceu de colar a API Key no arquivo ai_manager.py"
        
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 1. TENTA LISTAR MODELOS DISPONÍVEIS NA SUA CONTA
        print("🔍 Buscando modelos disponíveis na sua conta...")
        modelos_disponiveis = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    modelos_disponiveis.append(m.name)
        except Exception as e:
            print(f"⚠️ Erro ao listar modelos: {e}")

        # Se achou modelos, tenta usar o primeiro da lista que seja 'gemini'
        if modelos_disponiveis:
            print(f"📋 Modelos encontrados: {modelos_disponiveis}")
            
            # Prioriza modelos 'flash' ou 'pro' se houver
            melhor_modelo = None
            for m in modelos_disponiveis:
                if 'flash' in m: 
                    melhor_modelo = m
                    break
            
            # Se não achou flash, pega qualquer gemini
            if not melhor_modelo:
                for m in modelos_disponiveis:
                    if 'gemini' in m:
                        melhor_modelo = m
                        break
            
            # Se ainda não tem, pega o primeiro da lista
            if not melhor_modelo:
                melhor_modelo = modelos_disponiveis[0]

            # Tenta conectar
            try:
                print(f"👉 Tentando conectar ao modelo automático: {melhor_modelo}...")
                temp_model = genai.GenerativeModel(melhor_modelo)
                temp_model.generate_content("Oi", generation_config={'max_output_tokens': 1})
                gemini_model = temp_model
                print(f"✅ Conectado com sucesso ao: {melhor_modelo}")
                return "OK"
            except Exception as e:
                print(f"❌ Falha no modelo automático {melhor_modelo}: {e}")

        # 2. FALLBACK MANUAL (Se a lista falhar)
        print("⚠️ Lista automática falhou. Tentando nomes padrões...")
        modelos_para_tentar = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'models/gemini-1.5-flash', # Às vezes precisa do prefixo
            'gemini-1.0-pro',
            'gemini-pro'
        ]
        
        for modelo in modelos_para_tentar:
            try:
                print(f"Tentando: {modelo}...")
                temp_model = genai.GenerativeModel(modelo)
                temp_model.generate_content("Oi", generation_config={'max_output_tokens': 1})
                gemini_model = temp_model
                print(f"✅ Conectado com sucesso ao: {modelo}")
                return "OK"
            except Exception:
                continue 
        
        return "Erro: Nenhum modelo do Gemini respondeu."

    except Exception as e:
        return f"Erro de Conexão Geral: {str(e)}"

# Inicialização
init_status = setup_gemini()
if init_status == "OK":
    print("\n🚀 IA GENERATIVA PRONTA PARA USO!\n")
else:
    print(f"\n⚠️ STATUS DA IA: {init_status}\n")


MY_CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

class EcoBrain:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = load_model(MODEL_PATH)
            print("👁️ Modelo de Visão Carregado.")
        else:
            print(f"❌ ERRO: '{MODEL_PATH}' não encontrado.")
            self.model = None

    def prepare_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    def generate_explanation(self, class_name):
        global gemini_model
        
        if not gemini_model:
            status = setup_gemini()
            if status != "OK":
                return f"IA Indisponível. {status}"

        prompt = f"""
        Aja como um especialista em sustentabilidade do Brasil.
        Identificamos um resíduo: '{class_name}'.
        
        1. Em qual cor de lixeira descartar? (Seja direto).
        2. Dê uma curiosidade ou alerta de segurança importante.
        
        Use emojis. Texto curto (max 40 palavras).
        """
        
        try:
            # Tenta gerar o conteúdo
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Erro ao gerar com o modelo atual: {e}")
            # Se der erro durante a geração (ex: timeout), tentamos reconfigurar
            print("Tentando reconfigurar conexão com Gemini...")
            setup_gemini()
            return "Dica indisponível no momento, mas descarte com consciência!"

    def get_static_color(self, class_name):
        name = class_name.lower()
        if 'battery' in name: return "Laranja"
        if 'biological' in name: return "Marrom"
        if 'glass' in name: return "Verde"
        if 'paper' in name or 'cardboard' in name: return "Azul"
        if 'metal' in name: return "Amarela"
        if 'plastic' in name: return "Vermelha"
        return "Cinza"

    def analyze(self, img_path):
        if not self.model: return {"label": "Erro", "desc": "Modelo offline."}

        try:
            processed = self.prepare_image(img_path)
            preds = self.model.predict(processed)
            idx = np.argmax(preds[0])
            conf = preds[0][idx]
            
            if idx >= len(MY_CLASSES): return {"label": "Erro", "desc": "Classe inválida."}
            class_name = MY_CLASSES[idx]
            conf_percent = int(conf * 100)

            if conf < CONFIDENCE_THRESHOLD:
                return {
                    "label": "Incerto",
                    "conf": f"{conf_percent}%",
                    "desc": "Imagem pouco clara. Tente aproximar."
                }

            ai_description = self.generate_explanation(class_name)
            ui_color = self.get_static_color(class_name)

            display_label = class_name.replace("-", " ").title()
            translation = {
                "Battery": "Pilha / Bateria", "Biological": "Orgânico",
                "Trash": "Lixo Comum", "Cardboard": "Papelão",
                "Clothes": "Roupas", "Glass": "Vidro", "Metal": "Metal",
                "Paper": "Papel", "Plastic": "Plástico", "Shoes": "Sapatos"
            }
            
            for en, pt in translation.items():
                if en in display_label: display_label = pt

            return {
                "label": display_label,
                "conf": f"{conf_percent}%",
                "desc": ai_description,
                "color_code": ui_color
            }

        except Exception as e:
            print(f"Erro: {e}")
            return {"label": "Erro", "desc": "Falha técnica."}