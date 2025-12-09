# 1. Usa uma imagem base mais nova (Python 3.11) para evitar avisos de depreciação
FROM python:3.11-slim

# 2. Cria uma pasta de trabalho dentro do container
WORKDIR /app

# Configura variáveis de ambiente para limpar logs do TensorFlow
# -1 força o uso da CPU (evita erro de CUDA/GPU não encontrada)
ENV CUDA_VISIBLE_DEVICES="-1"
# Mudamos para '3' para silenciar o erro "failed call to cuInit" e avisos de compilação
ENV TF_CPP_MIN_LOG_LEVEL="3"

# 3. Copia o arquivo de dependências para o container
COPY requirements.txt .

# 4. Instala as bibliotecas
# Adicionamos 'gunicorn' explicitamente aqui para garantir que ele seja instalado
# mesmo que tenha sido esquecido no requirements.txt
RUN pip install --no-cache-dir gunicorn && pip install --no-cache-dir -r requirements.txt

# 5. Copia todo o projeto (incluindo os certificados gerados)
COPY . .

# 6. Informa ao Docker que o app vai usar a porta 5000
EXPOSE 5000

# 7. Roda com Gunicorn usando os certificados HTTPS
# Adicionamos --log-level error para esconder os avisos de certificado SSL auto-assinado
CMD ["gunicorn", "--log-level", "error", "--certfile=cert.pem", "--keyfile=key.pem", "-b", "0.0.0.0:5000", "app:app"]