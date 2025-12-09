from OpenSSL import crypto

def generate_self_signed_cert():
    # Cria um par de chaves (Pública e Privada)
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)

    # Cria o certificado auto-assinado
    cert = crypto.X509()
    cert.get_subject().C = "BR"
    cert.get_subject().ST = "Estado"
    cert.get_subject().L = "Cidade"
    cert.get_subject().O = "EcoIA Project"
    cert.get_subject().OU = "Dev"
    cert.get_subject().CN = "localhost"
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60) # Válido por 1 ano
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # Salva os arquivos no disco
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

    print("✅ Sucesso! Arquivos 'cert.pem' e 'key.pem' gerados.")

if __name__ == "__main__":
    try:
        generate_self_signed_cert()
    except ImportError:
        print("❌ Erro: Biblioteca 'pyopenssl' não instalada.")
        print("Rode: pip install pyopenssl")