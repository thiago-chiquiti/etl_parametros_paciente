import pandas as pd
import google.generativeai as genai
import getpass
import time

# 1. Solicitação da chave da API
print("--- Sistema de Análise de Sinais Vitais ---")
api_key = getpass.getpass("Insira sua Google API Key: ")

try:
    # Configura a API
    genai.configure(api_key=api_key)

    # Tenta encontrar o modelo correto de forma automática
    print("Verificando modelos disponíveis...")
    modelos = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    if not modelos:
        raise Exception("Nenhum modelo compatível encontrado. Verifique sua chave.")
    
    # Prioriza o flash, se não houver, pega o primeiro da lista
    model_name = 'models/gemini-1.5-flash' if 'models/gemini-1.5-flash' in modelos else modelos[0]
    print(f"Conectado ao modelo: {model_name}")
    
    model = genai.GenerativeModel(model_name)

    # 2. Leitura do arquivo CSV
    df = pd.read_csv('SSVV_pacientes.csv')

    # 3. Função para interagir com o Gemini
    def analisar_paciente(row):
        # Descrevendo o paciente para o agente de IA
        dados = (f"Idade: {row['Idade']}, Sexo: {row['Sexo']}, "
                 f"FC: {row['FC']} bpm, FR: {row['FR']} mpm, "
                 f"SatO2: {row['SatO2']}%, Temp: {row['Temp']}°C")
        
        prompt = f"Aja como um médico. Avalie: {dados}. Dê um diagnóstico de até 3 palavras."
        
        try:
            # Chamada da IA
            response = model.generate_content(prompt)
            time.sleep(1.5) # Pausa para evitar bloqueios
            return response.text.strip()
        except Exception as e:
            return f"Erro na linha: {e}"

    # 4. Processamento
    print(f"Analisando {len(df)} pacientes. Isso pode levar alguns minutos...")
    
    # Cria uma coluna de diagnóstico
    df['Diagnostico_IA'] = df.apply(analisar_paciente, axis=1)

    # 5. Salva o resultado em um novo CSV
    output_file = 'SSVV_pacientes_analisados.csv'
    df.to_csv(output_file, index=False)

    print("\n--- SUCESSO! ---")
    print(df[['Nome', 'Diagnostico_IA']].head())
    print(f"\nResultado salvo em: {output_file}")

except Exception as e:
    print(f"\nERRO CRÍTICO: {e}")