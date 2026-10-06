import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

print("==========================================")
print("EXERCÍCIO 1 - Diagnóstico Inicial")
print("==========================================")
df = pd.read_csv('Atividade Extra 01102026.csv', sep=';')

print(f"Quantidade total de linhas e colunas: {df.shape}")
print(f"Quantidade de ordens de serviço únicas: {df['id_ordem_servico'].nunique()}")
print(f"Quantidade de duplicados exatos: {df.duplicated().sum()}")
print(f"Quantidade de id_ordem_servico repetidos: {df['id_ordem_servico'].duplicated().sum()}")

campos = ['estado', 'forma_pagamento', 'canal_agendamento', 'dispositivo', 'tipo_servico', 'nivel_fidelidade']
for campo in campos:
    print(f"\nTop frequências para '{campo}':")
    print(df[campo].value_counts().head(5))


print("\n==========================================")
print("EXERCÍCIO 2 - Limpeza e Padronização")
print("==========================================")
df_limpo = df.copy()

# Remover duplicados exatos e de ID
antes = len(df_limpo)
df_limpo = df_limpo.drop_duplicates()
df_limpo = df_limpo.drop_duplicates(subset=['id_ordem_servico'], keep='first')
print(f"Registros antes da limpeza: {antes} | Após limpeza de duplicados: {len(df_limpo)}")

# Padronizações básicas
df_limpo['nome_cliente'] = df_limpo['nome_cliente'].str.strip().str.title()
df_limpo['email'] = df_limpo['email'].str.strip().str.lower()
df_limpo['telefone'] = df_limpo['telefone'].astype(str).apply(lambda x: re.sub(r'\D', '', x))
df_limpo['data_atendimento'] = pd.to_datetime(df_limpo['data_atendimento'], errors='coerce')
df_limpo['estado'] = df_limpo['estado'].str.strip().str.upper()


print("\n==========================================")
print("EXERCÍCIO 3 - Escalas Numéricas")
print("==========================================")
num_cols = ['idade_cliente', 'renda_mensal', 'valor_total']
print("Estatísticas (Antes):")
print(df_limpo[num_cols].agg(['min', 'max', 'mean', 'median']))

# Escaladores
df_limpo[['idade_minmax', 'renda_minmax', 'valor_minmax']] = MinMaxScaler().fit_transform(df_limpo[num_cols])
df_limpo[['idade_z', 'renda_z', 'valor_z']] = StandardScaler().fit_transform(df_limpo[num_cols])
df_limpo['renda_robusta'] = RobustScaler().fit_transform(df_limpo[['renda_mensal']])

print("\nTop 10 maiores rendas (Comparação):")
print(df_limpo.nlargest(10, 'renda_mensal')[['renda_mensal', 'renda_minmax', 'renda_z', 'renda_robusta']])


print("\n==========================================")
print("EXERCÍCIO 4 - Discretização")
print("==========================================")
# Faixa etária (Cortes fixos)
bins = [17, 24, 34, 44, 54, 64, 150]
labels_idade = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df_limpo['faixa_etaria'] = pd.cut(df_limpo['idade_cliente'], bins=bins, labels=labels_idade)

# Faixa de valor (Quantis)
df_limpo['faixa_valor'] = pd.qcut(df_limpo['valor_total'], q=4, labels=['Baixo', 'Medio-baixo', 'Medio-alto', 'Alto'])

print("Contagem faixa etária:\n", df_limpo['faixa_etaria'].value_counts())
print("\nContagem faixa valor (quantis):\n", df_limpo['faixa_valor'].value_counts())


print("\n==========================================")
print("EXERCÍCIO 5 - Binarização")
print("==========================================")
df_limpo['alto_valor'] = (df_limpo['valor_total'] > 1500).astype(int)
df_limpo['avaliacao_alta'] = (df_limpo['avaliacao'] >= 4).astype(int)

print("Distribuição 'alto_valor' (> R$ 1500):")
print(df_limpo['alto_valor'].value_counts(normalize=True) * 100)


print("\n==========================================")
print("EXERCÍCIO 6 - One-Hot Encoding")
print("==========================================")
cols_ohe = ['forma_pagamento', 'canal_agendamento', 'dispositivo']
df_limpo = pd.get_dummies(df_limpo, columns=cols_ohe, prefix=cols_ohe, dtype=int)
print(f"Total de colunas após One-Hot Encoding: {df_limpo.shape[1]}")


print("\n==========================================")
print("EXERCÍCIO 7 - Encoding Ordinal")
print("==========================================")
map_fidelidade = {'BRONZE': 0, 'PRATA': 1, 'OURO': 2, 'DIAMANTE': 3}
df_limpo['fidelidade_cod'] = df_limpo['nivel_fidelidade'].str.upper().map(map_fidelidade)
print(df_limpo[['nivel_fidelidade', 'fidelidade_cod']].head())


print("\n==========================================")
print("EXERCÍCIO 8 - Target Encoding sem Data Leakage")
print("==========================================")
X = df_limpo.drop(columns=['retornou_90d'])
y = df_limpo['retornou_90d']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

treino_aux = X_train.copy()
treino_aux['retornou_90d'] = y_train
media_modelo = treino_aux.groupby('id_modelo_veiculo')['retornou_90d'].mean()
media_global = y_train.mean()

X_test['modelo_veiculo_te'] = X_test['id_modelo_veiculo'].map(media_modelo).fillna(media_global)
X_train['modelo_veiculo_te'] = X_train['id_modelo_veiculo'].map(media_modelo).fillna(media_global)

print("Target encoding aplicado com sucesso sem vazar dados para o teste.")


print("\n==========================================")
print("EXERCÍCIO 9 - Base Final Pré-processada")
print("==========================================")
base_final = X_train.copy()
base_final['retornou_90d'] = y_train

print(f"Shape final da base de treino pré-processada: {base_final.shape}")
base_final.to_csv('base_oficina_preprocessada.csv', index=False, encoding='utf-8')
print("Arquivo 'base_oficina_preprocessada.csv' gerado com sucesso!")