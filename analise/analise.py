# ==============================================================================
# PROJETO DE ANÁLISE DE DADOS - PROBABILIDADE E ESTATÍSTICA
#
#SP3123766

# 1. IMPORTAÇÃO DE BIBLIOTECAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import geopandas as gpd
import unicodedata
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 2. CONFIGURAÇÃO E CARREGAMENTO DOS DADOS
NOME_ARQUIVO_MORTALIDADE = 'DO22OPEN.csv'
NOME_ARQUIVO_CENSO = 'tabela6591_con_urb.xlsx'
CAMINHO_DADOS = 'dados/'

print("Iniciando análise... Carregando dados...")
try:
    df_mortalidade = pd.read_csv(CAMINHO_DADOS + NOME_ARQUIVO_MORTALIDADE, sep=';', encoding='latin1', low_memory=False)
    df_censo = pd.read_excel(CAMINHO_DADOS + NOME_ARQUIVO_CENSO)
except FileNotFoundError:
    print(f"ERRO: Arquivos não encontrados no diretório '{CAMINHO_DADOS}'. Verifique os nomes e o caminho.")
    exit()

# 3. PREPARAÇÃO E LIMPEZA DOS DADOS DE MORTALIDADE
print("Limpando e preparando os dados de mortalidade...")
colunas_mortalidade = ['CAUSABAS', 'IDADE', 'SEXO', 'RACACOR', 'ESTCIV', 'ESC', 'CODMUNRES']
df_mort_selecionado = df_mortalidade[colunas_mortalidade].copy()
df_mort_selecionado.rename(columns={'CODMUNRES': 'cod_municipio'}, inplace=True)

df_mort_selecionado['IDADE'] = df_mort_selecionado['IDADE'].astype(str)
df_mort_selecionado['IDADE_ANOS'] = np.where(df_mort_selecionado['IDADE'].str.startswith('4'), df_mort_selecionado['IDADE'].str[1:], np.nan)

df_mort_selecionado.dropna(subset=['IDADE_ANOS', 'CAUSABAS', 'SEXO', 'cod_municipio'], inplace=True)
df_mort_selecionado['IDADE_ANOS'] = pd.to_numeric(df_mort_selecionado['IDADE_ANOS'], errors='coerce')
df_mort_selecionado.dropna(subset=['IDADE_ANOS'], inplace=True)
df_mort_selecionado['IDADE_ANOS'] = df_mort_selecionado['IDADE_ANOS'].astype(int)
df_mort_selecionado['CAPITULO_CID'] = df_mort_selecionado['CAUSABAS'].str[0]
df_mort_selecionado['cod_municipio_6dig'] = df_mort_selecionado['cod_municipio'].astype(str)

# 4. ANÁLISE DO CENSO E CRIAÇÃO DOS GRUPOS DE INFRAESTRUTURA
print("Criando grupos de cidades a partir dos dados do Censo...")
df_censo_analise = df_censo.copy()
df_censo_analise['NOME_CIDADE'] = df_censo_analise['Concentração Urbana'].str.split('/').str[0].str.strip()
df_censo_analise['SIGLA_UF'] = df_censo_analise['Concentração Urbana'].str.split('/', n=1).str[1]
df_censo_analise.dropna(subset=['SIGLA_UF'], inplace=True)

colunas_infra = {
    'perc_pavimentada': 'Via pavimentada - Existe',
    'perc_iluminacao': 'Existência de iluminação pública - Existe',
    'perc_calcada': 'Existência de calçada / passeio - Existe'
}
for nome_perc, coluna_origem in colunas_infra.items():
    df_censo_analise[nome_perc] = (df_censo_analise[coluna_origem] / df_censo_analise['Total']).fillna(0)

df_censo_analise['INDICE_INFRA'] = df_censo_analise[['perc_pavimentada', 'perc_iluminacao', 'perc_calcada']].mean(axis=1)
mediana_indice = df_censo_analise['INDICE_INFRA'].median()
lista_cidades_melhor_infra = df_censo_analise[df_censo_analise['INDICE_INFRA'] >= mediana_indice]['NOME_CIDADE'].tolist()
lista_cidades_pior_infra = df_censo_analise[df_censo_analise['INDICE_INFRA'] < mediana_indice]['NOME_CIDADE'].tolist()

# 5. ENRIQUECIMENTO E FINALIZAÇÃO DO DATAFRAME PRINCIPAL
print("Enriquecendo dados de mortalidade para análise final...")
try:
    url_municipios = 'https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/csv/municipios.csv'
    df_municipios = pd.read_csv(url_municipios)
    df_municipios_lookup = df_municipios[['codigo_ibge', 'nome']].copy()
    df_municipios_lookup['cod_municipio_6dig'] = df_municipios_lookup['codigo_ibge'].astype(str).str[:6]
    df_municipios_lookup.rename(columns={'nome': 'NOME_MUNICIPIO'}, inplace=True)
    df_mort_final = pd.merge(df_mort_selecionado, df_municipios_lookup[['cod_municipio_6dig', 'NOME_MUNICIPIO']], on='cod_municipio_6dig', how='inner')
except Exception:
    print("AVISO: Falha ao carregar nomes de municípios da internet. A análise prosseguirá sem os nomes.")
    df_mort_final = df_mort_selecionado.copy()
    df_mort_final['NOME_MUNICIPIO'] = 'N/A'

def normalizar_texto(texto):
    nfkd_form = unicodedata.normalize('NFD', str(texto))
    texto_sem_acentos = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return texto_sem_acentos.upper()

df_mort_final['NOME_MUNICIPIO_NORM'] = df_mort_final['NOME_MUNICIPIO'].apply(normalizar_texto)
lista_melhor_infra_norm = [normalizar_texto(cidade) for cidade in lista_cidades_melhor_infra]
lista_pior_infra_norm = [normalizar_texto(cidade) for cidade in lista_cidades_pior_infra]

df_mort_final['GRUPO_INFRA'] = 'Não Classificado'
df_mort_final.loc[df_mort_final['NOME_MUNICIPIO_NORM'].isin(lista_melhor_infra_norm), 'GRUPO_INFRA'] = 'Melhor Infra'
df_mort_final.loc[df_mort_final['NOME_MUNICIPIO_NORM'].isin(lista_pior_infra_norm), 'GRUPO_INFRA'] = 'Pior Infra'

# 6. GERAÇÃO DE SAÍDAS PARA O RELATÓRIO
print("\nGerando saídas (tabelas e gráficos)...")
sns.set_style("whitegrid")

# 6.1 - Saídas da Análise Descritiva
tabela_idade = df_mort_final['IDADE_ANOS'].describe()
print("\n\n--- TABELA 1: ESTATÍSTICAS DESCRITIVAS DA IDADE NO ÓBITO ---")
print("-" * 50)
for indice, valor in tabela_idade.items():
    print(f"{indice:<10s} | {valor:>15.2f}")
print("-" * 50)

plt.figure(figsize=(12, 6))
sns.histplot(data=df_mort_final, x='IDADE_ANOS', bins=40, kde=True)
plt.title('Distribuição da Idade no Óbito')
plt.xlabel('Idade (em anos)'); plt.ylabel('Número de Óbitos')
plt.savefig('grafico_01_idade_distribuicao.png', dpi=300, bbox_inches='tight')
plt.close() # Fecha a figura para não exibir se rodado em loop

# 6.2 - Saídas da Análise Comparativa
# Tabela 3
idades_melhor_infra = df_mort_final[df_mort_final['GRUPO_INFRA'] == 'Melhor Infra']['IDADE_ANOS']
idades_pior_infra = df_mort_final[df_mort_final['GRUPO_INFRA'] == 'Pior Infra']['IDADE_ANOS']
t_stat_geral, p_value_geral = stats.ttest_ind(idades_melhor_infra, idades_pior_infra, equal_var=False)
print("\n\n--- TABELA 3: RESULTADOS DO TESTE T - IDADE MÉDIA GERAL VS. INFRAESTRUTURA ---")
print("-" * 80)
print(f"{'Grupo':<20} | {'Nº de Óbitos':>15} | {'Idade Média (anos)':>20} |")
print("-" * 80)
print(f"{'Melhor Infraestrutura':<20} | {len(idades_melhor_infra):>15,d} | {idades_melhor_infra.mean():>20.2f} |")
print(f"{'Pior Infraestrutura':<20} | {len(idades_pior_infra):>15,d} | {idades_pior_infra.mean():>20.2f} |")
print("-" * 80)
print(f"Estatística t = {t_stat_geral:.4f} | Valor-p = {p_value_geral if p_value_geral > 0.001 else '< 0.001'}")
print("-" * 80)

# 6.3 - Saída da Análise Geoespacial
try:
    print("\nGerando mapa comparativo...")
    url_mapa_brasil = 'https://raw.githubusercontent.com/giuliano-macedo/geodata-br-states/refs/heads/main/geojson/br_states.json'
    gdf_brasil = gpd.read_file(url_mapa_brasil)
    gdf_brasil.rename(columns={'SIGLA': 'SIGLA_UF'}, inplace=True)
    
    dados_uf = [
        {'COD_ESTADO': 11, 'SIGLA_UF': 'RO'}, {'COD_ESTADO': 12, 'SIGLA_UF': 'AC'}, {'COD_ESTADO': 13, 'SIGLA_UF': 'AM'},
        {'COD_ESTADO': 14, 'SIGLA_UF': 'RR'}, {'COD_ESTADO': 15, 'SIGLA_UF': 'PA'}, {'COD_ESTADO': 16, 'SIGLA_UF': 'AP'},
        {'COD_ESTADO': 17, 'SIGLA_UF': 'TO'}, {'COD_ESTADO': 21, 'SIGLA_UF': 'MA'}, {'COD_ESTADO': 22, 'SIGLA_UF': 'PI'},
        {'COD_ESTADO': 23, 'SIGLA_UF': 'CE'}, {'COD_ESTADO': 24, 'SIGLA_UF': 'RN'}, {'COD_ESTADO': 25, 'SIGLA_UF': 'PB'},
        {'COD_ESTADO': 26, 'SIGLA_UF': 'PE'}, {'COD_ESTADO': 27, 'SIGLA_UF': 'AL'}, {'COD_ESTADO': 28, 'SIGLA_UF': 'SE'},
        {'COD_ESTADO': 29, 'SIGLA_UF': 'BA'}, {'COD_ESTADO': 31, 'SIGLA_UF': 'MG'}, {'COD_ESTADO': 32, 'SIGLA_UF': 'ES'},
        {'COD_ESTADO': 33, 'SIGLA_UF': 'RJ'}, {'COD_ESTADO': 35, 'SIGLA_UF': 'SP'}, {'COD_ESTADO': 41, 'SIGLA_UF': 'PR'},
        {'COD_ESTADO': 42, 'SIGLA_UF': 'SC'}, {'COD_ESTADO': 43, 'SIGLA_UF': 'RS'}, {'COD_ESTADO': 50, 'SIGLA_UF': 'MS'},
        {'COD_ESTADO': 51, 'SIGLA_UF': 'MT'}, {'COD_ESTADO': 52, 'SIGLA_UF': 'GO'}, {'COD_ESTADO': 53, 'SIGLA_UF': 'DF'}
    ]
    df_uf_lookup = pd.DataFrame(dados_uf)
    
    df_mort_final['COD_ESTADO'] = df_mort_final['cod_municipio_6dig'].str[:2].astype(int)
    estatisticas_por_estado = df_mort_final.groupby('COD_ESTADO')['IDADE_ANOS'].mean().reset_index()
    estatisticas_com_sigla = estatisticas_por_estado.merge(df_uf_lookup, on='COD_ESTADO', how='left')
    
    infra_por_estado = df_censo_analise.groupby('SIGLA_UF')['INDICE_INFRA'].mean().reset_index()
    
    mapa_para_plot = gdf_brasil.merge(estatisticas_com_sigla, on='SIGLA_UF', how='left').merge(infra_por_estado, on='SIGLA_UF', how='left')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle('Análise Comparativa Geográfica: Mortalidade e Infraestrutura', fontsize=20)
    
    mapa_para_plot.plot(column='IDADE_ANOS', cmap='viridis', ax=ax1, legend=True, missing_kwds={"color": "lightgrey"},
                        legend_kwds={'label': "Idade Média de Óbito", 'orientation': "horizontal"})
    ax1.set_title('Idade Média de Óbito por Estado', fontsize=16); ax1.axis('off')

    mapa_para_plot.plot(column='INDICE_INFRA', cmap='plasma', ax=ax2, legend=True, missing_kwds={"color": "lightgrey"},
                        legend_kwds={'label': "Índice de Infraestrutura (0 a 1)", 'orientation': "horizontal"})
    ax2.set_title('Índice Médio de Infraestrutura Urbana por Estado', fontsize=16); ax2.axis('off')

    plt.savefig('mapa_02_comparativo.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"\nAVISO: Não foi possível gerar o mapa geográfico. Erro: {e}")

print("\n\nAnálise concluída. Saídas geradas com sucesso.")