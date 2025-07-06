import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import plotly.express as px
import folium
from streamlit_folium import folium_static
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- Configurações da página ---
st.set_page_config(
    page_title="Dash - Reclamações Carrefour",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Função para carregar o GeoDataFrame das localidades ---
@st.cache_data
def load_localidade_geodf(path):
    df = pd.read_csv(path, sep=',')
    # Converte texto -> lista -> shapely.geometry.Polygon
    df['coords'] = gpd.GeoSeries.from_wkt(df['POLYGON'])
    gdf = gpd.GeoDataFrame(df, geometry='coords', crs="EPSG:4326")
    return gdf

# --- Função para carregar séries temporais ---
@st.cache_data
def load_series_temporais(path):
    df = pd.read_csv(path, sep=',', index_col=0, parse_dates=True)
    df["TEMPO"] = pd.to_datetime(df['TEMPO'], format='%d-%m-%Y', errors='coerce')
    return df

# --- Carregamento dos dados ---
gdf_estados = load_localidade_geodf("..\datasets\gdf_estados.csv")
gdf_municipios = load_localidade_geodf("..\datasets\gdf_municipios.csv")
df_reclamacoes = load_series_temporais('..\datasets\RECLAMEAQUI_CARREFUOR_CLS.csv')


# --- Título do Dashboard ---
st.title("🛍️ Dashboard de Reclamações Carrefour")
st.subheader("Análise de Reclamações por Localidade")
st.markdown("""
Esta é uma análise interativa das reclamações recebidas pelo Carrefour, permitindo filtrar por localidade e período.
""")

# --- Sidebar com seletores ---
st.sidebar.title("Filtros 🔍")

# Seletor de periodo
st.sidebar.header("Selecione o período")
data_inicio = st.sidebar.date_input("Data de Início",
                                     value=df_reclamacoes["TEMPO"].min(),
                                     min_value=df_reclamacoes["TEMPO"].min(),
                                     max_value=df_reclamacoes["TEMPO"].max(),
                                     format="DD-MM-YYYY")
data_inicio = pd.to_datetime(data_inicio, format='%d-%m-%Y', errors='coerce')

data_fim = st.sidebar.date_input("Data de Fim", 
                                  value=df_reclamacoes["TEMPO"].max(), 
                                  min_value=df_reclamacoes["TEMPO"].min(), 
                                  max_value=df_reclamacoes["TEMPO"].max(), 
                                  format="DD-MM-YYYY")
data_fim = pd.to_datetime(data_fim, format='%d-%m-%Y', errors='coerce')

# Seletor de localidade
st.sidebar.header("Selecione a localidade")
opcoes_estados = sorted(gdf_municipios['NM_UF'].unique())
opcoes_completas = ['Todos'] + opcoes_estados
estado = st.sidebar.selectbox("Estado", options=opcoes_completas)

# Seletor de situação
st.sidebar.header("Selecione a situação")
situacao_selecionada = st.sidebar.multiselect("Situação", options=sorted(df_reclamacoes['STATUS'].unique().tolist()))

# Filtrar o DataFrame com base nas datas selecionadas
df_filtrado = df_reclamacoes[(df_reclamacoes["TEMPO"] >= data_inicio) 
                           & (df_reclamacoes["TEMPO"] <= data_fim)]

if estado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['NOME_UF'] == estado]

if len(situacao_selecionada) > 0:
    df_filtrado = df_filtrado[df_filtrado['STATUS'].isin(situacao_selecionada)]

# --- Gráficos temporais por reclamações ---
st.subheader(f"🔢 Reclamações por situação")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    container = st.container(border=True)
    container.badge("Resolvido", icon="✅", color="green")
    resolvido = df_filtrado['STATUS'].value_counts().get('Resolvido', 0)
    container.metric("", int(resolvido))

with col2:
    container = st.container(border=True)
    container.badge("Respondida", icon="📑", color="blue")
    respondida = df_filtrado['STATUS'].value_counts().get('Respondida', 0)
    container.metric("", int(respondida))

with col3:
    container = st.container(border=True)
    container.badge("Em réplica", icon="🗯️", color="violet")
    em_replica = df_filtrado['STATUS'].value_counts().get('Em réplica', 0)
    container.metric("", int(em_replica))

with col4:
    container = st.container(border=True)
    container.badge("Não Respondida", icon="‼️", color="orange")
    nao_respondida = df_filtrado['STATUS'].value_counts().get('Não respondida', 0)
    container.metric("", int(nao_respondida))

with col5:
    container = st.container(border=True)
    container.badge("Não Resolvido", icon="❌", color="red")
    nao_resolvido = df_filtrado['STATUS'].value_counts().get('Não resolvido', 0)
    container.metric("", int(nao_resolvido))

with col6:
    container = st.container(border=True)
    container.badge("Total", icon="📊", color="gray")
    total_reclamacoes = df_filtrado['STATUS'].count()
    container.metric("", int(total_reclamacoes))

# --- Gráfico de Linha Interativo ---
# Agrupar por DATA e STATUS, contando quantas reclamações existem
df_grouped = df_filtrado.groupby([df_filtrado['TEMPO'].dt.date, 'STATUS']).size().reset_index(name='quantidade')

df_grouped.columns = ['DATA', 'STATUS', 'quantidade']
# Converter a coluna DATA para datetime
df_grouped['DATA'] = pd.to_datetime(df_grouped['DATA'])

# Pivotar o DataFrame: linhas = datas, colunas = status, valores = quantidade
df_pivot = df_grouped.pivot(index='DATA', columns='STATUS', values='quantidade').fillna(0)

# Configurando o gráfico de linha
fig = px.line(
    df_pivot,
    title='📈 Reclamações por situação - Temporal',
    labels={
        "DATA": "Data Reclamação",      # Renomeia o eixo X
        "value": "Nº de Reclamações",      # Renomeia o eixo Y
        "STATUS": "Situação"               # Renomeia o título da legenda e das séries no hover
    }
)

# Ajustando legenda do gráfico
fig.update_layout(
    legend=dict(
        title="Situação",
        orientation="h", # 'h' para horizontal
        yanchor="top",
        y=1.1, # Coloca um pouco abaixo do gráfico
        xanchor="right",
        x=0.5
    )
)

# Configurando o eixo X para exibir datas
fig.update_xaxes(
    tickformat='%d/%m',  
    tickangle=-45        # Opcional: rotaciona os rótulos para não sobrepor
)

# Exibir o gráfico de linha interativo
st.plotly_chart(fig, use_container_width=True)


# **Frequência de reclamações por estado / município.**
st.subheader("📊 Frequência de reclamações por estado / município")

if estado != 'Todos':
    # Filtrar o DataFrame com base no estado selecionado
    df_estado = df_filtrado[df_filtrado['NOME_UF'] == estado]
    df_agrupado = df_estado.groupby(['MUNICIPIO']).size().reset_index(name='Qtd_Reclamacoes')
    df_ordenado = df_agrupado.sort_values(by='Qtd_Reclamacoes', ascending=True)
    st.write(f"Total de reclamações em {estado}: {df_ordenado['Qtd_Reclamacoes'].sum()}")
    st.bar_chart(df_ordenado, 
                 horizontal=True,
                 x_label='Município', 
                 x='MUNICIPIO', 
                 y_label='Quantidade de Reclamações', 
                 y='Qtd_Reclamacoes', 
                 use_container_width=True)

else:  
    # Agrupar por NOME_UF e contar as reclamações
    df_estado = df_filtrado.groupby(['NOME_UF']).size().reset_index(name='Qtd_Reclamacoes')
    df_ordenado = df_estado.sort_values(by='Qtd_Reclamacoes', ascending=True)
    st.bar_chart(df_ordenado, 
                 x_label='Estado', 
                 x='NOME_UF', 
                 y_label='Quantidade de Reclamações', 
                 y='Qtd_Reclamacoes', 
                 use_container_width=True)


# **Distribuição do tamanho dos textos** das reclamações (coluna `DESCRIÇÃO`).
st.subheader("📏 Distribuição do Tamanho dos Textos das Reclamações")

# Calcular o tamanho dos textos
df_filtrado['Tamanho_Texto'] = df_filtrado['DESCRICAO'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

# Agrupar por tamanho do texto e contar as reclamações
df_tamanho_texto = df_filtrado.groupby('Tamanho_Texto').size().reset_index(name='Qtd_Reclamacoes')

# Metricas gerais
st.markdown("##### Métricas Gerais")
col1, col2, col3 = st.columns(3)
tamanho_medio = int(df_filtrado['Tamanho_Texto'].mean())
tamanho_max = int(df_filtrado['Tamanho_Texto'].max())
tamanho_min = int(df_filtrado['Tamanho_Texto'].min())

col1.metric("Tamanho Mínimo", f"{tamanho_min} caracteres")
col2.metric("Tamanho Médio", f"{tamanho_medio} caracteres")
col3.metric("Tamanho Máximo", f"{tamanho_max} caracteres")

tamanho = st.select_slider(
    "Filtre pelo intervalo de tamanho do texto:",
    options=sorted(df_filtrado['Tamanho_Texto'].unique()),
    value=(tamanho_min, tamanho_max) # Valor inicial pega o mínimo e máximo
)

# Filtrar o DataFrame principal com base na seleção do slider
df_para_plotar = df_filtrado[
    (df_filtrado['Tamanho_Texto'] >= tamanho[0]) &
    (df_filtrado['Tamanho_Texto'] <= tamanho[1])
]


if not df_para_plotar.empty:
    # O histograma é o gráfico ideal para ver a distribuição de uma variável numérica.
    fig = px.histogram(
        df_para_plotar,
        x='Tamanho_Texto',
        nbins=30, # Define o número de "barras" ou "faixas" do histograma
        title='Frequência de Reclamações por Faixa de Tamanho de Texto',
        labels={'Tamanho_Texto': 'Tamanho do Texto (em caracteres)', 'count': 'Nº de Reclamações'}
    )

    fig.update_layout(
        bargap=0.1, # Espaço entre as barras
        yaxis_title="Nº de Reclamações" # Título do eixo Y
    )

    # Configurando o eixo X para exibir datas
    fig.update_xaxes(
        tickformat='%d/%m',  
        tickangle=-45        # Opcional: rotaciona os rótulos para não sobrepor
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Nenhuma reclamação encontrada no intervalo de tamanho selecionado.")

st.subheader("📈 Dispersão: Tamanho do Texto vs. Tempo")

if not df_filtrado.empty:
    # --- Criação do Gráfico de Dispersão com Plotly ---
    fig = px.scatter(
        df_filtrado,
        x='TEMPO',
        y='Tamanho_Texto',
        color='STATUS',  # Usa a coluna 'STATUS' para colorir os pontos
        title='Cada ponto representa uma reclamação individual. Use o filtro para analisar por status',
        labels={'Tamanho_Texto': 'Tamanho do Texto (caracteres)', 'TEMPO': 'Data da Reclamação'},
        # over_data=['DESCRICAO'], # Mostra a descrição no hover
    )

    # Customizações visuais
    fig.update_traces(
        marker=dict(size=5, opacity=0.7), # Deixa os pontos menores e um pouco transparentes
        selector=dict(mode='markers')
    )

    # Configurando o eixo X para exibir datas
    fig.update_xaxes(
        tickformat='%d/%m/%Y',  
        tickangle=-45        # Opcional: rotaciona os rótulos para não sobrepor
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Nenhum dado para exibir com os filtros selecionados.")

# **WordCloud** com as palavras mais frequentes nos textos das descrições.
st.subheader("📝 WordCloud - Palavras mais Frequentes nas Descrições")

# Obter a lista de stopwords em português
stopwords_portugues = stopwords.words('portuguese')

novas_stopwords = ["empresa", "comprei", "loja", "não", "pra", "tive", "minha", "nao", "apenas"
                   , "ter", "bem", "bom", "muito", "pouco", "mais", "menos", "ainda", "já", "agora", "hoje"
                   , "ontem", "amanhã", "sempre", "nunca", "todo", "toda", "todos", "todas", "algum", "alguma"
                   , "alguns", "algumas", "cada", "qualquer", "quaisquer", "quem", "onde", "quando", "como", "porque"
                   , "dia", "dias", "reclame", "aqui", "problema", "já", "pois", "outro", "outra", "Carrefour"
                   , "lá", "dentro", "toda", "fiz", "R", "vez", "vou", "tudo", "porém", "então", "assim", "havia", "disse"
                   , "compra", "produto", "produtos", "serviço", "serviços", "cliente", "clientes", "deu", "falou", "sobre"
                   , "aí", "q", "após", "aí", "ir", "mesma", "passar", "forma", "levar", "comprar", "pedi", "nenhum", "volta"
                   , "voltar", "fazer", "Além", "fazendo", "favor", "deram", "chegou", "chegar", "ir", "vir", "iria", "quero"
                   , "queria", "querer", "ser", "caso", "casa", "informar", "informou", "informe", "ano", "reais", "pagar"
                   , "sendo", "nota", "falta", "faltar", "data", "novamente", "poder", "poderia", "pessoa", "absurdo"
                   , "momento", "Editado", "Editar", "hora", "falar"]

for palavra in novas_stopwords:
    stopwords_portugues.append(palavra)
# Concatenar todas as descrições em uma única string
textos = ' '.join(df_filtrado['DESCRICAO'].dropna().astype(str).tolist())

texto = " ".join(df_filtrado['DESCRICAO'].astype(str).tolist())
print(len(texto)) 
print(texto[:100])  

# Gerar a nuvem de palavras
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=stopwords_portugues,
    colormap='viridis', 
    max_words=50
).generate(texto)

# Plotar a WordCloud
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')  # Remove os eixos
st.pyplot(fig)


# **Mapa do Brasil com heatmap** mostrando a quantidade de reclamações por **ano**, com granularidade por **estado ou município**.
#  > O mapa **deve conter um seletor para o ano** que será visualizado.
st.subheader("🗺️ Mapa de calor - Reclamações por Estado / Município")
st.markdown("Para apresentar as informações por municípios, selecione um estado nos filtros laterais")

col1, col2, col3 = st.columns(3)

with col1:
    # Seletor de ano para mapa
    opcoes_anos = sorted(df_filtrado['ANO'].unique())
    todas_opcoes = ['Todos'] + opcoes_anos
    ano = st.selectbox("Selecione o ano:", options=todas_opcoes)

if ano != 'Todos':
    df_mapa = df_filtrado[df_filtrado['ANO'] == ano]
else:
    df_mapa = df_filtrado.copy()


if estado != 'Todos':
    df_mapa = df_mapa[df_mapa['NOME_UF'] == estado]
    gdf_municipios = gdf_municipios[gdf_municipios["NM_UF"] == estado]

    # Centralizar o mapa na área de interesse
    mapa = folium.Map(location=[gdf_municipios.geometry.centroid.y.mean(), gdf_municipios.geometry.centroid.x.mean()], zoom_start=6.3)

    # Agrupando informações dos estados
    df_mapa = df_mapa.groupby(['MUNICIPIO']).size().reset_index(name='Qtd_Reclamacoes')

    # Unificando com os dados de localização de cada estado
    gdf_final = gdf_municipios.merge(df_mapa, left_on='NM_MUN', right_on='MUNICIPIO', how='left')

    # Adicionando as informações no mapa
    choropleth = folium.Choropleth(
        geo_data=gdf_final,
        name='choropleth',
        data=gdf_final,
        columns=['MUNICIPIO', 'Qtd_Reclamacoes'],
        key_on='feature.properties.NM_MUN', # Chave no GeoJSON para conectar os dados
        fill_color='YlOrRd', # Nome do colormap (escala de cores)
        nan_fill_color='gray',
        nan_fill_opacity=0.4,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Quantidade de Reclamações',
        highlight=True, # Destaca a área ao passar o mouse
    )

    choropleth.add_to(mapa)

    # --- Adicionar Tooltips Interativos (Informação no Hover) ---
    tooltip = folium.features.GeoJsonTooltip(
        fields=['NM_MUN', 'AREA_KM2', 'Qtd_Reclamacoes'],
        aliases=['Município:', 'Área (Km²):', 'N° de Reclamações:'],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
        sticky=True
    )

    tooltip.add_to(choropleth.geojson)

else:

    # Centralizar o mapa na área de interesse
    mapa = folium.Map(location=[gdf_estados.geometry.centroid.y.mean(), gdf_estados.geometry.centroid.x.mean()], zoom_start=4.3)

    # Agrupando informações dos estados
    df_mapa = df_mapa.groupby(['NOME_UF']).size().reset_index(name='Qtd_Reclamacoes')

    # Unificando com os dados de localização de cada estado
    gdf_final = gdf_estados.merge(df_mapa, left_on='NM_UF', right_on='NOME_UF', how='left')

    # Adicionando as informações no mapa
    choropleth = folium.Choropleth(
        geo_data=gdf_final,
        name='choropleth',
        data=gdf_final,
        columns=['NOME_UF', 'Qtd_Reclamacoes'],
        key_on='feature.properties.NM_UF', # Chave no GeoJSON para conectar os dados
        fill_color='YlOrRd', # Nome do colormap (escala de cores)
        nan_fill_color='gray',
        nan_fill_opacity=0.4,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Quantidade de Reclamações',
        bins=[1, 20, 40, 80, 160, 320, 660],
        highlight=True, # Destaca a área ao passar o mouse
    )

    choropleth.add_to(mapa)

    # --- Adicionar Tooltips Interativos (Informação no Hover) ---
    tooltip = folium.features.GeoJsonTooltip(
        fields=['NM_UF', 'AREA_KM2', 'Qtd_Reclamacoes'],
        aliases=['Estado:', 'Área (Km²):', 'N° de Reclamações:'],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
        sticky=True
    )

    tooltip.add_to(choropleth.geojson)

    
folium_static(mapa, width=1000, height=600)


# ### **Os gráficos devem ser interativos e filtráveis com seletores de:**

# # Estado
# # Status
# # Faixa de tamanho do texto da reclamação