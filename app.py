from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def fast_sammon(X, n_components=2):
    # Fall back to t-SNE if sammon isn't available
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(X)


def pca(X, n_components=2):
    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sorted_idx]
    eig_vals = eig_vals[sorted_idx]
    eig_vecs = eig_vecs[:, :n_components]
    X_reduced = np.dot(X_meaned, eig_vecs)
    return X_reduced


# Optimización: MDS con preprocesamiento


def mds(X, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mds = MDS(n_components=n_components, n_init=4,
              max_iter=300, random_state=42)
    return mds.fit_transform(X_scaled)


# Configuración de página
st.set_page_config(
    page_title="🏁 Dashboard 24H Le Mans",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS optimizados
st.markdown("""
<style>
    .stButton>button { background-color: #004080; color: white; }
    .stTabs [role="tab"] { font-size: 16px; }
    [data-testid="stMetricValue"] { color: #004080; }
    .css-1d391kg { background: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

st.title("🏎 Análisis Interactivo 24 Hrs Le Mans")

# Cache para carga de datos


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')

    # Limpieza más eficiente
    df = df.drop(['S.No', 'Status'], axis=1)
    df.columns = df.columns.str.replace(r'^#\s*', '', regex=True).str.strip()

    # Procesamiento de columnas
    df['Car No.'] = df['Car No.'].astype(str)
    df['Category'] = df['Category'].fillna('Missing')
    df['Tyres'] = df['Tyres'].fillna('Missing')

    # Optimización: get_dummies solo para columnas necesarias
    df = pd.get_dummies(df, columns=['Tyres'], prefix='', prefix_sep='')

    # Conversión de tiempos optimizada
    df['Best Lap Kph'] = pd.to_numeric(df['Best Lap Kph'], errors='coerce')
    df['Best Lap Kph'] = df['Best Lap Kph'].fillna(df['Best Lap Kph'].median())

    # Manejo de tiempos
    df['total_time'] = pd.to_timedelta(df['Total Time']).dt.total_seconds()
    df['Hour'] = pd.to_timedelta(df['Total Time']).dt.components.hours
    df['Lap_record'] = pd.to_timedelta(
        '00:' + df['Best LapTime']).dt.total_seconds()

    # Variables numéricas
    df['Pitstops'] = pd.to_numeric(
        df['Pitstops'], errors='coerce').fillna(0).astype(int)
    df['Pitstop_Binary'] = (df['Pitstops'] > 0).astype(int)
    df['Laps'] = pd.to_numeric(
        df['Laps'], errors='coerce').fillna(0).astype(int)

    return df


# Carga de datos
le_mans = load_data('24LeMans.csv')

# Sidebar optimizado
with st.sidebar:
    with st.expander('🔍 Filtros', expanded=True):
        equipos = le_mans['Team'].unique().tolist()
        sel_equipos = st.multiselect(
            'Equipos', sorted(equipos), default=equipos)

        # Filtro de pilotos más eficiente
        pilotos_filtrados = le_mans[le_mans['Team'].isin(
            sel_equipos)]['Drivers'].unique()
        sel_piloto = st.selectbox(
            'Piloto', ['Todos'] + sorted(pilotos_filtrados))

        categorias = le_mans['Category'].unique().tolist()
        sel_categorias = st.multiselect(
            'Categorías', sorted(categorias), default=categorias)

        # Rangos dinámicos
        hmin, hmax = int(le_mans['Hour'].min()), int(le_mans['Hour'].max())
        sel_hour = st.slider('Hora (carrera)', hmin, hmax, (hmin, hmax))

        lmin, lmax = int(le_mans['Laps'].min()), int(le_mans['Laps'].max())
        sel_laps = st.slider('Vueltas', lmin, lmax, (lmin, lmax))

        show_raw = st.checkbox('Mostrar tabla completa', value=False)

# Filtrado optimizado
mask = (
    le_mans['Team'].isin(sel_equipos) &
    le_mans['Category'].isin(sel_categorias) &
    le_mans['Hour'].between(*sel_hour) &
    le_mans['Laps'].between(*sel_laps)
)

if sel_piloto != 'Todos':
    mask &= (le_mans['Drivers'] == sel_piloto)

filtered = le_mans.loc[mask].copy()

# Tabs con lazy loading
tabs = st.tabs([
    '📍 Mapa', '📊 Resumen', '📈 Visualizaciones', '📋 Datos',
    '🔀 Proyecciones', '📐 Correlaciones'
])

with tabs[0]:
    st.header('📍 Circuit de la Sarthe')
    st.image('assets/lemans.png', use_container_width=True)

with tabs[1]:
    st.header('🔑 Métricas Clave')
    cols = st.columns(4)
    with cols[0]:
        st.metric('Avg Total Time (s)', f"{filtered['total_time'].mean():.0f}")
    with cols[1]:
        st.metric('Avg Best Lap (s)', f"{filtered['Lap_record'].mean():.2f}")
    with cols[2]:
        st.metric('Avg Pit-stops', f"{filtered['Pitstops'].mean():.1f}")
    with cols[3]:
        st.metric('Registros', len(filtered))

with tabs[2]:
    st.header('📈 Visualizaciones')

    # Gráficos optimizados
    fig1 = px.histogram(filtered, x='Best Lap Kph', nbins=30, marginal='box')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(filtered, x='Hour', y='total_time',
                   color='Team', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        filtered.dropna(subset=['Laps', 'Lap_record', 'Best Lap Kph']),
        x='Laps', y='Lap_record', color='Team',
        size='Best Lap Kph', hover_data=['Car No.', 'Drivers']
    )
    st.plotly_chart(fig3, use_container_width=True)

with tabs[3]:
    st.header('🗄 Resumen Agregado de Datos por Hora y Coche')
    if show_raw:
        agg_df = filtered.groupby(['Hour', 'Car No.', 'Team', 'Drivers']).agg({
            'Laps': 'max',
            'Lap_record': 'mean',
            'Best Lap Kph': 'mean'
        }).reset_index()
        st.dataframe(agg_df, use_container_width=True)
    else:
        st.info("Activa 'Mostrar tabla completa' para ver los datos")

with tabs[4]:
    st.header('🔀 Proyecciones Dimensionales')

    # Variables para proyección
    X_vars = ['Best Lap Kph', 'total_time', 'Lap_record', 'Pitstops', 'Laps']
    df_clean = filtered[X_vars].dropna()

    if len(df_clean) < 3:
        st.warning("Se necesitan al menos 3 puntos para proyecciones")
        st.stop()

    method = st.radio('Método:', ['PCA', 'MDS', 'Sammon'])

    # Normalización de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # Proyección con manejo de errores
    if method == 'PCA':
        X_proj = pca(X_scaled)
    elif method == 'MDS':
        X_proj = mds(X_scaled)
    elif method == 'Sammon':
        X_proj = fast_sammon(X_scaled)

    # Visualización
    if X_proj is not None:
        proj_df = pd.DataFrame(X_proj, columns=['PC1', 'PC2'])
        proj_df['Team'] = filtered.loc[df_clean.index, 'Team'].values
        proj_df['Car No.'] = filtered.loc[df_clean.index, 'Car No.'].values

        fig = px.scatter(
            proj_df, x='PC1', y='PC2', color='Team',
            hover_data=['Car No.'], title=f'Proyección {method}'
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.header('📐 Indicadores de Correlación')

    # Variables numéricas
    num_vars = ['Best Lap Kph', 'total_time', 'Lap_record', 'Pitstops', 'Laps']
    corr_df = filtered[num_vars].dropna()

    # --- Pearson ---
    st.subheader('📊 Matriz de correlación - Pearson')
    corr_pearson = corr_df.corr(method='pearson').round(2)

    fig_p = go.Figure(data=go.Heatmap(
        z=corr_pearson.values,
        x=corr_pearson.columns,
        y=corr_pearson.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title='Coef. Pearson')
    ))
    fig_p.update_layout(
        title='Mapa de calor (Pearson)',
        xaxis_title='Variables',
        yaxis_title='Variables'
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # --- Spearman ---
    st.subheader('📊 Matriz de correlación - Spearman')
    corr_spearman = corr_df.corr(method='spearman').round(2)

    fig_s = go.Figure(data=go.Heatmap(
        z=corr_spearman.values,
        x=corr_spearman.columns,
        y=corr_spearman.index,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        colorbar=dict(title='Coef. Spearman')
    ))
    fig_s.update_layout(
        title='Mapa de calor (Spearman)',
        xaxis_title='Variables',
        yaxis_title='Variables'
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # --- Kendall ---
    st.subheader('📊 Matriz de correlación - Kendall')
    corr_kendall = corr_df.corr(method='kendall').round(2)

    fig_k = go.Figure(data=go.Heatmap(
        z=corr_kendall.values,
        x=corr_kendall.columns,
        y=corr_kendall.index,
        colorscale='Blues',
        zmin=-1, zmax=1,
        colorbar=dict(title='Coef. Kendall')
    ))
    fig_k.update_layout(
        title='Mapa de calor (Kendall)',
        xaxis_title='Variables',
        yaxis_title='Variables'
    )
    st.plotly_chart(fig_k, use_container_width=True)

    st.header('📐 Indicadores de Correlación')

    # Variables numéricas
    num_vars = ['Best Lap Kph', 'total_time', 'Lap_record', 'Pitstops', 'Laps']
    corr_df = filtered[num_vars].dropna()

    st.subheader('📊 Matriz de correlación - Pearson')
    corr_pearson = corr_df.corr(method='pearson')

    fig_p, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig_p)

    st.subheader('📊 Matriz de correlación - Spearman')
    corr_spearman = corr_df.corr(method='spearman')

    fig_s, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_spearman, annot=True, cmap='YlGnBu', ax=ax)
    st.pyplot(fig_s)

    st.subheader('📊 Matriz de correlación - Kendall')
    corr_kendall = corr_df.corr(method='kendall')

    fig_k, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_kendall, annot=True, cmap='PuBuGn', ax=ax)
    st.pyplot(fig_k)

st.markdown('---')
