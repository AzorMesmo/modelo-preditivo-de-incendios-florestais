import folium
import joblib
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

st.set_page_config(page_title="Monitor de Focos", page_icon="🔥", layout="wide")

# --- Inicialização do Estado (Memória da Aplicação) ---
if "pontos_analisados" not in st.session_state:
    st.session_state.pontos_analisados = []
if "ultimo_clique" not in st.session_state:
    st.session_state.ultimo_clique = None

st.title("🔥 Monitor Preditivo de Queimadas no Brasil")
st.markdown("Clique em qualquer ponto do mapa para analisar o risco de incêndio. "
            "**O modelo atua estritamente em território brasileiro.**")


# --- Função de Gradiente de Cor (Verde -> Amarelo -> Vermelho) ---
def obter_cor_gradiente(prob):
    p = max(0.0, min(1.0, prob))
    if p < 0.5:
        # De Verde (#2ecc71) para Amarelo (#f1c40f)
        f = p * 2.0
        r = int(46 + (241 - 46) * f)
        g = int(204 + (196 - 204) * f)
        b = int(113 + (15 - 113) * f)
    else:
        # De Amarelo (#f1c40f) para Vermelho (#e74c3c)
        f = (p - 0.5) * 2.0
        r = int(241 + (231 - 241) * f)
        g = int(196 + (76 - 196) * f)
        b = int(15 + (60 - 15) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


# --- 1. Carregamento do Modelo ---
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load('models/modelo_risco_incendio_xgb.pkl')
    except Exception:
        # Mock para testes caso não tenha o modelo
        class MockModel:
            def predict_proba(self, df):
                return [[0, min(df['TempMedia'].iloc[0] / 40.0, 0.99)]]

        return MockModel()


modelo = carregar_modelo()

# --- Layout em duas colunas ---
col_mapa, col_info = st.columns([2, 1])

with col_mapa:
    # --- 2. Configuração do Mapa com Limites do Brasil ---
    m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4, min_zoom=4,
        # Impede o usuário de dar zoom out no mundo inteiro
        max_bounds=True,  # Trava a área de arrasto
        min_lat=-35.0, max_lat=6.0,  # Limites Norte-Sul do BR
        min_lon=-75.0, max_lon=-34.0  # Limites Leste-Oeste do BR
    )

    # Adiciona camada para deixar o mundo cinza exceto o Brasil
    try:
        url_world = ("https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/"
                     "world-countries.json")
        world_geo = requests.get(url_world).json()

        folium.GeoJson(world_geo,
            style_function=lambda feature: {'fillColor': 'transparent' if feature['id'] == 'BRA' else '#cccccc',
                'color': '#888888', 'weight': 1, 'fillOpacity': 0.7 if feature['id'] != 'BRA' else 0.0}).add_to(m)
    except Exception:
        pass  # Fallback caso falhe o download do GeoJSON

    # Desenha todos os pontos que já estão na memória com cor em gradiente
    for ponto in st.session_state.pontos_analisados:
        cor_hex = obter_cor_gradiente(ponto['risco'])
        folium.CircleMarker(location=[ponto['lat'], ponto['lon']], radius=8, color=cor_hex, fill=True,
            fill_color=cor_hex, fill_opacity=0.8, tooltip=f"{ponto['cidade']} - Risco: {ponto['risco']:.1%}").add_to(m)

    # Renderiza o mapa e escuta eventos de clique
    mapa_interativo = st_folium(m, height=600, width=700, returned_objects=["last_clicked"])

with col_info:
    st.subheader("Análise da Região")

    # --- 3. Lógica de Clique e Processamento ---
    if mapa_interativo.get("last_clicked"):
        lat = mapa_interativo["last_clicked"]["lat"]
        lon = mapa_interativo["last_clicked"]["lng"]
        coordenada_atual = (lat, lon)

        if coordenada_atual != st.session_state.ultimo_clique:
            st.session_state.ultimo_clique = coordenada_atual

            with st.spinner("Verificando jurisdição territorial..."):
                geolocator = Nominatim(user_agent="fire_risk_app_br")
                try:
                    location = geolocator.reverse((lat, lon), language='pt')

                    if location:
                        endereco = location.raw.get('address', {})
                        pais = endereco.get('country_code', '')

                        # VERIFICAÇÃO DE TERRITÓRIO
                        if pais != 'br':
                            st.error(
                                "⚠️ **Fora de Limites!** O ponto clicado pertence a outro país (ou oceano). "
                                "Este modelo foi treinado exclusivamente para o Brasil.")
                        else:
                            # Extração de nome amigável para o local
                            nome_cidade = endereco.get('city', endereco.get('town', endereco.get('village',
                                                                                                 endereco.get(
                                                                                                     'municipality',
                                                                                                     'Área Remota'))))
                            estado = endereco.get('state', '')
                            local_nome = f"{nome_cidade}, {estado}" if estado else nome_cidade

                            # Requisição na API YR
                            headers = {'User-Agent': 'MonitorQueimadasApp/1.0'}
                            url_yr = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
                            resposta = requests.get(url_yr, headers=headers)

                            if resposta.status_code == 200:
                                dados_clima = resposta.json()
                                tempos = dados_clima['properties']['timeseries'][0]
                                temp_atual = tempos['data']['instant']['details']['air_temperature']
                                precip_6h = tempos['data'].get('next_6_hours', {}).get('details', {}).get(
                                    'precipitation_amount', 0.0)

                                precip_mensal_projetada = precip_6h * 30

                                # Inferência do Modelo
                                entrada_ia = pd.DataFrame(
                                    {'TempMedia': [temp_atual], 'PrecipTotal': [precip_mensal_projetada]})
                                probabilidade = modelo.predict_proba(entrada_ia)[0][1]

                                # Salva no estado
                                st.session_state.pontos_analisados.append(
                                    {'lat': lat, 'lon': lon, 'cidade': local_nome, 'risco': probabilidade,
                                        'temp': temp_atual, 'precip': precip_mensal_projetada})

                                st.rerun()
                            else:
                                st.error("Erro ao buscar dados climáticos para esta coordenada.")
                    else:
                        st.error("Coordenada não reconhecida.")
                except Exception as e:
                    st.error("Falha ao comunicar com o serviço de geolocalização.")

    # --- 4. Exibição do último resultado processado ---
    if st.session_state.pontos_analisados:
        ultimo_ponto = st.session_state.pontos_analisados[-1]
        st.write(f"📍 **Local:** {ultimo_ponto['cidade']}")
        st.info(f"🌡️ **Temp:** {ultimo_ponto['temp']}°C | 🌧️ **Chuva (proj):** {ultimo_ponto['precip']:.1f} mm")

        prob = ultimo_ponto['risco']
        if prob >= 0.7:
            st.error(f"🚨 **Risco Crítico:** {prob:.1%}")
        elif prob >= 0.4:
            st.warning(f"⚠ **Risco Moderado:** {prob:.1%}")
        else:
            st.success(f"✅ **Risco Baixo:** {prob:.1%}")
        st.progress(float(prob))
    elif not mapa_interativo.get("last_clicked"):
        st.write("👆 Clique no mapa para iniciar a análise.")
