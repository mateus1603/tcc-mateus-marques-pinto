import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import holidays

# Analisar padrões diários (24h)
def plot_daily_pattern(df, origem, destino):
    # Filtrar dados para o par origem-destino específico
    subset = df[(df['nom_subsistema_origem'] == origem) & 
                (df['nom_subsistema_destino'] == destino)].copy()
    
    if subset.empty:
        print(f"Nenhum dado encontrado para {origem} → {destino}")
        return None
    
    # Garantir que din_instante seja datetime
    subset['din_instante'] = pd.to_datetime(subset['din_instante'])
    
    # Extrair hora
    subset['hour'] = subset['din_instante'].dt.hour
    
    # Calcular estatísticas por hora
    daily_stats = subset.groupby('hour')['val_intercambiomwmed'].agg(['mean', 'std']).reset_index()
    daily_stats['std'] = daily_stats['std'].fillna(0)  # Substituir NaN por 0
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar área de desvio padrão
    fig.add_trace(go.Scatter(
        x=list(daily_stats['hour']) + list(daily_stats['hour'][::-1]),
        y=list(daily_stats['mean'] + daily_stats['std']) + list((daily_stats['mean'] - daily_stats['std'])[::-1]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='±1 Desvio Padrão'
    ))
    
    # Adicionar linha principal
    fig.add_trace(go.Scatter(
        x=daily_stats['hour'],
        y=daily_stats['mean'],
        mode='lines+markers',
        name='Intercâmbio Médio',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#1f77b4'),
        hovertemplate='<b>Hora:</b> %{x}:00<br>' +
                      '<b>Intercâmbio Médio:</b> %{y:.2f} MW<br>' +
                      '<extra></extra>'
    ))
    
    # Configurar layout
    fig.update_layout(
        title={
            'text': f'Padrão Diário de Intercâmbio: {origem} → {destino}',
            'x': 0.5,
            'font': {'size': 18}
        },
        xaxis=dict(
            title=dict(text='Hora do Dia', font=dict(size=14)),
            tickmode='linear',
            tick0=0,
            dtick=2,
            gridcolor='lightgray',
            gridwidth=1,
            range=[-0.5, 23.5]
        ),
        yaxis=dict(
            title=dict(text='Intercâmbio Médio (MW)', font=dict(size=14)),
            gridcolor='lightgray',
            gridwidth=1
        ),
        width=900,
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Adicionar anotações para picos (apenas se houver dados)
    if not daily_stats.empty:
        max_idx = daily_stats['mean'].idxmax()
        min_idx = daily_stats['mean'].idxmin()
        
        max_hour = daily_stats.loc[max_idx, 'hour']
        max_value = daily_stats.loc[max_idx, 'mean']
        min_hour = daily_stats.loc[min_idx, 'hour']
        min_value = daily_stats.loc[min_idx, 'mean']
        
        fig.add_annotation(
            x=max_hour,
            y=max_value,
            text=f"Pico: {max_value:.1f} MW",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red"
        )
        
        fig.add_annotation(
            x=min_hour,
            y=min_value,
            text=f"Mínimo: {min_value:.1f} MW",
            showarrow=True,
            arrowhead=2,
            arrowcolor="blue",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue"
        )
    
    fig.show()
    return fig

# Padrão semanal (168h)
def plot_weekly_pattern(df, origem, destino):
    # Filtrar dados para o par origem-destino específico
    subset = df[(df['nom_subsistema_origem'] == origem) & 
                (df['nom_subsistema_destino'] == destino)].copy()
    
    if subset.empty:
        print(f"Nenhum dado encontrado para {origem} → {destino}")
        return None
    
    # Garantir que din_instante seja datetime
    subset['din_instante'] = pd.to_datetime(subset['din_instante'])
    
    # Extrair componentes de tempo
    subset['dayofweek'] = subset['din_instante'].dt.dayofweek
    subset['hour'] = subset['din_instante'].dt.hour
    subset['hourofweek'] = subset['dayofweek'] * 24 + subset['hour']
    
    # Calcular estatísticas por hora da semana
    weekly_stats = subset.groupby('hourofweek')['val_intercambiomwmed'].agg(['mean', 'std']).reset_index()
    weekly_stats['std'] = weekly_stats['std'].fillna(0)  # Substituir NaN por 0
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar área de desvio padrão
    fig.add_trace(go.Scatter(
        x=list(weekly_stats['hourofweek']) + list(weekly_stats['hourofweek'][::-1]),
        y=list(weekly_stats['mean'] + weekly_stats['std']) + list((weekly_stats['mean'] - weekly_stats['std'])[::-1]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='±1 Desvio Padrão'
    ))
    
    # Preparar dados customizados para hover
    custom_data = []
    days = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    for h in weekly_stats['hourofweek']:
        day_name = days[h // 24]
        hour_of_day = h % 24
        custom_data.append([day_name, hour_of_day])
    
    # Adicionar linha principal
    fig.add_trace(go.Scatter(
        x=weekly_stats['hourofweek'],
        y=weekly_stats['mean'],
        mode='lines',
        name='Intercâmbio Médio',
        line=dict(color='#1f77b4', width=2),
        customdata=custom_data,
        hovertemplate='<b>Dia:</b> %{customdata[0]}<br>' +
                      '<b>Hora:</b> %{customdata[1]}:00<br>' +
                      '<b>Intercâmbio Médio:</b> %{y:.2f} MW<br>' +
                      '<extra></extra>'
    ))
    
    # Adicionar linhas verticais para separar os dias
    days_short = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink']
    
    for i in range(7):
        fig.add_vline(
            x=i*24, 
            line_dash="dash", 
            line_color=colors[i], 
            opacity=0.5,
            annotation_text=days_short[i],
            annotation_position="top"
        )
    
    # Configurar layout
    fig.update_layout(
        title={
            'text': f'Padrão Semanal de Intercâmbio: {origem} → {destino}',
            'x': 0.5,
            'font': {'size': 18}
        },
        xaxis=dict(
            title=dict(text='Hora da Semana (0-167)', font=dict(size=14)),
            tickmode='array',
            tickvals=[i*24 for i in range(8)],
            ticktext=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom', ''],
            gridcolor='lightgray',
            gridwidth=1,
            range=[-2, 170]
        ),
        yaxis=dict(
            title=dict(text='Intercâmbio Médio (MW)', font=dict(size=14)),
            gridcolor='lightgray',
            gridwidth=1
        ),
        width=1200,
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Adicionar retângulos coloridos para destacar fins de semana
    fig.add_vrect(
        x0=5*24, x1=7*24,
        fillcolor="lightblue", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Fim de Semana",
        annotation_position="top left"
    )
    
    fig.show()
    return fig

def plot_timeseries_analysis(df, origem, destino):
    """
    Cria visualização completa de série temporal com análise de lacunas e interpolação
    """
    # Selecionar par origem-destino específico
    target_df = df[(df['nom_subsistema_origem'] == origem) & 
                   (df['nom_subsistema_destino'] == destino)].copy()
    
    if target_df.empty:
        print(f"Nenhum dado encontrado para {origem} → {destino}")
        return None
    
    # Garantir que din_instante seja datetime
    target_df['din_instante'] = pd.to_datetime(target_df['din_instante'])
    
    # Reorganizar para formato de série temporal
    ts_df = target_df.set_index('din_instante')[['val_intercambiomwmed']].copy()
    ts_df.columns = ['MW']
    
    # Verificar lacunas na série temporal
    full_range = pd.date_range(start=ts_df.index.min(), end=ts_df.index.max(), freq='1h')
    missing_times = full_range.difference(ts_df.index)
    
    print(f"Período: {ts_df.index.min().strftime('%Y-%m-%d %H:%M')} até {ts_df.index.max().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total de pontos esperados: {len(full_range)}")
    print(f"Pontos existentes: {len(ts_df)}")
    print(f"Timestamps faltantes: {len(missing_times)}")
    
    # Criar dados originais e interpolados
    original_data = ts_df.copy()
    
    # Preencher lacunas se existirem
    if len(missing_times) > 0:
        ts_df_interpolated = ts_df.reindex(full_range)
        ts_df_interpolated = ts_df_interpolated.interpolate(method='linear')
        print(f"Interpolação aplicada para {len(missing_times)} pontos faltantes")
    else:
        ts_df_interpolated = ts_df.copy()
        print("Nenhuma interpolação necessária - série completa")
    
    # Identificar pontos interpolados
    interpolated_mask = ts_df_interpolated.index.isin(missing_times)
    
    # Criar subplot com 2 gráficos
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            'Série Temporal Completa (Original + Interpolado)',
            'Estatísticas e Lacunas na Série'
        ],
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )
    
    # Gráfico principal - série temporal
    # Dados originais
    fig.add_trace(
        go.Scatter(
            x=original_data.index,
            y=original_data['MW'],
            mode='lines',
            name='Dados Originais',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Data/Hora:</b> %{x}<br>' +
                          '<b>Intercâmbio:</b> %{y:.2f} MW<br>' +
                          '<b>Status:</b> Original<br>' +
                          '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Pontos interpolados (se houver)
    if len(missing_times) > 0:
        interpolated_points = ts_df_interpolated[interpolated_mask]
        fig.add_trace(
            go.Scatter(
                x=interpolated_points.index,
                y=interpolated_points['MW'],
                mode='markers',
                name='Pontos Interpolados',
                marker=dict(color='red', size=4, symbol='diamond'),
                hovertemplate='<b>Data/Hora:</b> %{x}<br>' +
                              '<b>Intercâmbio:</b> %{y:.2f} MW<br>' +
                              '<b>Status:</b> Interpolado<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Gráfico secundário - estatísticas por dia
    daily_stats = ts_df_interpolated.resample('D').agg({
        'MW': ['mean', 'min', 'max', 'std']
    }).round(2)
    daily_stats.columns = ['Media', 'Minimo', 'Maximo', 'DesvPadrao']
    
    # Média diária
    fig.add_trace(
        go.Scatter(
            x=daily_stats.index,
            y=daily_stats['Media'],
            mode='lines+markers',
            name='Média Diária',
            line=dict(color='green', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Data:</b> %{x}<br>' +
                          '<b>Média:</b> %{y:.2f} MW<br>' +
                          '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Área de variação (min-max) diária
    fig.add_trace(
        go.Scatter(
            x=list(daily_stats.index) + list(daily_stats.index[::-1]),
            y=list(daily_stats['Maximo']) + list(daily_stats['Minimo'][::-1]),
            fill='toself',
            fillcolor='rgba(0,176,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Faixa Min-Max Diária'
        ),
        row=2, col=1
    )
    
    # Configurar layout
    fig.update_layout(
        title={
            'text': f'Análise de Série Temporal: {origem} → {destino}',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=800,
        width=1200,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98
        )
    )
    
    # Configurar eixos do gráfico principal
    fig.update_xaxes(
        title=dict(text='Data/Hora', font=dict(size=14)),
        gridcolor='lightgray',
        row=1, col=1
    )
    fig.update_yaxes(
        title=dict(text='Intercâmbio (MW)', font=dict(size=14)),
        gridcolor='lightgray',
        row=1, col=1
    )
    
    # Configurar eixos do gráfico secundário
    fig.update_xaxes(
        title=dict(text='Data', font=dict(size=12)),
        gridcolor='lightgray',
        row=2, col=1
    )
    fig.update_yaxes(
        title=dict(text='MW (Estatísticas Diárias)', font=dict(size=12)),
        gridcolor='lightgray',
        row=2, col=1
    )
    
    fig.show()
    
    # Retornar dados para análises adicionais
    return {
        'original_data': original_data,
        'interpolated_data': ts_df_interpolated,
        'missing_times': missing_times,
        'daily_stats': daily_stats,
        'figure': fig
    }

def plot_simple_timeseries(df, origem, destino):
    """
    Versão simplificada para visualização rápida da série temporal
    """
    # Selecionar par origem-destino específico
    target_df = df[(df['nom_subsistema_origem'] == origem) & 
                   (df['nom_subsistema_destino'] == destino)].copy()
    
    if target_df.empty:
        print(f"Nenhum dado encontrado para {origem} → {destino}")
        return None
    
    # Garantir que din_instante seja datetime
    target_df['din_instante'] = pd.to_datetime(target_df['din_instante'])
    
    # Reorganizar para formato de série temporal
    ts_df = target_df.set_index('din_instante')[['val_intercambiomwmed']].copy()
    ts_df.columns = ['MW']
    
    # Verificar e preencher lacunas
    full_range = pd.date_range(start=ts_df.index.min(), end=ts_df.index.max(), freq='1h')
    missing_times = full_range.difference(ts_df.index)
    
    if len(missing_times) > 0:
        ts_df = ts_df.reindex(full_range)
        ts_df = ts_df.interpolate(method='linear')
        print(f"Interpolação aplicada para {len(missing_times)} pontos faltantes")
    
    # Criar figura
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ts_df.index,
        y=ts_df['MW'],
        mode='lines',
        name='Intercâmbio',
        line=dict(color='#1f77b4', width=1.5),
        hovertemplate='<b>Data/Hora:</b> %{x}<br>' +
                      '<b>Intercâmbio:</b> %{y:.2f} MW<br>' +
                      '<extra></extra>'
    ))
    
    # Configurar layout
    fig.update_layout(
        title={
            'text': f'Série Temporal de Intercâmbio: {origem} → {destino}',
            'x': 0.5,
            'font': {'size': 18}
        },
        xaxis=dict(
            title=dict(text='Data/Hora', font=dict(size=14)),
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(text='Intercâmbio (MW)', font=dict(size=14)),
            gridcolor='lightgray'
        ),
        width=1200,
        height=500,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    fig.show()
    return fig

def analyze_data_quality(df, origem, destino):
    """
    Função para analisar a qualidade dos dados da série temporal
    """
    target_df = df[(df['nom_subsistema_origem'] == origem) & 
                   (df['nom_subsistema_destino'] == destino)].copy()
    
    if target_df.empty:
        print(f"Nenhum dado encontrado para {origem} → {destino}")
        return None
    
    target_df['din_instante'] = pd.to_datetime(target_df['din_instante'])
    ts_df = target_df.set_index('din_instante')[['val_intercambiomwmed']]
    
    # Estatísticas básicas
    print(f"\n=== ANÁLISE DE QUALIDADE DOS DADOS ===")
    print(f"Par: {origem} → {destino}")
    print(f"Período: {ts_df.index.min()} até {ts_df.index.max()}")
    print(f"Duração: {(ts_df.index.max() - ts_df.index.min()).days + 1} dias")
    
    # Verificar lacunas
    expected_range = pd.date_range(start=ts_df.index.min(), end=ts_df.index.max(), freq='1h')
    missing_times = expected_range.difference(ts_df.index)
    
    print(f"\nPontos de dados:")
    print(f"- Esperados: {len(expected_range)}")
    print(f"- Existentes: {len(ts_df)}")
    print(f"- Faltantes: {len(missing_times)} ({len(missing_times)/len(expected_range)*100:.2f}%)")
    
    # Estatísticas dos valores
    print(f"\nEstatísticas dos valores:")
    print(f"- Média: {ts_df['val_intercambiomwmed'].mean():.2f} MW")
    print(f"- Mediana: {ts_df['val_intercambiomwmed'].median():.2f} MW")
    print(f"- Desvio Padrão: {ts_df['val_intercambiomwmed'].std():.2f} MW")
    print(f"- Mínimo: {ts_df['val_intercambiomwmed'].min():.2f} MW")
    print(f"- Máximo: {ts_df['val_intercambiomwmed'].max():.2f} MW")
    
    # Valores extremos
    q99 = ts_df['val_intercambiomwmed'].quantile(0.99)
    q01 = ts_df['val_intercambiomwmed'].quantile(0.01)
    outliers = ts_df[(ts_df['val_intercambiomwmed'] > q99) | (ts_df['val_intercambiomwmed'] < q01)]
    print(f"- Outliers (1% e 99%): {len(outliers)} pontos")
    
    return {
        'stats': ts_df['val_intercambiomwmed'].describe(),
        'missing_times': missing_times,
        'outliers': outliers
    }

def add_calendar_features(df, feriados):
    """
    Adiciona features de calendário ao DataFrame.
    
    Args:
        df: DataFrame com índice datetime
        
    Returns:
        DataFrame com features de calendário adicionadas
    """
    df = df.copy()
    
    # Features básicas de tempo
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['year'] = df.index.year
    
    # Encoding cíclico
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Identificadores de períodos especiais
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Simplificação: como não temos uma lista de feriados,
    # vamos apenas fingir que os fins de semana são feriados
    # Converter as datas dos feriados para datetime.date
    feriados_dates = [date.date() if isinstance(date, pd.Timestamp) else date for date in feriados]
    
    # Verificar se cada data do índice é um feriado
    df['is_holiday'] = [1 if date.date() in feriados_dates else 0 for date in df.index]
    
    # Lags importantes (t-24 e t-168)
    df['MW_lag24'] = df['MW'].shift(24)   # Valor de 24 horas atrás (mesmo horário do dia anterior)
    df['MW_lag168'] = df['MW'].shift(168) # Valor de 168 horas atrás (mesmo horário da semana anterior)
    
    return df

