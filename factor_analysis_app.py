import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer.rotator import Rotator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(
    page_title="Факторный анализ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class FactorAnalysisApp:
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.factor_analyzer = None
        self.pca_model = None
        self.scaler = StandardScaler()
        
    def load_data(self, uploaded_file):
        """Загрузка данных из файла"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(uploaded_file)
            else:
                st.error("Поддерживаются только файлы CSV и Excel")
                return False
            return True
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")
            return False
    
    def clean_data(self, columns_to_use, missing_strategy='mean', normalize=True):
        """Очистка и предобработка данных"""
        try:
            # Выбираем только числовые колонки
            numeric_data = self.data[columns_to_use].select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                st.error("Выбранные колонки не содержат числовых данных")
                return False
            
            # Обработка пропусков
            if missing_strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif missing_strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif missing_strategy == 'drop':
                numeric_data = numeric_data.dropna()
                if normalize:
                    numeric_data = pd.DataFrame(
                        self.scaler.fit_transform(numeric_data),
                        columns=numeric_data.columns,
                        index=numeric_data.index
                    )
                self.cleaned_data = numeric_data
                return True
            
            imputed_data = imputer.fit_transform(numeric_data)
            
            # Нормализация
            if normalize:
                scaled_data = self.scaler.fit_transform(imputed_data)
                self.cleaned_data = pd.DataFrame(
                    scaled_data, 
                    columns=numeric_data.columns,
                    index=numeric_data.index
                )
            else:
                self.cleaned_data = pd.DataFrame(
                    imputed_data,
                    columns=numeric_data.columns,
                    index=numeric_data.index
                )
            
            return True
        except Exception as e:
            st.error(f"Ошибка при очистке данных: {str(e)}")
            return False
    
    def check_factor_analysis_suitability(self):
        """Проверка применимости факторного анализа"""
        try:
            # Тест Бартлетта
            chi_square, p_value = calculate_bartlett_sphericity(self.cleaned_data)
            
            # Тест KMO
            kmo_all, kmo_model = calculate_kmo(self.cleaned_data)
            
            return {
                'bartlett_chi2': chi_square,
                'bartlett_p': p_value,
                'kmo_model': kmo_model,
                'kmo_variables': kmo_all
            }
        except Exception as e:
            st.error(f"Ошибка при проверке применимости: {str(e)}")
            return None
    
    def correlation_matrix(self):
        """Построение матрицы корреляций"""
        return self.cleaned_data.corr()
    
    def perform_pca(self):
        """Выполнение анализа главных компонент"""
        try:
            self.pca_model = PCA()
            self.pca_model.fit(self.cleaned_data)
            return True
        except Exception as e:
            st.error(f"Ошибка при выполнении PCA: {str(e)}")
            return False
    
    def perform_factor_analysis(self, n_factors, method='principal', rotation='varimax'):
        """Выполнение факторного анализа"""
        try:
            if method == 'principal':
                self.factor_analyzer = FactorAnalyzer(
                    n_factors=n_factors,
                    rotation=rotation,
                    method='principal'
                )
            else:
                self.factor_analyzer = FactorAnalyzer(
                    n_factors=n_factors,
                    rotation=rotation,
                    method='ml'
                )
            
            self.factor_analyzer.fit(self.cleaned_data)
            return True
        except Exception as e:
            st.error(f"Ошибка при выполнении факторного анализа: {str(e)}")
            return False
    
    def get_eigenvalues(self):
        """Получение собственных значений"""
        if self.pca_model is not None:
            return self.pca_model.explained_variance_
        return None
    
    def plot_scree(self):
        """График Scree plot"""
        if self.pca_model is None:
            return None
        
        eigenvalues = self.pca_model.explained_variance_
        components = range(1, len(eigenvalues) + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(components),
            y=eigenvalues,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=8),
            name='Собственные значения'
        ))
        
        # Линия критерия Кайзера (eigenvalue = 1)
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Критерий Кайзера (λ=1)")
        
        fig.update_layout(
            title='Scree Plot - График собственных значений',
            xaxis_title='Компонента',
            yaxis_title='Собственное значение',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_factor_loadings(self):
        """График факторных нагрузок"""
        if self.factor_analyzer is None:
            return None
        
        loadings = self.factor_analyzer.loadings_
        variables = self.cleaned_data.columns
        n_factors = loadings.shape[1]
        
        fig = go.Figure()
        
        for i in range(n_factors):
            fig.add_trace(go.Bar(
                name=f'Фактор {i+1}',
                x=variables,
                y=loadings[:, i],
                opacity=0.8
            ))
        
        fig.update_layout(
            title='Факторные нагрузки',
            xaxis_title='Переменные',
            yaxis_title='Нагрузка',
            height=500,
            barmode='group'
        )
        
        return fig
    
    def plot_biplot(self):
        """Двумерный biplot"""
        if self.factor_analyzer is None or self.factor_analyzer.loadings_.shape[1] < 2:
            return None
        
        loadings = self.factor_analyzer.loadings_
        scores = self.factor_analyzer.transform(self.cleaned_data)
        variables = self.cleaned_data.columns
        
        fig = go.Figure()
        
        # Точки наблюдений
        fig.add_trace(go.Scatter(
            x=scores[:, 0],
            y=scores[:, 1],
            mode='markers',
            marker=dict(size=6, opacity=0.6),
            name='Наблюдения',
            hovertemplate='Наблюдение: %{pointNumber}<br>' +
                         'Фактор 1: %{x:.2f}<br>' +
                         'Фактор 2: %{y:.2f}<extra></extra>'
        ))
        
        # Векторы переменных
        for i, var in enumerate(variables):
            fig.add_trace(go.Scatter(
                x=[0, loadings[i, 0] * 3],
                y=[0, loadings[i, 1] * 3],
                mode='lines+text',
                line=dict(color='red', width=2),
                text=['', var],
                textposition='top center',
                name=var,
                showlegend=False,
                hovertemplate=f'{var}<br>' +
                             'Нагрузка Ф1: %{x:.3f}<br>' +
                             'Нагрузка Ф2: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Biplot - Двумерное представление факторов',
            xaxis_title='Фактор 1',
            yaxis_title='Фактор 2',
            height=600,
            showlegend=True
        )
        
        return fig

def main():
    st.title("📊 Факторный анализ - Аналог SPSS Statistics")
    st.markdown("---")
    
    # Инициализация приложения
    if 'app' not in st.session_state:
        st.session_state.app = FactorAnalysisApp()
    
    app = st.session_state.app
    
    # Боковая панель
    with st.sidebar:
        st.header("🎯 Цель анализа")
        goal = st.text_area(
            "Опишите цель вашего факторного анализа:",
            placeholder="Например: выявление скрытых факторов в поведении пользователей",
            height=100
        )
        
        if goal:
            st.success(f"Цель: {goal}")
    
    # Основное содержимое
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Данные", 
        "🧹 Подготовка", 
        "🔍 Анализ применимости", 
        "📈 Факторный анализ", 
        "📊 Результаты"
    ])
    
    with tab1:
        st.header("Импорт данных")
        uploaded_file = st.file_uploader(
            "Загрузите файл Excel или CSV",
            type=['csv', 'xlsx', 'xls'],
            help="Поддерживаются форматы: CSV, XLSX, XLS"
        )
        
        if uploaded_file is not None:
            if app.load_data(uploaded_file):
                st.success("✅ Данные успешно загружены!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Количество строк", app.data.shape[0])
                with col2:
                    st.metric("Количество столбцов", app.data.shape[1])
                
                st.subheader("Предварительный просмотр данных")
                st.dataframe(app.data.head(10))
                
                st.subheader("Информация о данных")
                buffer = io.StringIO()
                app.data.info(buf=buffer)
                st.text(buffer.getvalue())
    
    with tab2:
        if app.data is not None:
            st.header("Очистка и предобработка данных")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Выбор переменных")
                numeric_columns = app.data.select_dtypes(include=[np.number]).columns.tolist()
                selected_columns = st.multiselect(
                    "Выберите числовые переменные для анализа:",
                    numeric_columns,
                    default=numeric_columns
                )
            
            with col2:
                st.subheader("Параметры очистки")
                missing_strategy = st.selectbox(
                    "Обработка пропусков:",
                    ['mean', 'median', 'drop'],
                    format_func=lambda x: {
                        'mean': 'Заполнить средним',
                        'median': 'Заполнить медианой',
                        'drop': 'Удалить строки'
                    }[x]
                )
                
                normalize = st.checkbox("Нормализация данных", value=True)
            
            if st.button("🧹 Очистить данные"):
                if selected_columns:
                    if app.clean_data(selected_columns, missing_strategy, normalize):
                        st.success("✅ Данные успешно очищены!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Финальные строки", app.cleaned_data.shape[0])
                        with col2:
                            st.metric("Финальные столбцы", app.cleaned_data.shape[1])
                        with col3:
                            st.metric("Пропуски", app.cleaned_data.isnull().sum().sum())
                        
                        st.subheader("Очищенные данные")
                        st.dataframe(app.cleaned_data.head())
                        
                        st.subheader("Статистика")
                        st.dataframe(app.cleaned_data.describe())
                else:
                    st.warning("Выберите хотя бы одну переменную")
        else:
            st.warning("Сначала загрузите данные в разделе 'Данные'")
    
    with tab3:
        if app.cleaned_data is not None:
            st.header("Проверка применимости факторного анализа")
            
            if st.button("🔍 Проверить применимость"):
                suitability = app.check_factor_analysis_suitability()
                
                if suitability:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Тест Бартлетта на сферичность")
                        st.metric("χ² статистика", f"{suitability['bartlett_chi2']:.2f}")
                        st.metric("p-значение", f"{suitability['bartlett_p']:.2e}")
                        
                        if suitability['bartlett_p'] < 0.05:
                            st.markdown('<div class="success-box">✅ Тест Бартлетта: данные подходят для факторного анализа (p < 0.05)</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-box">❌ Тест Бартлетта: данные НЕ подходят для факторного анализа (p ≥ 0.05)</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Тест KMO (Kaiser-Meyer-Olkin)")
                        st.metric("KMO общий", f"{suitability['kmo_model']:.3f}")
                        
                        if suitability['kmo_model'] >= 0.8:
                            kmo_text = "Отличная применимость"
                            kmo_class = "success-box"
                        elif suitability['kmo_model'] >= 0.7:
                            kmo_text = "Хорошая применимость"
                            kmo_class = "success-box"
                        elif suitability['kmo_model'] >= 0.6:
                            kmo_text = "Средняя применимость"
                            kmo_class = "warning-box"
                        elif suitability['kmo_model'] >= 0.5:
                            kmo_text = "Слабая применимость"
                            kmo_class = "warning-box"
                        else:
                            kmo_text = "Неприменимо"
                            kmo_class = "error-box"
                        
                        st.markdown(f'<div class="{kmo_class}">KMO: {kmo_text}</div>', unsafe_allow_html=True)
                    
                    st.subheader("Матрица корреляций")
                    corr_matrix = app.correlation_matrix()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    plt.title('Матрица корреляций')
                    st.pyplot(fig)
                    plt.close()
        else:
            st.warning("Сначала очистите данные в разделе 'Подготовка'")
    
    with tab4:
        if app.cleaned_data is not None:
            st.header("Факторный анализ")
            
            # Выполнение PCA для определения количества факторов
            if st.button("📊 Выполнить предварительный анализ (PCA)"):
                if app.perform_pca():
                    st.success("✅ PCA выполнен успешно!")
                    
                    # Scree plot
                    scree_fig = app.plot_scree()
                    if scree_fig:
                        st.plotly_chart(scree_fig, use_container_width=True)
                    
                    # Таблица собственных значений
                    eigenvalues = app.get_eigenvalues()
                    explained_variance = app.pca_model.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance)
                    
                    results_df = pd.DataFrame({
                        'Компонента': range(1, len(eigenvalues) + 1),
                        'Собственное значение': eigenvalues,
                        'Объясненная дисперсия (%)': explained_variance * 100,
                        'Кумулятивная дисперсия (%)': cumulative_variance * 100
                    })
                    
                    st.subheader("Собственные значения и объясненная дисперсия")
                    st.dataframe(results_df)
                    
                    # Рекомендация по количеству факторов
                    kaiser_factors = np.sum(eigenvalues > 1)
                    st.info(f"🎯 Рекомендуемое количество факторов по критерию Кайзера: {kaiser_factors}")
            
            st.markdown("---")
            
            # Настройки факторного анализа
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_factors = st.number_input(
                    "Количество факторов:",
                    min_value=1,
                    max_value=min(10, app.cleaned_data.shape[1]-1) if app.cleaned_data is not None else 5,
                    value=3
                )
            
            with col2:
                method = st.selectbox(
                    "Метод извлечения:",
                    ['principal', 'ml'],
                    format_func=lambda x: {
                        'principal': 'Главные компоненты (PCA)',
                        'ml': 'Максимальное правдоподобие'
                    }[x]
                )
            
            with col3:
                rotation = st.selectbox(
                    "Метод вращения:",
                    ['varimax', 'oblimin', 'quartimax'],
                    format_func=lambda x: {
                        'varimax': 'Varimax (ортогональное)',
                        'oblimin': 'Oblimin (наклонное)',
                        'quartimax': 'Quartimax (ортогональное)'
                    }[x]
                )
            
            if st.button("🎯 Выполнить факторный анализ"):
                if app.perform_factor_analysis(n_factors, method, rotation):
                    st.success("✅ Факторный анализ выполнен успешно!")
                    
                    # Факторные нагрузки
                    loadings_fig = app.plot_factor_loadings()
                    if loadings_fig:
                        st.plotly_chart(loadings_fig, use_container_width=True)
                    
                    # Biplot
                    if n_factors >= 2:
                        biplot_fig = app.plot_biplot()
                        if biplot_fig:
                            st.plotly_chart(biplot_fig, use_container_width=True)
        else:
            st.warning("Сначала очистите данные в разделе 'Подготовка'")
    
    with tab5:
        if app.factor_analyzer is not None:
            st.header("Результаты и интерпретация")
            
            # Матрица факторных нагрузок
            st.subheader("Матрица факторных нагрузок")
            loadings_df = pd.DataFrame(
                app.factor_analyzer.loadings_,
                columns=[f'Фактор {i+1}' for i in range(app.factor_analyzer.loadings_.shape[1])],
                index=app.cleaned_data.columns
            )
            st.dataframe(loadings_df.round(3))
            
            # Объясненная дисперсия
            st.subheader("Объясненная дисперсия")
            if hasattr(app.factor_analyzer, 'get_factor_variance'):
                variance_info = app.factor_analyzer.get_factor_variance()
                variance_df = pd.DataFrame({
                    'Фактор': [f'Фактор {i+1}' for i in range(len(variance_info[0]))],
                    'Собственное значение': variance_info[0],
                    'Дисперсия (%)': variance_info[1] * 100,
                    'Кумулятивная дисперсия (%)': np.cumsum(variance_info[1]) * 100
                })
                st.dataframe(variance_df.round(3))
                
                total_variance = np.sum(variance_info[1]) * 100
                st.metric("Общая объясненная дисперсия", f"{total_variance:.1f}%")
            
            # Интерпретация факторов
            st.subheader("Интерпретация факторов")
            st.info("""
            **Рекомендации по интерпретации:**
            - Нагрузки > 0.7 считаются высокими
            - Нагрузки 0.4-0.7 считаются умеренными  
            - Нагрузки < 0.4 считаются низкими
            
            Для каждого фактора найдите переменные с наивысшими нагрузками 
            и дайте содержательную интерпретацию фактора.
            """)
            
            # Применение результатов
            st.subheader("Факторные оценки")
            if st.button("Рассчитать факторные оценки"):
                factor_scores = app.factor_analyzer.transform(app.cleaned_data)
                scores_df = pd.DataFrame(
                    factor_scores,
                    columns=[f'Фактор {i+1}' for i in range(factor_scores.shape[1])]
                )
                st.dataframe(scores_df.head(10))
                
                # Возможность скачать результаты
                csv = scores_df.to_csv(index=False)
                st.download_button(
                    label="📥 Скачать факторные оценки (CSV)",
                    data=csv,
                    file_name="factor_scores.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Сначала выполните факторный анализ в разделе 'Факторный анализ'")
    
    # Футер
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Факторный анализ - Аналог SPSS Statistics | Разработано на Python + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
