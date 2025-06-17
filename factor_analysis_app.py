import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Настройка страницы
st.set_page_config(
    page_title="Факторный анализ - Аналог SPSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FactorAnalysisApp:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self, file):
        """Загрузка данных из файла"""
        try:
            if file.name.endswith('.csv'):
                self.data = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file)
            else:
                st.error("Поддерживаются только форматы CSV и Excel")
                return False
            return True
        except Exception as e:
            st.error(f"Ошибка загрузки файла: {str(e)}")
            return False
    
    def preprocess_data(self, handle_missing='mean', normalize=True, selected_columns=None):
        """Предобработка данных"""
        if self.data is None:
            return False
        
        # Выбор колонок
        if selected_columns:
            numeric_data = self.data[selected_columns]
        else:
            numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            st.error("Не найдено числовых колонок для анализа")
            return False
        
        # Обработка пропусков
        if handle_missing == 'mean':
            self.imputer.set_params(strategy='mean')
        elif handle_missing == 'median':
            self.imputer.set_params(strategy='median')
        elif handle_missing == 'drop':
            numeric_data = numeric_data.dropna()
        
        if handle_missing != 'drop':
            numeric_data = pd.DataFrame(
                self.imputer.fit_transform(numeric_data),
                columns=numeric_data.columns
            )
        
        # Нормализация
        if normalize:
            self.processed_data = pd.DataFrame(
                self.scaler.fit_transform(numeric_data),
                columns=numeric_data.columns
            )
        else:
            self.processed_data = numeric_data
        
        return True
    
    def calculate_correlation_matrix(self):
        """Расчет корреляционной матрицы"""
        if self.processed_data is None:
            return None
        return self.processed_data.corr()
    
    def perform_pca(self, n_components=None):
        """Проведение PCA анализа"""
        if self.processed_data is None:
            return None, None
        
        if n_components is None:
            n_components = min(self.processed_data.shape)
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.processed_data)
        
        return pca, pca_result
    
    def perform_factor_analysis(self, n_factors, rotation='varimax'):
        """Проведение факторного анализа"""
        if self.processed_data is None:
            return None
        
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(self.processed_data)
        
        return fa
    
    def calculate_adequacy_tests(self):
        """Расчет тестов адекватности (KMO и Барлетта)"""
        if self.processed_data is None:
            return None, None
        
        # Тест Барлетта
        chi_square, p_value = calculate_bartlett_sphericity(self.processed_data)
        
        # KMO тест
        kmo_all, kmo_model = calculate_kmo(self.processed_data)
        
        return (chi_square, p_value), (kmo_all, kmo_model)

def main():
    st.markdown('<h1 class="main-header">📊 Факторный анализ - Аналог SPSS Statistics</h1>', 
                unsafe_allow_html=True)
    
    app = FactorAnalysisApp()
    
    # Боковая панель
    st.sidebar.title("Настройки анализа")
    
    # Загрузка данных
    st.sidebar.markdown("### 📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Выберите файл (CSV или Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Загрузите файл с данными для анализа"
    )
    
    if uploaded_file is not None:
        if app.load_data(uploaded_file):
            st.success(f"✅ Файл успешно загружен: {uploaded_file.name}")
            
            # Просмотр данных
            st.markdown('<h2 class="section-header">📋 Просмотр данных</h2>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Количество строк", app.data.shape[0])
            with col2:
                st.metric("Количество столбцов", app.data.shape[1])
            with col3:
                numeric_cols = len(app.data.select_dtypes(include=[np.number]).columns)
                st.metric("Числовых столбцов", numeric_cols)
            
            # Отображение первых строк
            st.dataframe(app.data.head(), use_container_width=True)
            
            # Выбор колонок для анализа
            st.sidebar.markdown("### 🎯 Выбор переменных")
            numeric_columns = app.data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_columns:
                selected_columns = st.sidebar.multiselect(
                    "Выберите переменные для анализа",
                    numeric_columns,
                    default=numeric_columns[:min(10, len(numeric_columns))],
                    help="Выберите числовые переменные для факторного анализа"
                )
                
                if selected_columns:
                    # Настройки предобработки
                    st.sidebar.markdown("### ⚙️ Предобработка данных")
                    
                    handle_missing = st.sidebar.selectbox(
                        "Обработка пропусков",
                        ['mean', 'median', 'drop'],
                        format_func=lambda x: {
                            'mean': 'Заменить средним',
                            'median': 'Заменить медианой',
                            'drop': 'Удалить строки'
                        }[x]
                    )
                    
                    normalize_data = st.sidebar.checkbox(
                        "Стандартизировать данные", 
                        value=True,
                        help="Рекомендуется для факторного анализа"
                    )
                    
                    # Предобработка данных
                    if app.preprocess_data(handle_missing, normalize_data, selected_columns):
                        
                        # Описательная статистика
                        st.markdown('<h2 class="section-header">📈 Описательная статистика</h2>', 
                                   unsafe_allow_html=True)
                        st.dataframe(app.processed_data.describe(), use_container_width=True)
                        
                        # Информация о пропусках
                        missing_info = app.data[selected_columns].isnull().sum()
                        if missing_info.sum() > 0:
                            st.warning(f"⚠️ Обнаружено {missing_info.sum()} пропущенных значений")
                            st.dataframe(missing_info[missing_info > 0], use_container_width=True)
                        
                        # Корреляционная матрица
                        st.markdown('<h2 class="section-header">🔗 Корреляционная матрица</h2>', 
                                   unsafe_allow_html=True)
                        
                        corr_matrix = app.calculate_correlation_matrix()
                        
                        # Тепловая карта корреляций
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Корреляционная матрица"
                        )
                        fig_corr.update_traces(texttemplate="%{z:.2f}", textfont_size=10)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Тесты адекватности
                        st.markdown('<h2 class="section-header">✅ Тесты адекватности</h2>', 
                                   unsafe_allow_html=True)
                        
                        bartlett_test, kmo_test = app.calculate_adequacy_tests()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Тест Барлетта**")
                            chi_square, p_value = bartlett_test
                            st.metric("Chi-square", f"{chi_square:.3f}")
                            st.metric("p-value", f"{p_value:.6f}")
                            if p_value < 0.05:
                                st.success("✅ Данные подходят для факторного анализа")
                            else:
                                st.warning("⚠️ Данные могут не подходить для факторного анализа")
                        
                        with col2:
                            st.markdown("**KMO тест**")
                            kmo_all, kmo_model = kmo_test
                            st.metric("KMO общий", f"{kmo_all:.3f}")
                            
                            if kmo_all >= 0.8:
                                st.success("✅ Превосходно")
                            elif kmo_all >= 0.7:
                                st.success("✅ Хорошо")
                            elif kmo_all >= 0.6:
                                st.warning("⚠️ Удовлетворительно")
                            else:
                                st.error("❌ Неудовлетворительно")
                        
                        # Настройки анализа
                        st.sidebar.markdown("### 🎛️ Настройки факторного анализа")
                        
                        analysis_type = st.sidebar.selectbox(
                            "Тип анализа",
                            ['PCA', 'Factor Analysis'],
                            help="PCA - анализ главных компонент, Factor Analysis - факторный анализ"
                        )
                        
                        if analysis_type == 'PCA':
                            # PCA анализ
                            st.markdown('<h2 class="section-header">🔍 Анализ главных компонент (PCA)</h2>', 
                                       unsafe_allow_html=True)
                            
                            pca, pca_result = app.perform_pca()
                            
                            if pca is not None:
                                # Scree plot
                                fig_scree = go.Figure()
                                fig_scree.add_trace(go.Scatter(
                                    x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                                    y=pca.explained_variance_ratio_,
                                    mode='lines+markers',
                                    name='Собственные значения',
                                    line=dict(width=3),
                                    marker=dict(size=8)
                                ))
                                fig_scree.update_layout(
                                    title="Scree Plot - График осыпи",
                                    xaxis_title="Компонент",
                                    yaxis_title="Доля объясненной дисперсии",
                                    hovermode='x'
                                )
                                st.plotly_chart(fig_scree, use_container_width=True)
                                
                                # Кумулятивная дисперсия
                                cumsum_var = np.cumsum(pca.explained_variance_ratio_)
                                fig_cum = go.Figure()
                                fig_cum.add_trace(go.Scatter(
                                    x=list(range(1, len(cumsum_var) + 1)),
                                    y=cumsum_var,
                                    mode='lines+markers',
                                    name='Кумулятивная дисперсия',
                                    line=dict(width=3),
                                    marker=dict(size=8)
                                ))
                                fig_cum.add_hline(y=0.8, line_dash="dash", 
                                                 annotation_text="80% дисперсии")
                                fig_cum.update_layout(
                                    title="Кумулятивная объясненная дисперсия",
                                    xaxis_title="Количество компонент",
                                    yaxis_title="Кумулятивная доля дисперсии"
                                )
                                st.plotly_chart(fig_cum, use_container_width=True)
                                
                                # Компоненты для анализа
                                n_components_80 = np.argmax(cumsum_var >= 0.8) + 1
                                st.info(f"💡 Для объяснения 80% дисперсии требуется {n_components_80} компонент")
                                
                                # Нагрузки компонент
                                components_df = pd.DataFrame(
                                    pca.components_[:5].T,  # Показываем первые 5 компонент
                                    columns=[f'PC{i+1}' for i in range(min(5, pca.n_components_))],
                                    index=selected_columns
                                )
                                
                                st.markdown("**Нагрузки главных компонент**")
                                st.dataframe(components_df.round(3), use_container_width=True)
                                
                                # Biplot
                                if len(selected_columns) <= 20:  # Для читаемости
                                    fig_biplot = go.Figure()
                                    
                                    # Точки наблюдений
                                    fig_biplot.add_trace(go.Scatter(
                                        x=pca_result[:, 0],
                                        y=pca_result[:, 1],
                                        mode='markers',
                                        name='Наблюдения',
                                        marker=dict(size=5, opacity=0.6),
                                        showlegend=True
                                    ))
                                    
                                    # Векторы переменных
                                    for i, var in enumerate(selected_columns):
                                        fig_biplot.add_trace(go.Scatter(
                                            x=[0, pca.components_[0, i] * 3],
                                            y=[0, pca.components_[1, i] * 3],
                                            mode='lines+text',
                                            name=var,
                                            text=['', var],
                                            textposition='top center',
                                            line=dict(width=2),
                                            showlegend=False
                                        ))
                                    
                                    fig_biplot.update_layout(
                                        title="Biplot - Первые две главные компоненты",
                                        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)",
                                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)"
                                    )
                                    st.plotly_chart(fig_biplot, use_container_width=True)
                        
                        else:
                            # Факторный анализ
                            st.markdown('<h2 class="section-header">🔬 Факторный анализ</h2>', 
                                       unsafe_allow_html=True)
                            
                            n_factors = st.sidebar.slider(
                                "Количество факторов",
                                min_value=1,
                                max_value=min(10, len(selected_columns)-1),
                                value=min(3, len(selected_columns)-1),
                                help="Выберите количество факторов для извлечения"
                            )
                            
                            rotation = st.sidebar.selectbox(
                                "Тип вращения",
                                ['varimax', 'promax', 'oblimin', 'oblimax', 'quartimin'],
                                help="Varimax - ортогональное вращение (рекомендуется)"
                            )
                            
                            fa = app.perform_factor_analysis(n_factors, rotation)
                            
                            if fa is not None:
                                # Факторные нагрузки
                                loadings_df = pd.DataFrame(
                                    fa.loadings_,
                                    columns=[f'Фактор {i+1}' for i in range(n_factors)],
                                    index=selected_columns
                                )
                                
                                st.markdown("**Матрица факторных нагрузок**")
                                st.dataframe(loadings_df.round(3), use_container_width=True)
                                
                                # Тепловая карта нагрузок
                                fig_loadings = px.imshow(
                                    loadings_df.T,
                                    text_auto=True,
                                    aspect="auto",
                                    color_continuous_scale='RdBu_r',
                                    title="Матрица факторных нагрузок"
                                )
                                fig_loadings.update_traces(texttemplate="%{z:.2f}", textfont_size=10)
                                st.plotly_chart(fig_loadings, use_container_width=True)
                                
                                # Коммунальности
                                communalities = pd.DataFrame({
                                    'Переменная': selected_columns,
                                    'Коммунальность': fa.get_communalities()
                                })
                                
                                st.markdown("**Коммунальности переменных**")
                                st.dataframe(communalities.round(3), use_container_width=True)
                                
                                # Собственные значения
                                eigenvalues = fa.get_eigenvalues()[0]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Собственные значения факторов**")
                                    eigenvals_df = pd.DataFrame({
                                        'Фактор': [f'Фактор {i+1}' for i in range(len(eigenvalues))],
                                        'Собственное значение': eigenvalues,
                                        'Доля дисперсии': eigenvalues / len(selected_columns)
                                    })
                                    st.dataframe(eigenvals_df.round(3), use_container_width=True)
                                
                                with col2:
                                    # График собственных значений
                                    fig_eigen = go.Figure()
                                    fig_eigen.add_trace(go.Bar(
                                        x=[f'F{i+1}' for i in range(len(eigenvalues))],
                                        y=eigenvalues,
                                        name='Собственные значения'
                                    ))
                                    fig_eigen.add_hline(y=1, line_dash="dash", 
                                                       annotation_text="Критерий Кайзера")
                                    fig_eigen.update_layout(
                                        title="Собственные значения факторов",
                                        xaxis_title="Фактор",
                                        yaxis_title="Собственное значение"
                                    )
                                    st.plotly_chart(fig_eigen, use_container_width=True)
                        
                        # Экспорт результатов
                        st.markdown('<h2 class="section-header">💾 Экспорт результатов</h2>', 
                                   unsafe_allow_html=True)
                        
                        if st.button("Скачать результаты анализа"):
                            # Создание Excel файла с результатами
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # Исходные данные
                                app.data.to_excel(writer, sheet_name='Исходные данные', index=False)
                                
                                # Обработанные данные
                                app.processed_data.to_excel(writer, sheet_name='Обработанные данные', index=False)
                                
                                # Корреляционная матрица
                                corr_matrix.to_excel(writer, sheet_name='Корреляции')
                                
                                if analysis_type == 'PCA' and pca is not None:
                                    # PCA результаты
                                    components_df.to_excel(writer, sheet_name='PCA нагрузки')
                                    
                                    # Объясненная дисперсия
                                    var_df = pd.DataFrame({
                                        'Компонент': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                                        'Дисперсия': pca.explained_variance_ratio_,
                                        'Кумулятивная': np.cumsum(pca.explained_variance_ratio_)
                                    })
                                    var_df.to_excel(writer, sheet_name='PCA дисперсия', index=False)
                                
                                elif analysis_type == 'Factor Analysis' and fa is not None:
                                    # Факторный анализ результаты
                                    loadings_df.to_excel(writer, sheet_name='Факторные нагрузки')
                                    communalities.to_excel(writer, sheet_name='Коммунальности', index=False)
                                    eigenvals_df.to_excel(writer, sheet_name='Собственные значения', index=False)
                            
                            st.download_button(
                                label="📥 Скачать Excel файл с результатами",
                                data=output.getvalue(),
                                file_name=f"factor_analysis_results_{analysis_type.lower()}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                
                else:
                    st.warning("⚠️ Выберите хотя бы одну переменную для анализа")
            else:
                st.error("❌ В файле не найдено числовых столбцов для анализа")
    
    else:
        st.info("👆 Загрузите файл с данными для начала анализа")
        
        # Пример использования
        st.markdown('<h2 class="section-header">📖 Инструкция по использованию</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### Как использовать программу:
        
        1. **Загрузите данные**: Выберите CSV или Excel файл с вашими данными
        2. **Выберите переменные**: Отметьте числовые переменные для анализа
        3. **Настройте предобработку**: Выберите способ обработки пропусков и нормализации
        4. **Проверьте адекватность**: Оцените результаты тестов Барлетта и KMO
        5. **Выберите тип анализа**: PCA или факторный анализ
        6. **Интерпретируйте результаты**: Изучите графики и таблицы
        7. **Экспортируйте результаты**: Скачайте Excel файл с результатами
        
        ### Рекомендации:
        - **KMO > 0.7** - данные подходят для факторного анализа
        - **p-value теста Барлетта < 0.05** - корреляции между переменными значимы
        - **Собственные значения > 1** - критерий Кайзера для выбора количества факторов
        - **80% дисперсии** - рекомендуемый порог для PCA
        """)

if __name__ == "__main__":
    main()