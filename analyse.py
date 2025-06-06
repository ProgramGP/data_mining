import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Attention
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
def load_and_preprocess(population_density_path,population_size_path,urbanization_path,
    employment_path,wage_path,agearch_path,living_path):
    
    # 加载人口密度数据
    popuden_df = pd.read_excel(population_density_path)
    popuden_df.rename(columns={'城市名称': 'city', '年份': 'year', '人口密度（人/平方公里）': 'population_density'}, inplace=True)

    # 加载人口规模数据
    popusize_df = pd.read_excel(population_size_path)
    popusize_df.rename(columns={'城市名称': 'city', '年份': 'year', '常住人口（万人）': 'longterm_population','户籍人口（万人）': 'household_population'}, inplace=True)

    # 加载城镇化率数据
    urb_df = pd.read_excel(urbanization_path)
    urb_df.rename(columns={'城市名称': 'city', '年份': 'year', 'urbanizationRate': 'urbanization_rate'}, inplace=True)
    
    # 加载就业信息数据
    emp_df = pd.read_excel(employment_path, sheet_name='城镇失业率')
    emp_df.rename(columns={'城市名称': 'city', '年份': 'year', 'unemploymentRate': 'unemployment_rate'}, inplace=True)
    
    # 加载从业人员数据
    workers_df = pd.read_excel(employment_path, sheet_name='从业人员数')
    workers_df.rename(columns={'城市名称': 'city', '年份': 'year', 'employeesNumber': 'employees_number'}, inplace=True)
    
    # 加载产业就业数据
    industry_df = pd.read_excel(employment_path, sheet_name='第一、二、三产业就业人数')
    industry_df.rename(columns={
        '城市名称': 'city', 
        '年份': 'year',
        'pi_Employment': 'primary_industry',
        'si_Employment': 'secondary_industry',
        'ti_Employment': 'tertiary_industry'
    }, inplace=True)
    
    # 加载工资数据
    wage_df = pd.read_excel(wage_path, sheet_name='职工平均工资')
    wage_df = wage_df.melt(id_vars=['averageWage'], var_name='city', value_name='average_wage')
    wage_df['year'] = wage_df['averageWage'].str.extract('(\d+)').astype(int)
    wage_df.drop(columns=['averageWage'], inplace=True)

    # 加载年龄结构
    age_df = pd.read_excel(agearch_path)
    age_df.rename(columns={
        '城市': 'city', 
        '年份': 'year',
        '0-14': 'youth_population',
        '15-64': 'teenager_population',
        '65+': 'olds_population'
    }, inplace=True)
    
    # 加载工资数据
    wage_df = pd.read_excel(wage_path, sheet_name='职工平均工资')
    wage_df = wage_df.melt(id_vars=['averageWage'], var_name='city', value_name='average_wage')
    wage_df['year'] = wage_df['averageWage'].str.extract('(\d+)').astype(int)
    wage_df.drop(columns=['averageWage'], inplace=True)

    # 加载生活水平数据
    def load_living_data(sheet_name):
        df = pd.read_excel(living_path, sheet_name=sheet_name, header=None)
        
        # 找到包含"城市名称"的行作为列名
        header_row = df[df.iloc[:, 0].str.contains('城市名称', na=False)].index[0]
        df = pd.read_excel(living_path, sheet_name=sheet_name, header=header_row)
        
        # 清理数据
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # 设置城市名称为索引
        if '城市名称' in df.columns:
            df = df.set_index('城市名称')
        
        # 将列名转换为年份
        df.columns = [col if isinstance(col, str) and col.isdigit() else str(col) for col in df.columns]
        
        # 重置索引并将数据转为长格式
        df = df.reset_index().melt(id_vars=['城市名称'], var_name='year', value_name=sheet_name)
        df['year'] = df['year'].astype(int)
        df.rename(columns={'城市名称': 'city'}, inplace=True)
        
        return df

    # 加载所有生活水平相关表
    living_sheets = {
        '人均可支配收入': 'disposable_income',
        '人均消费支出': 'consumption_expenditure',
        '城镇居民消费支出': 'urban_consumption',
        '农村居民消费支出': 'rural_consumption',
        '城镇居民人均收入': 'urban_income',
        '农村居民人均收入': 'rural_income'
    }
    living_dfs = []
    for sheet_name, col_name in living_sheets.items():
        try:
            df = load_living_data(sheet_name)
            df = df.rename(columns={sheet_name: col_name})
            living_dfs.append(df)
        except Exception as e:
            print(f"Error loading {sheet_name}: {str(e)}")
    
    # 合并生活水平数据
    living_df = living_dfs[0]
    for df in living_dfs[1:]:
        living_df = pd.merge(living_df, df, on=['city', 'year'], how='outer')
    
    # 合并数据
    df = pd.merge(popuden_df, popusize_df, on=['city', 'year'], how='left')
    df = pd.merge(df, urb_df, on=['city', 'year'], how='left')
    df = pd.merge(df, emp_df, on=['city', 'year'], how='left')
    df = pd.merge(df, workers_df, on=['city', 'year'], how='left')
    df = pd.merge(df, industry_df, on=['city', 'year'], how='left')
    df = pd.merge(df, wage_df, on=['city', 'year'], how='left')
    df = pd.merge(df, age_df, on=['city', 'year'], how='left')
    df = pd.merge(df, living_df, on=['city', 'year'], how='left')

    
    # 处理缺失值
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # 特征工程
    # TODO: 增加特征工程计算
    df['non_agricultural_ratio'] = (df['secondary_industry'] + df['tertiary_industry']) / df['employees_number']
    df['industry_balance'] = df['tertiary_industry'] / (df['secondary_industry'] + 1e-6)
    df['wage_per_employee'] = df['average_wage'] / (df['employees_number'] + 1e-6)
    df['productivity'] = df['average_wage'] * df['employees_number'] / 1e6  # 百万为单位
    
    # 滞后特征
    for lag in [1, 2, 3]:
        for col in ['urbanization_rate', 'employees_number', 'average_wage', 'unemployment_rate']:
            df[f'{col}_lag_{lag}'] = df.groupby('city')[col].shift(lag)
    
    # 时间特征
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df.set_index('year', inplace=True)
    
    return df

# 2. 特征选择与数据集划分
def prepare_datasets(df, target_city, test_year=2018, target_var='employees_number'):
    city_df = df[df['city'] == target_city].copy()
    
    # TODO: 增加特征
    # 特征选择 
    base_features = [
        'urbanization_rate', 'unemployment_rate', 'employees_number',
        'primary_industry', 'secondary_industry', 'tertiary_industry',
        'average_wage', 'non_agricultural_ratio', 'industry_balance',
        'wage_per_employee', 'productivity'
    ]
    
    lag_features = [f'{col}_lag_{lag}' 
                   for col in ['urbanization_rate', 'employees_number', 'average_wage', 'unemployment_rate']
                   for lag in [1, 2, 3]]
    
    features = base_features + lag_features
    target = target_var
    
    # 划分训练测试集
    train = city_df[city_df.index.year < test_year]
    test = city_df[city_df.index.year >= test_year]
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, features

# 3. LightGBM模型
def train_lgbm(X_train, y_train):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    models = []
    
    for train_idx, val_idx in tscv.split(X_train):
        train_data = lgb.Dataset(X_train[train_idx], label=y_train.iloc[train_idx])
        val_data = lgb.Dataset(X_train[val_idx], label=y_train.iloc[val_idx])
        
        model = lgb.train(params, 
                         train_data, 
                         valid_sets=[val_data],
                         num_boost_round=500)
                        #  early_stopping_rounds=30,
                        #  verbose_eval=False)
        models.append(model)
    
    return models

# 4. Prophet模型
def train_prophet(df, target_city, target_var='employees_number'):
    city_df = df[df['city'] == target_city].copy().reset_index()
    prophet_df = city_df[['year', target_var]].rename(columns={'year': 'ds', target_var: 'y'})
    
    # 添加外生变量
    for feature in ['urbanization_rate', 'unemployment_rate', 'average_wage']:
        prophet_df[feature] = city_df[feature].values
    
    model = Prophet()
    for feature in ['urbanization_rate', 'unemployment_rate', 'average_wage']:
        model.add_regressor(feature)
    
    model.fit(prophet_df)
    return model

# 5. LSTM模型
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Attention(),
        LSTM(16),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mae')
    return model

# 6. 模型集成与评估
def evaluate_models(models, X_test, y_test, features, scaler=None):
    predictions = []
    
    for model in models:
        if isinstance(model, lgb.Booster):  # LightGBM
            if scaler:
                X_scaled = scaler.transform(X_test[features])
            else:
                X_scaled = X_test
            pred = model.predict(X_scaled)
        elif hasattr(model, 'predict'):  # Prophet
            future = model.make_future_dataframe(periods=len(X_test), include_history=False)
            for feature in ['urbanization_rate', 'unemployment_rate', 'average_wage']:
                future[feature] = X_test[feature].values
            pred = model.predict(future)['yhat'].values
        predictions.append(pred)
    
    ensemble_pred = np.mean(predictions, axis=0)
    mae = mean_absolute_error(y_test, ensemble_pred)
    
    return ensemble_pred, mae

# 7. 可视化结果
def plot_results(y_true, y_pred, years, city_name):
    plt.figure(figsize=(12, 6))
    plt.plot(years, y_true, 'b-', label='Actual')
    plt.plot(years, y_pred, 'r--', label='Predicted')
    plt.title(f'Population Prediction for {city_name}')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.show()

# 主流程
def main():
    # 数据准备
    df = load_and_preprocess(
        population_density_path='./data/人口密度.xlsx',
        population_size_path = './data/人口规模.xlsx',
        urbanization_path='./data/城镇化率.xlsx',
        employment_path='./data/就业信息.xlsx',
        wage_path='./data/工资水平.xlsx',
        agearch_path = './data/年龄结构.xlsx',
        living_path = './data/生活水平.xlsx'
    )
    # 检查数据
    print(df)
    # 这里输出的df似乎有些问题，尤其是年份,可能需要进一步处理

    # TODO: 更改目标城市和目标变量
    target_city = 'city1'  # 可更改为其他城市
    target_var = 'employees_number'  # 可更改为其他目标变量
    
    # 准备数据集
    X_train, y_train, X_test, y_test, scaler, features = prepare_datasets(
        df, target_city, test_year=2018, target_var=target_var
    )
    
    # 训练LightGBM
    lgb_models = train_lgbm(X_train, y_train)
    
    # 训练Prophet
    prophet_model = train_prophet(df, target_city, target_var)
    
    # 准备LSTM数据
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # 训练LSTM
    lstm_model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=8, verbose=0)
    
    # 评估模型
    test_df = df[(df['city'] == target_city) & (df.index.year >= 2018)].copy()
    ensemble_pred, mae = evaluate_models(
        lgb_models + [prophet_model, lstm_model],
        test_df,
        y_test,
        features,
        scaler
    )
    
    print(f"MAE for {target_city}: {mae:.2f}")
    
    # 可视化结果
    results = pd.DataFrame({
        'year': test_df.index.year,
        'actual': y_test,
        'predicted': ensemble_pred
    })
    print(results)
    
    plot_results(
        y_test, 
        ensemble_pred, 
        test_df.index.year, 
        target_city
    )

if __name__ == "__main__":
    main()