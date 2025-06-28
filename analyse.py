import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
# from run_prophet import Prophet
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

    # print(df)
    # 处理缺失值
    # df.fillna(method='ffill', inplace=True)
    # df.fillna(method='bfill', inplace=True)
    # 使用新API实现相同功能
    df = df.ffill().bfill()

    # 额外添加：确保无残留NaN
    print("NaN值统计：")
    print(df.isnull().sum())
    # 特征工程
    # TODO: 增加特征工程计算
    df['non_agricultural_ratio'] = (df['secondary_industry'] + df['tertiary_industry']) / df['employees_number']
    df['industry_balance'] = df['tertiary_industry'] / (df['secondary_industry'] + 1e-6)
    df['wage_per_employee'] = df['average_wage'] / (df['employees_number'] + 1e-6)
    df['productivity'] = df['average_wage'] * df['employees_number'] / 1e6  # 百万为单位
    
    # 滞后特征
    for lag in [1,2,3]:
        for col in ['urbanization_rate', 'average_wage', 'unemployment_rate','household_population']:
            df[f'{col}_lag_{lag}'] = df.groupby('city')[col].shift(lag)
    
    # 时间特征
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df.set_index('year', inplace=True)

    df = pd.get_dummies(df, columns=['city'], prefix='city')
    
    
    return df

# 2. 特征选择与数据集划分
def prepare_datasets(df, target_city, test_year=2018, target_var='household_population'):
    city_df = df[df['city'] == target_city].copy()
    
    # TODO: 增加特征
    # 特征选择 
    base_features = [
        'urbanization_rate', 'unemployment_rate', 'employees_number','longterm_population','household_population',
        'primary_industry', 'secondary_industry', 'tertiary_industry',
        'average_wage', 'non_agricultural_ratio', 'industry_balance',
        'wage_per_employee', 'productivity'
    ]
    
    lag_features = [f'{col}_lag_{lag}' 
                   for col in ['urbanization_rate', 'average_wage', 'unemployment_rate','household_population']
                   for lag in [1,2,3]]
    
    features = base_features + lag_features
    target = target_var
    
    # 划分训练测试集
    train = city_df[city_df.index.year < test_year]
    test = city_df[city_df.index.year >= test_year]
    
    # 处理缺失值
    train = train.ffill().bfill()
    test = test.ffill().bfill()
    
    # 选择目标变量
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    if test.empty:
        print(f"警告: {target_city} 在 {test_year} 及之后无测试数据")
        X_test_scaled = np.empty((0, len(features)))  # 创建空数组
        y_test = pd.Series([], dtype=y_train.dtype)   # 创建空序列
    else:
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, features

# 3. LightGBM模型
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

def train_lgbm(X_train, y_train, X_val=None, y_val=None, use_tscv=False):
    """
    优化后的LightGBM模型训练函数
    
    参数:
    X_train -- 训练集特征
    y_train -- 训练集目标值
    X_val -- 验证集特征 (可选)
    y_val -- 验证集目标值 (可选)
    use_tscv -- 是否使用时序交叉验证 (默认True)
    
    返回:
    单个模型或多个模型的列表
    """
    # 优化后的参数设置
    params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 40,  # 适中的叶子数
    'max_depth': 10,   # 适中的深度
    'learning_rate': 0.01,  # 比原设置更高的学习率
    'num_boost_round': 1000,  # 增加迭代次数
    # 'early_stopping_rounds': 50,  # 早停机制
    'feature_fraction': 0.7,  # 特征采样比例
    'bagging_fraction': 0.7,  # 样本采样比例
    'bagging_freq': 5,  # 采样频率
    'lambda_l1': 0.2,  # 增强L1正则化
    'lambda_l2': 0.2,  # 增强L2正则化
    'min_child_samples': 30,  # 增加叶子节点样本数
    'verbose': -1
    }
    
    # 如果有显式提供的验证集
    if X_val is not None and y_val is not None:
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1500,  # 增加迭代次数
            valid_sets=[val_data],
            early_stopping_rounds=100,  # 启用早停
            verbose_eval=50
        )
        return model
    
    # 使用时序交叉验证
    if use_tscv:
        tscv = TimeSeriesSplit(n_splits=5)  # 增加交叉验证折数
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            print(f"\n训练折叠 {fold+1}/{tscv.n_splits}")
            
            # 创建数据集
            train_data = lgb.Dataset(
                X_train[train_idx], 
                label=y_train.iloc[train_idx]
            )
            val_data = lgb.Dataset(
                X_train[val_idx], 
                label=y_train.iloc[val_idx]
            )
            
            # 训练模型
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1500,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(period=50)
                ]
            )
            models.append(model)
            
            # 打印当前折叠的验证MAE
            val_pred = model.predict(X_train[val_idx])
            val_mae = mean_absolute_error(y_train.iloc[val_idx], val_pred)
            print(f"折叠 {fold+1} 验证MAE: {val_mae:.4f}")
        
        return models
    
    # 如果没有验证集且不使用交叉验证
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500
    )
    return model

# 在原有代码基础上添加以下函数

def prepare_2023_data(df, target_city, scaler, features):
    """
    准备2023年的预测数据
    """
    # 获取目标城市的最新数据（优先使用2022年）
    city_data = df[df['city'] == target_city].copy()
    
    if city_data.empty:
        raise ValueError(f"找不到城市 '{target_city}' 的数据")
    
    # 确保按时间排序
    city_data = city_data.sort_index()
    
    # 智能获取最新数据（不限定2022年）
    if not city_data.empty:
        # 获取最新年份的数据
        latest_year = city_data.index.max().year
        latest_data = city_data[city_data.index.year == latest_year].iloc[-1].copy()
        
        if latest_year < 2022:
            print(f"警告: {target_city} 最新数据年份是 {latest_year}，将使用此数据预测2023年")
    else:
        raise ValueError(f"{target_city} 没有可用数据")
    
    # 创建2023年的数据行
    new_row = latest_data.copy()
    new_row.name = pd.to_datetime('2023-01-01')  # 设置2023年的时间索引
    
    # 更新年份为2023
    new_row['year'] = 2023
    
    # 更新滞后特征 - 使用2022年的值作为2023年的滞后特征
    lag_features = ['urbanization_rate', 'average_wage', 'unemployment_rate','household_population']
    
    for col in lag_features:
        # 滞后1特征 = 2022年的实际值
        new_row[f'{col}_lag_1'] = latest_data[col]
        
        # 滞后2特征 = 2021年的实际值
        if (2021) in city_data.index.year:
            data_2021 = city_data[city_data.index.year == 2021][col].values[0]
            new_row[f'{col}_lag_2'] = data_2021
        else:
            new_row[f'{col}_lag_2'] = latest_data[col]  # 如果没有2021年数据，使用2022年值
            
        # 滞后3特征 = 2020年的实际值
        if (2020) in city_data.index.year:
            data_2020 = city_data[city_data.index.year == 2020][col].values[0]
            new_row[f'{col}_lag_3'] = data_2020
        else:
            new_row[f'{col}_lag_3'] = latest_data[col]  # 如果没有2020年数据，使用2022年值
    
    # 创建包含2023年数据的DataFrame
    future_df = pd.DataFrame([new_row])
    
    # 提取特征并标准化
    X_future = future_df[features]
    X_future_scaled = scaler.transform(X_future)
    
    return X_future_scaled, future_df

def predict_2023_population(model, X_2023):
    """
    预测2023年人口
    """
    # 使用模型进行预测
    if isinstance(model, list):  # 如果是多个模型的列表（交叉验证结果）
        predictions = np.zeros(len(X_2023))
        for m in model:
            predictions += m.predict(X_2023)
        predictions /= len(model)
    else:  # 单个模型
        predictions = model.predict(X_2023)
    
    return predictions[0]

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
    
    # 创建空DataFrame收集所有结果
    all_results = pd.DataFrame()
    target_cities = ['city1', 'city2', 'city3', 'city4', 'city5','city6','city7','city8','city9','city10',
                     'city11','city12','city13','city14','city15','city16','city17','city18','city19','city20'
                     ,'city21','city22','city23','city24','city25','city26','city27','city28','city29','city30'
                     ,'city31','city32','city33','city34','city35','city36','city37','city38','city39','city40']
    for target_city in target_cities:
        target_var = 'household_population'
        
        # 准备数据集
        X_train, y_train, X_test, y_test, scaler, features = prepare_datasets(
            df, target_city, test_year=2023, target_var=target_var
        )
        
        # 训练LightGBM模型
        print(f"训练模型预测{target_city}的常住人口...")
        lgb_models = train_lgbm(X_train, y_train)
        
        # 准备2023年的数据
        X_2023, future_df = prepare_2023_data(df, target_city, scaler, features)
        
        # 预测2023年人口
        prediction_2023 = predict_2023_population(lgb_models, X_2023)
        print(f"{target_city} 2023年常住人口预测值: {prediction_2023:.2f} 万人")
        
        # 保存预测结果
        result_df = pd.DataFrame({
            'city_id': [target_city],
            'year': [2023],
            'pred': [prediction_2023]
        })
        all_results = pd.concat([all_results, result_df], ignore_index=True)
    
    # 循环结束后，一次性保存所有结果
    all_results.to_csv('2023_all_cities_population_prediction.csv', index=False)
    print(f"所有城市预测结果已保存至 2023_all_cities_population_prediction.csv")

if __name__ == "__main__":
    main()