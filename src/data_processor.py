import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, 
    RANDOM_SEED, PROCESSED_DATA_DIR,
    DATA_CONFIG
)
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def process_traffic_data(self, df):
        """Process traffic data
        
        Args:
            df (pd.DataFrame): Original traffic data
        
        Returns:
            pd.DataFrame: Processed data
        """
        # Convert time index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Calculate the average speed of all sensors
        df['avg_speed'] = df.mean(axis=1)
        
        # Remove null values
        df = df.dropna()
        
        # Detect and process outliers using the IQR method
        Q1 = df['avg_speed'].quantile(0.25)
        Q3 = df['avg_speed'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out constant values outside the range
        df = df[(df['avg_speed'] >= lower_bound) & (df['avg_speed'] <= upper_bound)]
        
        return df
        
    def process_weather_data(self, df):
        """Process weather data"""
        logging.info("Start processing weather data...")
        
        # Analyze missing value patterns
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        logging.info(f"Total missing values in weather data: {total_missing}")
        logging.info(f"Overview of missing values in each column:\n{missing_summary}")
        
        # Visualize missing value patterns (optional)
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.heatmap(df.isnull(), cbar=False)
        # plt.show()

        # Choose the appropriate filling method according to the situation
        # For example, use forward filling for some features and mean filling for others
        df_filled = df.copy()
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype == 'float':
                    # Use interpolation for continuous variables
                    df_filled[column] = df[column].interpolate(method='time')
                else:
                    # Use forward filling for categorical variables
                    df_filled[column] = df[column].fillna(method='ffill')
        
        # If there are still missing values, use backward filling
        df_filled = df_filled.fillna(method='bfill')
        
        # Check if there are still missing values
        remaining_missing = df_filled.isnull().sum().sum()
        if remaining_missing > 0:
            logging.warning(f"Number of missing values after filling: {remaining_missing}")
        else:
            logging.info("Weather data missing value processing completed")
        
        # Post-processing...
        df = df_filled
        
        # Select important weather features
        selected_features = [
            'TMAX', 'TMIN',  # Temperature
            'PRCP',         # Precipitation
            'AWND',         # Wind speed
            'RHAV',         # Relative humidity
            'ASLP'          # Air pressure
        ]
        
        df = df[selected_features]
        
        # Add derived features
        df['temp_diff'] = df['TMAX'] - df['TMIN']  # Temperature difference
        df['is_raining'] = (df['PRCP'] > 0).astype(int)  # Is it raining
        
        # Process outliers
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        logging.info("Weather data processing completed")
        return df
        
    def align_and_merge_data(self, traffic_df, weather_df):
        """Align and merge traffic and weather data
        
        Args:
            traffic_df: Processed traffic data
            weather_df: Processed weather data
        Returns:
            Merged DataFrame
        """
        logging.info("Start merging data...")
        
        # Ensure consistent time frequency (use 5-minute interval)
        traffic_df = traffic_df.resample('5T').mean()
        weather_df = weather_df.resample('5T').mean()
        
        # Align time index
        common_idx = traffic_df.index.intersection(weather_df.index)
        traffic_df = traffic_df.loc[common_idx]
        weather_df = weather_df.loc[common_idx]
        
        # Merge data
        merged_df = pd.concat([traffic_df, weather_df], axis=1)
        
        # Remove rows containing missing values
        merged_df = merged_df.dropna()
        
        logging.info(f"Data merge completed, final data shape: {merged_df.shape}")
        return merged_df
        
    def create_features(self, df, include_weather=True):
        """Create features"""
        # Ensure data is sorted by time
        df = df.sort_index()
        
        # Create time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Add holiday flag
        # Assume using public holidays in the United States
        import holidays
        us_holidays = holidays.US()
        df['is_holiday'] = df.index.to_series().apply(lambda x: 1 if x in us_holidays else 0)
        
        # Create lag features
        df['speed_lag1'] = df['avg_speed'].shift(1)
        df['speed_lag2'] = df['avg_speed'].shift(2)
        df['speed_lag3'] = df['avg_speed'].shift(3)
        
        # Create longer moving average features
        df['speed_ma5'] = df['avg_speed'].rolling(window=5).mean()
        df['speed_ma10'] = df['avg_speed'].rolling(window=10).mean()
        df['speed_ma15'] = df['avg_speed'].rolling(window=15).mean()
        df['speed_ma30'] = df['avg_speed'].rolling(window=30).mean()
        
        # Weather type encoding
        if include_weather:
            # Assume there is a weather description field in the weather data, such as 'weather_description'
            # Need to encode the weather description as a category
            if 'weather_description' in df.columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df['weather_encoded'] = le.fit_transform(df['weather_description'])
        
        # Remove rows containing NaN
        df = df.dropna()
        
        return df
        
    def split_data(self, data):
        """Split the dataset"""
        # Set the random seed
        np.random.seed(RANDOM_SEED)
        
        # Extract the target variable
        y = data['target']
        # Remove the target variable column
        X = data.drop('target', axis=1)
        
        # Use sklearn's train_test_split, and set the random seed
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1-TRAIN_RATIO), random_state=RANDOM_SEED
        )
        
        # Split the remaining data into validation and test sets
        val_ratio_adjusted = VAL_RATIO / (1-TRAIN_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(TEST_RATIO/(TEST_RATIO+VAL_RATIO)),
            random_state=RANDOM_SEED
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def prepare_data(self, traffic_data, weather_data=None):
        """Prepare experimental data"""
        # Basic data processing
        processed_data = traffic_data.copy()
        
        # Set the target variable (assuming the last column is the target variable)
        processed_data['target'] = processed_data.iloc[:, -1]
        
        if weather_data is not None:
            # Merge weather data
            processed_data = pd.concat([processed_data, weather_data], axis=1)
        
        # Data cleaning
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        processed_data = processed_data.fillna(method='ffill')
        processed_data = processed_data.fillna(method='bfill')
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(processed_data),
            columns=processed_data.columns,
            index=processed_data.index
        )
        
        return scaled_data
        
    def preprocess_weather_features(self, weather_data):
        """Optimize the preprocessing of weather features"""
        # 1. Create composite weather features
        weather_data['temp_range'] = weather_data['TMAX'] - weather_data['TMIN']  # Temperature difference
        weather_data['feels_like'] = weather_data['TMAX'] - 0.55 * (1 - weather_data['RHAV']/100) * (weather_data['TMAX'] - 14.5)  # Feels like temperature
        weather_data['wind_chill'] = 13.12 + 0.6215 * weather_data['TMAX'] - 11.37 * (weather_data['AWND']**0.16) + 0.3965 * weather_data['TMAX'] * (weather_data['AWND']**0.16)  # Wind chill index
        
        # 2. Create weather condition indicators
        weather_data['severe_weather'] = ((weather_data['PRCP'] > weather_data['PRCP'].quantile(0.95)) | 
                                        (weather_data['AWND'] > weather_data['AWND'].quantile(0.95))).astype(int)
        
        # 3. Add interaction terms between time features and weather
        weather_data['hour'] = weather_data.index.hour
        weather_data['is_rush_hour'] = ((weather_data['hour'] >= 7) & (weather_data['hour'] <= 9) | 
                                       (weather_data['hour'] >= 16) & (weather_data['hour'] <= 18)).astype(int)
        weather_data['rush_hour_rain'] = weather_data['is_rush_hour'] * (weather_data['PRCP'] > 0).astype(int)
        
        # 4. Add weather change rate
        weather_data['temp_change'] = weather_data['TMAX'].diff()
        weather_data['precip_change'] = weather_data['PRCP'].diff()
        weather_data['wind_change'] = weather_data['AWND'].diff()
        
        # 5. Add weather trend features
        weather_data['temp_trend'] = weather_data['TMAX'].rolling(window=12).mean()
        weather_data['precip_trend'] = weather_data['PRCP'].rolling(window=12).mean()
        weather_data['wind_trend'] = weather_data['AWND'].rolling(window=12).mean()
        
        # 6. Handle missing values
        weather_data = weather_data.fillna(method='ffill').fillna(method='bfill')
        
        return weather_data

    def prepare_sequences(self, traffic_data, weather_data=None, sequence_length=12):
        """Modify the data preparation process"""
        # Set the random seed
        np.random.seed(RANDOM_SEED)
        
        if weather_data is not None:
            # Preprocess weather features
            weather_data = self.preprocess_weather_features(weather_data)
            
            # Select the most important weather features
            weather_features = [
                'TMAX', 'TMIN', 'PRCP', 'AWND', 'RHAV',
                'temp_range', 'feels_like', 'wind_chill',
                'severe_weather', 'rush_hour_rain'
            ]
            weather_data = weather_data[weather_features]
            
            # Standardize weather features
            weather_data = (weather_data - weather_data.mean()) / weather_data.std()
        
        # Process traffic data
        traffic_processed = self.process_traffic_data(traffic_data)
        
        if weather_data is not None:
            # Merge data
            data = self.align_and_merge_data(traffic_processed, weather_data)
        else:
            data = traffic_processed
            
        # Create features
        data = self.create_features(data, include_weather=(weather_data is not None))
        
        # Standardize data
        data_scaled = self.prepare_data(data)
        
        # Prepare features and target variables
        X = data_scaled.drop('target', axis=1)
        y = data_scaled['target']
        
        # Create sequence data
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X.iloc[i:(i + sequence_length)].values)
            y_sequences.append(y.iloc[i + sequence_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split the data set according to the ratio in the configuration file
        total_samples = len(X_sequences)
        train_size = int(total_samples * TRAIN_RATIO)
        val_size = int(total_samples * VAL_RATIO)
        
        # Split the data
        X_train = X_sequences[:train_size]
        y_train = y_sequences[:train_size]
        
        X_val = X_sequences[train_size:train_size+val_size]
        y_val = y_sequences[train_size:train_size+val_size]
        
        X_test = X_sequences[train_size+val_size:]
        y_test = y_sequences[train_size+val_size:]
        
        logging.info("Data sequence preparation completed")
        logging.info(f"Sequence shape: {X_sequences.shape}")
        logging.info(f"Training set: {X_train.shape}")
        logging.info(f"Validation set: {X_val.shape}")
        logging.info(f"Test set: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def augment_weather_data(self, X_weather, y):
        """Augment weather data"""
        augmented_X = [X_weather]
        augmented_y = [y]
        
        # 1. Add Gaussian noise
        noise = np.random.normal(0, 0.01, X_weather.shape)
        augmented_X.append(X_weather + noise)
        augmented_y.append(y)
        
        # 2. Randomly scale weather features
        scale = np.random.uniform(0.95, 1.05, X_weather.shape)
        augmented_X.append(X_weather * scale)
        augmented_y.append(y)
        
        return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0)