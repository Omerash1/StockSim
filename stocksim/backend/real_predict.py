import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RealStockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def get_stock_data(self, symbol: str, period: str = "1y"):
        """Get real stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for prediction"""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # High/Low ratios
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_to_High'] = df['Close'] / df['High']
        df['Close_to_Low'] = df['Close'] / df['Low']
        
        return df
    
    def prepare_features(self, data):
        """Prepare features for machine learning model"""
        df = self.calculate_technical_indicators(data)
        
        # Feature columns
        feature_columns = [
            'SMA_5', 'SMA_20', 'SMA_50', 'Volatility',
            'Price_Change', 'Price_Change_5d', 'Volume_Ratio',
            'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
            'High_Low_Ratio', 'Close_to_High', 'Close_to_Low'
        ]
        
        # Create features and targets
        features = []
        targets = []
        
        # Use 30 days of data to predict next day
        window_size = 30
        
        for i in range(window_size, len(df) - 1):
            # Features: last 30 days of indicators
            feature_window = df.iloc[i-window_size:i][feature_columns].values.flatten()
            features.append(feature_window)
            
            # Target: next day's price change percentage
            current_price = df.iloc[i]['Close']
            next_price = df.iloc[i + 1]['Close']
            price_change = (next_price - current_price) / current_price
            targets.append(price_change)
        
        return np.array(features), np.array(targets)
    
    def train_model(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']):
        """Train the model on multiple stocks"""
        print("Training prediction model...")
        
        all_features = []
        all_targets = []
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            data = self.get_stock_data(symbol, period="2y")  # 2 years of data
            
            if data is not None and len(data) > 100:
                features, targets = self.prepare_features(data)
                all_features.extend(features)
                all_targets.extend(targets)
        
        if len(all_features) > 0:
            X = np.array(all_features)
            y = np.array(all_targets)
            
            # Handle NaN values
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            if len(X) > 0:
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.model.fit(X_scaled, y)
                self.is_trained = True
                print(f"Model trained on {len(X)} samples")
                
                # Calculate accuracy
                predictions = self.model.predict(X_scaled)
                direction_accuracy = np.mean(np.sign(predictions) == np.sign(y))
                print(f"Direction accuracy: {direction_accuracy:.2%}")
                
                return True
        
        print("Failed to train model - insufficient data")
        return False
    
    def predict_stock(self, symbol: str, days: int = 1):
        """Make real prediction for a stock"""
        if not self.is_trained:
            print("Training model first...")
            if not self.train_model():
                return self._fallback_prediction(symbol)
        
        try:
            # Get recent data
            data = self.get_stock_data(symbol, period="6mo")
            
            if data is None or len(data) < 50:
                return self._fallback_prediction(symbol)
            
            # Calculate features
            df = self.calculate_technical_indicators(data)
            
            feature_columns = [
                'SMA_5', 'SMA_20', 'SMA_50', 'Volatility',
                'Price_Change', 'Price_Change_5d', 'Volume_Ratio',
                'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
                'High_Low_Ratio', 'Close_to_High', 'Close_to_Low'
            ]
            
            # Get last 30 days for prediction
            recent_features = df.tail(30)[feature_columns].values.flatten()
            
            # Handle NaN values
            if np.isnan(recent_features).any():
                return self._fallback_prediction(symbol)
            
            # Scale and predict
            recent_features_scaled = self.scaler.transform([recent_features])
            predicted_change = self.model.predict(recent_features_scaled)[0]
            
            # Get current price
            current_price = float(data['Close'].iloc[-1])
            predicted_price = current_price * (1 + predicted_change)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(df, predicted_change)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': predicted_change * current_price,
                'predicted_change_percent': predicted_change * 100,
                'confidence': confidence,
                'model_version': 'real_ml_v1.0',
                'indicators': self._get_current_indicators(df)
            }
            
        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
            return self._fallback_prediction(symbol)
    
    def _calculate_confidence(self, df, predicted_change):
        """Calculate prediction confidence based on market conditions"""
        try:
            # Recent volatility
            recent_volatility = df['Volatility'].tail(5).mean()
            volatility_factor = max(0.3, 1 - (recent_volatility / df['Close'].iloc[-1]) * 10)
            
            # RSI extremes (more confident at extremes)
            rsi = df['RSI'].iloc[-1]
            rsi_factor = 0.7
            if rsi > 70 or rsi < 30:  # Overbought or oversold
                rsi_factor = 0.9
            
            # Volume confirmation
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            volume_factor = min(1.0, max(0.5, volume_ratio / 2))
            
            # Trend consistency
            price_changes = df['Price_Change'].tail(5)
            trend_consistency = 1 - abs(price_changes.std())
            trend_factor = max(0.4, min(1.0, trend_consistency))
            
            # Combine factors
            base_confidence = 0.65
            confidence = base_confidence * volatility_factor * rsi_factor * volume_factor * trend_factor
            
            return max(0.3, min(0.95, confidence))
            
        except:
            return 0.65
    
    def _get_current_indicators(self, df):
        """Get current technical indicators for display"""
        try:
            latest = df.iloc[-1]
            return {
                'rsi': round(latest['RSI'], 1),
                'macd': round(latest['MACD'], 3),
                'bb_position': round(latest['BB_Position'], 2),
                'volume_ratio': round(latest['Volume_Ratio'], 2),
                'volatility': round(latest['Volatility'], 2)
            }
        except:
            return {}
    
    def _fallback_prediction(self, symbol):
        """Simple fallback when ML fails"""
        # Get basic price for fallback
        try:
            data = self.get_stock_data(symbol, period="5d")
            current_price = float(data['Close'].iloc[-1])
            
            # Simple momentum-based prediction
            recent_change = data['Close'].pct_change().tail(3).mean()
            predicted_change = recent_change * 0.5  # Assume some continuation
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': current_price * (1 + predicted_change),
                'predicted_change': predicted_change * current_price,
                'predicted_change_percent': predicted_change * 100,
                'confidence': 0.45,
                'model_version': 'fallback_v1.0',
                'indicators': {}
            }
        except:
            return {
                'symbol': symbol,
                'current_price': 100.0,
                'predicted_price': 102.0,
                'predicted_change': 2.0,
                'predicted_change_percent': 2.0,
                'confidence': 0.3,
                'model_version': 'error_fallback',
                'indicators': {}
            }