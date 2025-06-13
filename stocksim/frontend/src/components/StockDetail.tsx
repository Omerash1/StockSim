import React, { useState, useEffect } from 'react';
import { stockService, portfolioService } from '../services/api';

interface StockDetailProps {
  symbol: string;
}

interface PredictionData {
  symbol: string;
  current_price: number;
  predicted_price: number;
  predicted_change: number;
  predicted_change_percent: number;
  confidence: number;
  model_version: string;
  training_accuracy?: number;
  indicators?: {
    rsi?: number;
    macd?: number;
    bb_position?: number;
    volume_ratio?: number;
    volatility?: number;
  };
}

export const StockDetail: React.FC<StockDetailProps> = ({ symbol }) => {
  const [stock, setStock] = useState<any>(null);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [shares, setShares] = useState(1);
  const [tradeType, setTradeType] = useState<'BUY' | 'SELL'>('BUY');
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);

  // Helper functions for better formatting
  const getConfidenceDisplay = (confidence: number) => {
    const percentage = (confidence * 100);
    
    if (percentage >= 80) {
      return {
        level: 'High',
        color: 'text-green-600',
        bgColor: 'bg-green-50 border-green-200',
        progressColor: 'bg-green-500',
        icon: '🟢',
        description: 'Strong signals indicate reliable prediction'
      };
    } else if (percentage >= 65) {
      return {
        level: 'Good',
        color: 'text-blue-600',
        bgColor: 'bg-blue-50 border-blue-200',
        progressColor: 'bg-blue-500',
        icon: '🔵',
        description: 'Good technical alignment supports this prediction'
      };
    } else if (percentage >= 50) {
      return {
        level: 'Moderate',
        color: 'text-yellow-600',
        bgColor: 'bg-yellow-50 border-yellow-200',
        progressColor: 'bg-yellow-500',
        icon: '🟡',
        description: 'Mixed signals suggest uncertain market conditions'
      };
    } else {
      return {
        level: 'Low',
        color: 'text-red-600',
        bgColor: 'bg-red-50 border-red-200',
        progressColor: 'bg-red-500',
        icon: '🔴',
        description: 'Conflicting indicators suggest high uncertainty'
      };
    }
  };

  const formatPrice = (price: number): string => {
    return price.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  const formatChange = (change: number): string => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`;
  };

  const formatPercentage = (percentage: number): string => {
    const sign = percentage >= 0 ? '+' : '';
    return `${sign}${percentage.toFixed(2)}%`;
  };

  const getIndicatorStatus = (indicator: string, value: number | undefined) => {
    if (!value) return { status: 'N/A', color: 'text-gray-500', description: '' };
    
    switch (indicator) {
      case 'rsi':
        if (value > 70) return { 
          status: 'Overbought', 
          color: 'text-red-600',
          description: 'Stock may be due for a pullback'
        };
        if (value < 30) return { 
          status: 'Oversold', 
          color: 'text-green-600',
          description: 'Stock may be due for a bounce'
        };
        return { 
          status: 'Neutral', 
          color: 'text-gray-700',
          description: 'No strong directional signals'
        };
      
      case 'volume_ratio':
        if (value > 1.5) return { 
          status: 'High', 
          color: 'text-blue-600',
          description: 'Above average trading interest'
        };
        if (value < 0.7) return { 
          status: 'Low', 
          color: 'text-gray-500',
          description: 'Below average trading interest'
        };
        return { 
          status: 'Normal', 
          color: 'text-gray-700',
          description: 'Typical trading volume'
        };
      
      case 'volatility':
        if (value > 5) return { 
          status: 'High', 
          color: 'text-red-600',
          description: 'Large price swings expected'
        };
        if (value < 2) return { 
          status: 'Low', 
          color: 'text-green-600',
          description: 'Stable price movement'
        };
        return { 
          status: 'Normal', 
          color: 'text-gray-700',
          description: 'Typical price volatility'
        };
      
      default:
        return { status: '', color: 'text-gray-700', description: '' };
    }
  };

  useEffect(() => {
    const fetchStock = async () => {
      try {
        const stockData = await stockService.getStock(symbol);
        setStock(stockData);
      } catch (error) {
        console.error('Failed to fetch stock:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStock();
  }, [symbol]);

  const handlePredict = async () => {
    setPredicting(true);
    try {
      const predictionData = await stockService.predict(symbol);
      setPrediction(predictionData);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Prediction failed! Please try again.');
    } finally {
      setPredicting(false);
    }
  };

  const handleTrade = async () => {
    try {
      await portfolioService.trade(symbol, shares, tradeType);
      alert(`${tradeType} order for ${shares} shares of ${symbol} executed!`);
    } catch (error) {
      console.error('Trade failed:', error);
      alert('Trade failed!');
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-16">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Loading stock data...</span>
      </div>
    );
  }

  if (!stock) {
    return (
      <div className="text-center py-16">
        <div className="text-gray-500 text-lg">Stock not found</div>
        <p className="text-gray-400 mt-2">Please check the symbol and try again</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stock Header */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-start mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{stock.symbol}</h1>
            <p className="text-gray-600 text-lg">{stock.name}</p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold text-gray-900">{formatPrice(stock.current_price)}</p>
            <p className={`text-lg font-medium ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatChange(stock.change)} ({formatPercentage(stock.change_percent)})
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-sm text-gray-500">Volume</p>
            <p className="font-semibold">{stock.volume?.toLocaleString() || 'N/A'}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-sm text-gray-500">Market Cap</p>
            <p className="font-semibold">$2.8T</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-sm text-gray-500">P/E Ratio</p>
            <p className="font-semibold">28.5</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-sm text-gray-500">52W High</p>
            <p className="font-semibold">$198.23</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trading Panel */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-900">Trade {symbol}</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Trade Type</label>
              <select 
                value={tradeType} 
                onChange={(e) => setTradeType(e.target.value as 'BUY' | 'SELL')}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="BUY">Buy</option>
                <option value="SELL">Sell</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Shares</label>
              <input
                type="number"
                value={shares}
                onChange={(e) => setShares(parseInt(e.target.value) || 1)}
                min="1"
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Total Cost</label>
              <p className="text-xl font-bold text-gray-900">{formatPrice(stock.current_price * shares)}</p>
            </div>
            
            <button
              onClick={handleTrade}
              className={`w-full py-3 px-4 rounded-md font-medium text-white transition-colors ${
                tradeType === 'BUY' 
                  ? 'bg-green-600 hover:bg-green-700 focus:ring-2 focus:ring-green-500' 
                  : 'bg-red-600 hover:bg-red-700 focus:ring-2 focus:ring-red-500'
              }`}
            >
              {tradeType} {shares} Share{shares !== 1 ? 's' : ''}
            </button>
          </div>
        </div>

        {/* AI Prediction Panel - Cleaned Up */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">🤖 AI Price Prediction</h2>
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">24h Forecast</span>
          </div>
          
          <button
            onClick={handlePredict}
            disabled={predicting}
            className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white font-medium py-3 px-4 rounded-md mb-4 transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500"
          >
            {predicting ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Analyzing Market Data...
              </div>
            ) : (
              'Predict Tomorrow\'s Price'
            )}
          </button>

          {prediction && (
            <div className="space-y-4">
              {/* Current vs Predicted */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4 text-center">
                  <p className="text-sm text-gray-500 mb-1">Current Price</p>
                  <p className="text-xl font-bold text-gray-900">{formatPrice(prediction.current_price)}</p>
                </div>
                
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <p className="text-sm text-purple-600 mb-1">Tomorrow's Prediction</p>
                  <p className="text-xl font-bold text-purple-900">
                    {formatPrice(prediction.predicted_price)}
                  </p>
                  <p className={`text-sm font-medium ${
                    prediction.predicted_change >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatChange(prediction.predicted_change)} 
                    ({formatPercentage(prediction.predicted_change_percent)})
                  </p>
                </div>
              </div>

              {/* Simplified Confidence Display */}
              <div className={`rounded-lg p-4 border ${getConfidenceDisplay(prediction.confidence).bgColor}`}>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-gray-700">Prediction Confidence</h4>
                  <span className="text-lg">{getConfidenceDisplay(prediction.confidence).icon}</span>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="flex-1">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-500 ${getConfidenceDisplay(prediction.confidence).progressColor}`}
                        style={{ width: `${prediction.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`text-sm font-semibold ${getConfidenceDisplay(prediction.confidence).color}`}>
                      {getConfidenceDisplay(prediction.confidence).level}
                    </p>
                    <p className={`text-xs ${getConfidenceDisplay(prediction.confidence).color}`}>
                      {(prediction.confidence * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
                
                <div className="mt-2">
                  <p className="text-xs text-gray-600">
                    {getConfidenceDisplay(prediction.confidence).description}
                  </p>
                </div>
              </div>

              {/* Simplified Technical Indicators */}
              {prediction.indicators && Object.keys(prediction.indicators).length > 0 && (
                <div className="bg-blue-50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-blue-800 mb-3">Key Indicators</h4>
                  <div className="space-y-3">
                    {prediction.indicators.rsi && (
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Market Sentiment (RSI)</span>
                        <div className="text-right">
                          <span className={`text-sm font-semibold ${getIndicatorStatus('rsi', prediction.indicators.rsi).color}`}>
                            {getIndicatorStatus('rsi', prediction.indicators.rsi).status}
                          </span>
                          <p className="text-xs text-gray-500">{getIndicatorStatus('rsi', prediction.indicators.rsi).description}</p>
                        </div>
                      </div>
                    )}
                    
                    {prediction.indicators.volume_ratio && (
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Trading Activity</span>
                        <div className="text-right">
                          <span className={`text-sm font-semibold ${getIndicatorStatus('volume_ratio', prediction.indicators.volume_ratio).color}`}>
                            {getIndicatorStatus('volume_ratio', prediction.indicators.volume_ratio).status}
                          </span>
                          <p className="text-xs text-gray-500">{getIndicatorStatus('volume_ratio', prediction.indicators.volume_ratio).description}</p>
                        </div>
                      </div>
                    )}
                    
                    {prediction.indicators.volatility && (
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Price Stability</span>
                        <div className="text-right">
                          <span className={`text-sm font-semibold ${getIndicatorStatus('volatility', prediction.indicators.volatility).color}`}>
                            {getIndicatorStatus('volatility', prediction.indicators.volatility).status}
                          </span>
                          <p className="text-xs text-gray-500">{getIndicatorStatus('volatility', prediction.indicators.volatility).description}</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Simplified Model Status */}
              {prediction.model_version.includes('real_ml') && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <span className="text-green-600">✅</span>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-green-800">
                        <strong>AI Model Active:</strong> Using machine learning trained on historical market data.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Simplified Disclaimer */}
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
                <p className="text-xs text-yellow-800">
                  <strong>Educational Tool:</strong> This prediction is for learning purposes only and should not be considered financial advice. 
                  Stock markets are unpredictable. Always do your own research and consult with a financial advisor before investing.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Simplified Info Section */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900">How AI Predictions Work</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-800 mb-2">📊 Data Analysis</h3>
            <p className="text-blue-700">
              Analyzes price patterns, trading volume, and market trends 
              from recent trading sessions to identify potential movements.
            </p>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4">
            <h3 className="font-semibold text-green-800 mb-2">🤖 Machine Learning</h3>
            <p className="text-green-700">
              Uses AI trained on historical stock data to recognize patterns 
              and predict next-day price changes with measurable accuracy.
            </p>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4">
            <h3 className="font-semibold text-purple-800 mb-2">🎯 24-Hour Forecast</h3>
            <p className="text-purple-700">
              Provides predictions for tomorrow's closing price based on current 
              market conditions and technical indicators.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};