import React, { useState, useEffect } from 'react';
import { stockService, portfolioService } from '../services/api';

interface DashboardProps {
  onSelectStock: (symbol: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onSelectStock }) => {
  const [stocks, setStocks] = useState<any[]>([]);
  const [portfolio, setPortfolio] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [stocksData, portfolioData] = await Promise.all([
          stockService.getTrending(),
          portfolioService.getPortfolio()
        ]);
        setStocks(stocksData);
        setPortfolio(portfolioData);
      } catch (error) {
        console.error('Failed to fetch data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <div className="text-center py-8">Loading...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Portfolio Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-500">Total Value</p>
            <p className="text-2xl font-bold text-green-600">
              ${portfolio?.total_value?.toLocaleString() || '0'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-500">Cash Balance</p>
            <p className="text-2xl font-bold">
              ${portfolio?.cash_balance?.toLocaleString() || '0'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-500">Day's Change</p>
            <p className={`text-2xl font-bold ${portfolio?.day_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ${portfolio?.day_change?.toLocaleString() || '0'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-500">Positions</p>
            <p className="text-2xl font-bold">
              {portfolio?.positions?.length || 0}
            </p>
          </div>
        </div>
      </div>

      {/* Trending Stocks */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Trending Stocks</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {stocks.map((stock: any) => (
            <div key={stock.symbol} className="border rounded-lg p-4 hover:shadow-md cursor-pointer"
                 onClick={() => onSelectStock(stock.symbol)}>
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-semibold text-lg">{stock.symbol}</h3>
                  <p className="text-gray-500 text-sm">Stock Corporation</p>
                </div>
                <div className="text-right">
                  <p className="font-semibold">${stock.price}</p>
                  <p className={`text-sm ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {stock.change >= 0 ? '+' : ''}{stock.change} ({stock.change_percent}%)
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Current Positions */}
      {portfolio?.positions && portfolio.positions.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Your Positions</h2>
          <div className="space-y-3">
            {portfolio.positions.map((position: any) => (
              <div key={position.symbol} className="flex justify-between items-center p-3 border rounded">
                <div>
                  <span className="font-semibold">{position.symbol}</span>
                  <span className="text-gray-500 ml-2">{position.shares} shares</span>
                </div>
                <div className="text-right">
                  <p className="font-semibold">Avg: ${position.avg_cost}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};