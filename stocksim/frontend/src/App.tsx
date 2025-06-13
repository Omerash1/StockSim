import React, { useState } from 'react';
import { Login } from './components/Login';
import { Dashboard } from './components/Dashboard';
import { StockDetail } from './components/StockDetail';

function App() {
  const [user, setUser] = useState(null);
  const [currentView, setCurrentView] = useState('dashboard');
  const [selectedStock, setSelectedStock] = useState('AAPL');

  if (!user) {
    return <Login onLogin={setUser} />;
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-blue-600 text-white p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold">StockSim Demo</h1>
          <div className="space-x-4">
            <button 
              onClick={() => setCurrentView('dashboard')}
              className={`px-3 py-1 rounded ${currentView === 'dashboard' ? 'bg-blue-800' : 'bg-blue-500'}`}
            >
              Dashboard
            </button>
            <button 
              onClick={() => setCurrentView('stock')}
              className={`px-3 py-1 rounded ${currentView === 'stock' ? 'bg-blue-800' : 'bg-blue-500'}`}
            >
              Stock Detail
            </button>
            <button 
              onClick={() => setUser(null)}
              className="px-3 py-1 rounded bg-red-500 hover:bg-red-600"
            >
              Logout
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-6">
        {currentView === 'dashboard' && (
          <Dashboard onSelectStock={(symbol) => {
            setSelectedStock(symbol);
            setCurrentView('stock');
          }} />
        )}
        {currentView === 'stock' && (
          <StockDetail symbol={selectedStock} />
        )}
      </main>
    </div>
  );
}

export default App;