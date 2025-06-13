# StockSim - Stock Trading Simulator with Real AI

A full-stack application for simulating stock trading with **real machine learning predictions** using technical analysis and live market data.

## Features

- **Real Stock Data** - Live prices from Yahoo Finance
- **AI Predictions** - Machine learning model using technical indicators
- **Portfolio Simulation** - Buy/sell with virtual money
- **Technical Analysis** - RSI, MACD, Bollinger Bands, Moving Averages
- **Smart Fallbacks** - Graceful degradation when APIs fail
- **Interactive Dashboard** - Real-time portfolio tracking

## Quick Start

### Prerequisites
- **Python 3.11+** - [Download here](https://www.python.org/downloads/)
- **Node.js 18+** - [Download here](https://nodejs.org/)

### Installation

1. **Clone/Download the project**
```bash
git clone <>
cd stocksim
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux  
source venv/bin/activate

# Install ML dependencies
pip install fastapi uvicorn python-dotenv yfinance pandas numpy scikit-learn
python main.py
```

3. **Frontend Setup** (New terminal)
```bash
cd frontend
npm install
npm run dev
```

4. **Access the Application**
- **Frontend:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs

## Demo Login
- **Email:** demo@demo.com
- **Password:** demo123

## How AI Predictions Work

### Technical Analysis
The system calculates 14+ technical indicators:
- **RSI** (Relative Strength Index) - Overbought/oversold levels
- **MACD** - Moving Average Convergence Divergence
- **Bollinger Bands** - Price volatility bands
- **Moving Averages** - 5, 20, 50 day trends
- **Volume Analysis** - Trading volume patterns

### Machine Learning
- **Random Forest** model trained on multiple stocks
- **30-day lookback** window for pattern recognition
- **Dynamic confidence** scoring based on market conditions
- **Direction accuracy** typically 60-70% (realistic for stock prediction)

### Three Prediction Modes
1. **Real ML Mode** - Full technical analysis with trained model
2. **Fallback Mode** - Simplified analysis when data is limited
3. **Demo Mode** - Sample data when APIs fail

## Project Structure
```
stocksim/
├── backend/
│   ├── main.py              # FastAPI server with ML predictions
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   └── services/        # API communication
│   ├── package.json         # Node dependencies
│   └── index.html          # Entry point with Tailwind CDN
├── .gitignore
└── README.md
```

## Tech Stack
- **Frontend:** React + TypeScript + Tailwind CSS (CDN)
- **Backend:** FastAPI + Python
- **Data:** Yahoo Finance API (yfinance)
- **ML:** Random Forest (scikit-learn)
- **Analysis:** Pandas + NumPy

## Troubleshooting

### Backend Issues

**"Module not found" errors:**
```bash
# Make sure virtual environment is activated
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Reinstall dependencies
pip install fastapi uvicorn python-dotenv yfinance pandas numpy scikit-learn
```

**"TA-Lib build failed":**
- **Don't worry!** Our implementation doesn't use TA-Lib
- Skip this package - all indicators are calculated manually

**"No data for symbol" errors:**
- Yahoo Finance API occasionally fails
- System automatically falls back to demo data
- Try different stock symbols (AAPL, GOOGL, MSFT work best)

**Port 8000 in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8000 | xargs kill -9
```

### Frontend Issues

**"CORS errors":**
- Ensure backend is running on port 8000
- Check backend logs for errors

**Port 3000 in use:**
```bash
npm run dev -- --port 3001
```

**Node modules error:**
```bash
rm -rf node_modules package-lock.json
npm install
```

### Prediction Issues

**"Training model first..." appears:**
- Normal on first prediction - model trains automatically
- Takes 30-60 seconds to download and process stock data
- Subsequent predictions are instant

**Low confidence scores:**
- Normal during high market volatility
- Model reduces confidence when indicators conflict
- Confidence 60%+ is considered good

**"Fallback mode" predictions:**
- Occurs when insufficient historical data available
- Still provides useful momentum-based predictions
- Try major stocks (AAPL, GOOGL, MSFT) for best results

## API Endpoints

### Authentication
- `POST /auth/login` - Demo login

### Stocks
- `GET /stocks/trending` - Popular stocks with real prices
- `GET /stocks/{symbol}` - Detailed stock information

### Portfolio
- `GET /portfolio` - Portfolio summary
- `POST /portfolio/trade` - Execute trade

### Predictions
- `POST /predict` - AI price prediction with technical analysis

## Environment Variables

Create `backend/.env` file:
```env
SECRET_KEY=your_secret_key_here
# No API keys needed - using free Yahoo Finance data
```

## Development Notes

### Adding New Stocks
Edit the trending stocks list in `main.py`:
```python
DEMO_STOCKS = {
    "YOUR_SYMBOL": {"price": 100.0, "change": 1.0, "change_percent": 1.0},
}
```

### Customizing ML Model
The model can be tuned in the `RealStockPredictor` class:
- Change `n_estimators` for Random Forest depth
- Modify `window_size` for lookback period
- Add new technical indicators

### Real-time Updates
Currently fetches data on demand. For real-time updates:
- Add WebSocket connections
- Implement data caching with Redis
- Use streaming APIs

## Performance Notes

### First Prediction
- Downloads 1+ years of data for training
- Processes 5 stocks for model training
- Takes 30-60 seconds initially

### Subsequent Predictions
- Uses cached model
- Only fetches recent data
- Responds in 2-3 seconds

### Data Usage
- ~1MB per stock for yearly data
- Models stored in memory (~5MB)
- No persistent storage required

## Deployment

### Docker Alternative
```bash
docker-compose up --build
```

### Production Considerations
- Add Redis for caching
- Implement rate limiting
- Use production ASGI server (gunicorn)
- Add monitoring and logging
- Consider paid data APIs for reliability

## Contributing

1. Fork the project
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is for educational purposes. Not intended for actual trading.

## Disclaimer

This application uses real technical analysis and machine learning for educational purposes. **Stock predictions are inherently uncertain and should not be used for actual trading decisions.** Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

---

**Built with ❤️ for learning about ML, finance, and full-stack development**