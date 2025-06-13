import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
});

export const authService = {
  login: async (email: string, password: string) => {
    const response = await api.post('/auth/login', { email, password });
    return response.data;
  }
};

export const stockService = {
  getTrending: async () => {
    const response = await api.get('/stocks/trending');
    return response.data;
  },
  getStock: async (symbol: string) => {
    const response = await api.get(`/stocks/${symbol}`);
    return response.data;
  },
  predict: async (symbol: string) => {
    const response = await api.post('/predict', { symbol });
    return response.data;
  }
};

export const portfolioService = {
  getPortfolio: async () => {
    const response = await api.get('/portfolio');
    return response.data;
  },
  trade: async (symbol: string, shares: number, type: string) => {
    const response = await api.post('/portfolio/trade', { symbol, shares, type });
    return response.data;
  }
};