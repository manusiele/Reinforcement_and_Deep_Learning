# Stock Trend Forecasting with LSTM

An LSTM-based deep learning model for predicting next-day stock price direction (up/down) using TensorFlow/Keras.

## Project Overview

This project implements a binary classification LSTM model to predict whether a stock's closing price will go up or down the next day based on historical price sequences.

## Objective

Build an LSTM model to predict the next day's closing price direction using:
- Historical daily closing prices
- Sequence-based learning (15-day windows)
- Binary classification (up=1, down=0)

## Model Architecture

### LSTM Design:
- **LSTM Layer 1:** 64 units, return sequences
- **Dropout 1:** 0.3 rate
- **LSTM Layer 2:** 32 units
- **Dropout 2:** 0.3 rate
- **Dense Layer:** 16 units, ReLU activation
- **Dropout 3:** 0.2 rate
- **Output Layer:** 1 unit, Sigmoid activation

### Architecture Rationale:
- **Two LSTM layers** capture temporal dependencies at different scales
- **Dropout layers** prevent overfitting (0.3, 0.3, 0.2)
- **Sigmoid output** produces probability for binary classification
- **Binary cross-entropy loss** optimizes for direction prediction

## Dataset

- **Source:** Yahoo Finance (via yfinance)
- **Stock:** AAPL (Apple Inc.)
- **Period:** 2019-01-01 to 2024-12-31 (5 years)
- **Feature:** Closing prices only
- **Sequence length:** 15 days
- **Split:** 70% train / 15% validation / 15% test

## Getting Started

### Prerequisites

```bash
pip install yfinance tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### Running on Google Colab

1. Upload `stock_lstm_forecasting.ipynb` to Google Colab
2. Run all cells sequentially
3. All dependencies will be installed automatically



##  Training Details

- **Epochs:** 50 (with early stopping)
- **Batch size:** 32
- **Optimizer:** Adam
- **Loss function:** Binary Cross-Entropy
- **Validation split:** 15%
- **Early stopping:** Patience of 10 epochs
- **Training time:** ~3-5 minutes on GPU

## Evaluation Metrics

### Model Performance:
- Test accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Training/validation curves

### Baseline Comparisons:
- Always predict "UP"
- Always predict "DOWN"
- Random prediction
- LSTM model improvement over baseline

## Key Findings

### Sequence Length Choice (15 days):
- Captures approximately 3 weeks of trading patterns
- Balance between capturing trends and avoiding noise
- Too short (5 days): May miss longer-term patterns
- Too long (30+ days): May include irrelevant historical data

### Preprocessing Decisions:
- **MinMax Scaling:** Normalizes prices to [0,1] range for stable training
- **Binary Labels:** Simplifies problem to direction prediction
- **Sequential Split:** Maintains temporal order (no shuffling)

### Why Stock Prediction is Challenging:

1. **Market Efficiency Hypothesis:** Prices already reflect all available information
2. **Random Walk Theory:** Stock prices follow largely random patterns
3. **External Factors:** News, sentiment, macroeconomic events not captured in price data
4. **Non-Stationarity:** Market dynamics and patterns change over time
5. **High Noise-to-Signal Ratio:** Daily price movements contain significant noise
6. **Limited Features:** Using only closing prices ignores volume, volatility, etc.
7. **Overfitting Risk:** Model may learn historical patterns that don't generalize

### Is LSTM Better Than Baseline?

The model's performance should be evaluated against simple baselines:
- If improvement > 5%: Model learned meaningful patterns
- If improvement 0-5%: Marginal improvement, may not be statistically significant
- If improvement < 0%: Model underperforms, likely overfitting or market too random

## Potential Improvements

1. **Additional Features:**
   - Trading volume
   - Moving averages (SMA, EMA)
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Volatility measures

2. **Advanced Techniques:**
   - Sentiment analysis from news/social media
   - Attention mechanisms to focus on important time steps
   - Ensemble methods combining multiple models
   - Transformer architecture instead of LSTM
   - Multi-stock training for better generalization

3. **Hyperparameter Tuning:**
   - Different sequence lengths (10, 20, 30 days)
   - Various LSTM unit sizes
   - Learning rate scheduling
   - Different dropout rates

4. **Data Augmentation:**
   - Multiple stocks from same sector
   - Different time periods
   - Cross-validation across time windows

## Important Disclaimer

**This model is for EDUCATIONAL purposes ONLY:**
- Past performance does NOT guarantee future results
- DO NOT use this model for actual trading decisions
- Financial markets are complex and unpredictable
- Always consult with financial professionals before investing
- The creators assume NO responsibility for financial losses

## Project Structure

```
stock-lstm-forecasting/
│
├── stock_lstm_forecasting.ipynb  # Complete notebook with all code
├── README.md                      # Project documentation
└── stock_lstm_model.h5           # Saved model (generated after training)
```

## Academic Context

**Course:** Deep Learning / Neural Networks  
**Task:** Practical Task 2 - RNN/LSTM for Stock Trend Forecasting  
**Marks:** 10  

## References

- Yahoo Finance API: https://finance.yahoo.com/
- yfinance Library: https://pypi.org/project/yfinance/
- TensorFlow Documentation: https://www.tensorflow.org/
- LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## License

This project is created for educational purposes.

## 👤 Author

Created as part of a deep learning practical assignment.

---

**Note:** This implementation is optimized for Google Colab but can run locally with appropriate environment setup.
