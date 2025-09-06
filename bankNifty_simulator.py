import numpy as np
import pandas as pd

from bankNifty_feature_engineering import FeatureEngineer
from bankNifty_ML_model import create_encoder, create_actor_critic

# Placeholder for data loading function
def load_backtest_data(file_path):
    """
    Loads and prepares data for backtesting.
    This is a placeholder and needs to be implemented.
    """
    print("Loading dummy data for backtesting...")
    dates = pd.date_range(start='2023-01-01', periods=20000, freq='1min')
    data = np.random.rand(20000, 5)
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'], index=dates)
    return df

class BacktestSimulator:
    def __init__(self, df, actor, sequence_length=60):
        self.df = df
        self.actor = actor
        self.sequence_length = sequence_length
        self.feature_cols = [col for col in df.columns if '_norm' in col]

    def _get_state(self, step):
        return self.df[self.feature_cols].iloc[step - self.sequence_length:step].values

    def run(self):
        capital = 100000
        equity_curve = [capital]
        position = 0 # 0: cash, 1: long, -1: short
        wins = 0
        losses = 0

        for i in range(self.sequence_length, len(self.df) - 1):
            state = self._get_state(i)
            action_probs = self.actor.predict(np.expand_dims(state, axis=0))[0]
            action = np.argmax(action_probs) # 0: Long, 1: Cash, 2: Short

            current_price = self.df['close'].iloc[i]
            pnl = 0

            if action == 0 and position != 1: # Go Long
                position = 1
            elif action == 2 and position != -1: # Go Short
                position = -1
            elif action == 1:
                position = 0

            if position != 0:
                next_price = self.df['close'].iloc[i+1]
                if position == 1:
                    pnl = next_price - current_price
                else:
                    pnl = current_price - next_price
                
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

            capital += pnl
            equity_curve.append(capital)

        return equity_curve, wins, losses

def calculate_metrics(equity_curve):
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() # Annualized for daily returns, adjust for 1-min
    max_drawdown = (pd.Series(equity_curve).cummax() - pd.Series(equity_curve)).max()
    return sharpe_ratio, max_drawdown

def backtest_model():
    """
    Main function to run the backtest.
    """
    # Load and process data
    df = load_backtest_data('path/to/your/data.csv')
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.add_features(df)

    # Load the trained actor model
    num_features = len([col for col in df_features.columns if '_norm' in col])
    encoder = create_encoder(input_shape=(60, num_features))
    actor, _ = create_actor_critic(encoder)
    # actor.load_weights('actor_weights.h5') # Load trained weights here

    # Run the backtest
    simulator = BacktestSimulator(df_features, actor)
    equity_curve, wins, losses = simulator.run()

    # Calculate and print metrics
    sharpe, drawdown = calculate_metrics(equity_curve)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    print(f"Backtest Results:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {drawdown:.2f}")
    print(f"Win Rate: {win_rate:.2%}")

if __name__ == '__main__':
    backtest_model()
