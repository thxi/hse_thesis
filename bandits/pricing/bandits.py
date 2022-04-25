import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
from gym import Env, spaces

from bandits.bandits import BinomialBanditEnv


class PricingBernoulliBanditEnv(Env):
    def __init__(self, num_arms, dist, p_min=1, p_max=17, n_customers=100):
        super(PricingBernoulliBanditEnv, self).__init__()

        self.num_arms = num_arms
        self.dist = dist  # scipy dist
        self.p_min = p_min
        self.p_max = p_max

        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(1)  # no observations, only rewards

        # normalize the prices to [0, 1]
        self.action_to_price = np.linspace(p_min, p_max, num_arms)
        self.mus = 1 - dist.cdf(self.action_to_price)
        self.b_bandit = BinomialBanditEnv(n=n_customers, probs=self.mus)

        self.max_reward = np.max(self.mus * self.action_to_price)

    def step(self, action):
        assert self.b_bandit.action_space.contains(action)

        observation, conversion_reward, done, info = self.b_bandit.step(action)
        price = self.action_to_price[action]
        reward = conversion_reward * price
        return observation, reward, done, info

    def reset(self):
        return 0


def get_avocado_df(avocado_path):
    df = pd.read_csv(avocado_path)
    df = df.drop(columns=["Unnamed: 0"])
    df["date"] = df["Date"].astype("datetime64[ns]")
    df = df.sort_values("Date")
    df = df[df["date"] < "2018-01-01"]
    df = df[df["type"] == "conventional"].reset_index(drop=True)

    df["price"] = df["AveragePrice"]
    df["quantity"] = df["Total Volume"]

    cols = ["date", "price", "quantity", "region"]
    df = df[cols].copy()

    aggregated_regions = [
        "TotalUS",
        "West",
        "SouthCentral",
        "Northeast",
        "Southeast",
        "Midsouth",
        "Plains",
        "GreatLakes",
        "California",
    ]
    df = df[~df.region.isin(aggregated_regions)]
    region_to_volume = df.groupby(["region"]).quantity.sum().sort_values(ascending=False).reset_index()
    good_regions = set(region_to_volume[:20].region) - set(["LosAngeles"])
    df = df[df.region.isin(good_regions)]
    return df


categorical_columns = ["month", "region"]
model_cols = ["price", "region", "year-month", "year", "month"]


def cols_to_categorical(df, categorical_columns):
    df[categorical_columns] = df[categorical_columns].astype("category")


def featurize(df):
    df["year-month"] = df["date"].dt.year * 100 + df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month


class PricingAvocadoBanditEnv(Env):
    def __init__(self, num_arms, avocado_df, region, start_date, model_path="../data/avocado_lgbm_model.txt", T=10000):
        super(PricingAvocadoBanditEnv, self).__init__()

        self.num_arms = num_arms
        self.start_date = start_date
        self.current_date = self.start_date
        self.region = region
        mm_prices = avocado_df[avocado_df.region == region].price.apply(["min", "max"])
        self.p_min = mm_prices["min"]
        self.p_max = mm_prices["max"]
        self.q_std = avocado_df[avocado_df.region == region].quantity.std()
        self.quantity_norm = avocado_df[avocado_df.region == region].quantity.max()

        self.model = lgb.Booster(model_file=model_path)

        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(1)  # no observations, only rewards

        self.action_to_price = np.linspace(self.p_min, self.p_max, num_arms)
        self.max_reward = np.max(self.action_to_price)

        end_date = start_date + pd.Timedelta(T - 1, unit="D")
        dates = pd.date_range(start=start_date, end=end_date)
        predict_df = pd.DataFrame(list(itertools.product(self.action_to_price, dates)), columns=["price", "date"])
        predict_df["region"] = region
        featurize(predict_df)
        cols_to_categorical(predict_df, categorical_columns)
        predict_df["quantity_without_noise"] = self.model.predict(predict_df[model_cols])
        e = np.random.normal(loc=0, scale=self.q_std)
        predict_df["quantity"] = predict_df["quantity_without_noise"] + e
        predict_df["quantity_norm"] = predict_df["quantity"] / self.quantity_norm

        self.predict_df = predict_df

    def step(self, action):
        assert self.action_space.contains(action)

        price = self.action_to_price[action]
        mask = (self.predict_df["date"] == self.current_date) & (np.isclose(self.predict_df["price"], price))
        observation = 0
        conversion_reward = self.predict_df[mask]["quantity_norm"].iloc[0]
        self.current_date += pd.Timedelta(1, unit="D")
        done = False
        info = None
        price = self.action_to_price[action]
        reward = conversion_reward * price
        return observation, reward, done, info

    def reset(self):
        return 0
