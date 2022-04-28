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
    good_regions = set(region_to_volume[:20].region) - set(["LosAngeles", "NewYork"])
    df = df[df.region.isin(good_regions)]
    return df


class PricingAvocadoBanditEnv(Env):
    def __init__(
        self,
        num_arms,
        avocado_df,
        region,
        start_date,
        model_path="../data/avocado_lgbm_model.txt",
        T=10000,
        p_min=0.1,
        p_max=1,
    ):
        super(PricingAvocadoBanditEnv, self).__init__()

        self.num_arms = num_arms
        self.start_date = start_date
        self.current_idx = 0
        self.region = region
        mm_prices = avocado_df[avocado_df.region == region].price.apply(["min", "max"])
        self.p_min_dataset = mm_prices["min"]
        self.p_max_dataset = mm_prices["max"]
        self.p_min_scale = p_min
        self.p_max_scale = p_max

        self.model = lgb.Booster(model_file=model_path)

        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(1)  # no observations, only rewards

        self.action_to_price = np.linspace(self.p_min_scale, self.p_max_scale, num_arms)
        self.action_to_price_dataset = np.linspace(self.p_min_dataset, self.p_max_dataset, num_arms)

        self._prepare_predict_df(avocado_df, T)

    def step(self, action):
        assert self.action_space.contains(action)

        price = self.action_to_price_dataset[action]
        predict_df = self.price_to_predict_df[price]
        observation = 0
        conversion_reward = predict_df.iloc[self.current_idx, :]["quantity_norm"]
        # print(predict_df.iloc[self.current_idx, :])
        self.current_idx += 1
        done = False
        info = None
        price = self.action_to_price[action]
        reward = conversion_reward * price
        return observation, reward, done, info

    def reset(self):
        return 0

    def _prepare_predict_df(self, avocado_df, T):
        # Preparing the prediction dataframe from which the rewards will be drawn
        # basically, just predicting the grid of [prices, dates]

        def cols_to_categorical(df, categorical_columns):
            df[categorical_columns] = df[categorical_columns].astype("category")

        def featurize(df):
            df["year-month"] = df["date"].dt.year * 100 + df["date"].dt.month
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

        end_date = self.start_date + pd.Timedelta(T - 1, unit="D")
        dates = pd.date_range(start=self.start_date, end=end_date)
        predict_df = pd.DataFrame(
            list(itertools.product(self.action_to_price_dataset, dates)),
            columns=["price", "date"],
        )
        predict_df["region"] = self.region
        featurize(predict_df)
        categorical_columns = ["region"]
        cols_to_categorical(predict_df, categorical_columns)
        model_cols = ["price", "region"]
        predict_df["quantity_without_noise"] = self.model.predict(predict_df[model_cols])
        self.q_std = avocado_df[avocado_df.region == self.region].quantity.std()
        self.quantity_norm = avocado_df[avocado_df.region == self.region].quantity.max()
        e = np.random.normal(loc=0, scale=self.q_std, size=predict_df.shape[0]) / 5
        predict_df["quantity"] = predict_df["quantity_without_noise"] + e
        predict_df["quantity_norm"] = predict_df["quantity"] / self.quantity_norm
        predict_df["quantity_norm"] = predict_df["quantity"] / self.quantity_norm
        means = predict_df.groupby("price")["quantity_norm"].mean().reset_index()
        means["mean_reward"] = means["quantity_norm"] * self.action_to_price
        self.max_reward = np.max(means["mean_reward"])
        self.predict_df = predict_df

        # splitting the dataframe into slices based on prices
        # would speed up the self.step() significantly
        self.price_to_predict_df = {}
        for p in self.action_to_price_dataset:
            mask = np.isclose(self.predict_df["price"], p)
            self.price_to_predict_df[p] = self.predict_df[mask]
