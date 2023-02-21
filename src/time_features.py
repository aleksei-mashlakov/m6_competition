import datetime
import time
from datetime import datetime
from typing import Optional

import holidays
import numpy as np
import pandas as pd
from gluonts.time_feature.holiday import (
    BLACK_FRIDAY,
    CHRISTMAS_DAY,
    CHRISTMAS_EVE,
    COLUMBUS_DAY,
    CYBER_MONDAY,
    EASTER_MONDAY,
    EASTER_SUNDAY,
    GOOD_FRIDAY,
    INDEPENDENCE_DAY,
    LABOR_DAY,
    MARTIN_LUTHER_KING_DAY,
    MEMORIAL_DAY,
    MOTHERS_DAY,
    NEW_YEARS_DAY,
    NEW_YEARS_EVE,
    PRESIDENTS_DAY,
    SUPERBOWL,
    THANKSGIVING,
    SpecialDateFeatureSet,
    squared_exponential_kernel,
)
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USFederalHolidayCalendar,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from sklearn.preprocessing import MinMaxScaler

from src.transformers import DateTimeTransformer, periodic_spline_transformer

# Example use for using a squared exponential kernel:
kernel = squared_exponential_kernel(alpha=1.0)
sfs = SpecialDateFeatureSet(
    [
        NEW_YEARS_DAY,
        MARTIN_LUTHER_KING_DAY,
        PRESIDENTS_DAY,
        GOOD_FRIDAY,
        MEMORIAL_DAY,
        INDEPENDENCE_DAY,
        LABOR_DAY,
        THANKSGIVING,
        CHRISTMAS_DAY,
    ],
    squared_exponential_kernel(alpha=1.0),
)

sfs2 = SpecialDateFeatureSet(
    [
        SUPERBOWL,
        CHRISTMAS_EVE,
        EASTER_SUNDAY,
        EASTER_MONDAY,
        MOTHERS_DAY,
        COLUMBUS_DAY,
        NEW_YEARS_EVE,
        BLACK_FRIDAY,
        CYBER_MONDAY,
    ],
    squared_exponential_kernel(alpha=1.0),
)


def reindex_weekdays(
    df: pd.DataFrame,
    drop_weekends: bool = True,
    start_index: pd.Timestamp = None,
    end_index: pd.Timestamp = None,
    fill_method: str = "ffill",
    extra_fill_method: Optional[str] = "bfill",
    freq: str = "D",
) -> pd.DataFrame:
    if start_index is None:
        start_index = df.index[0]
    if end_index is None:
        end_index = df.index[-1]

    df = (
        df.reindex(pd.date_range(start=start_index, end=end_index, freq=freq))
        .fillna(method=fill_method)
        .fillna(method=extra_fill_method)
    )
    if drop_weekends:
        return df.loc[~df.index.day_name().isin(["Saturday", "Sunday"]), :]
    return df


def get_datetime_covariates(
    start_index, end_index, memory_transforms, date_time_transforms
):
    calendar = NYSECalendar()
    index = pd.date_range(start=start_index, end=end_index, freq="D")
    holiday_dates = calendar.holidays(start_index, end_index, return_name=True).index
    covariates = pd.DataFrame(index=index)
    covariates.loc[:, ["one_hot_weekends", "one_hot_holidays"]] = 0
    covariates.loc[covariates.index.isin(holiday_dates), "one_hot_holidays"] = 1
    covariates.loc[
        covariates.index.day_name().isin(["Saturday", "Sunday"]), "one_hot_weekends"
    ] = 1
    covariates.loc[:, "kernel_holidays"] = sfs(covariates.index).max(
        axis=0
    )  # np.prod(sfs(covariates.index), axis=1)
    covariates.loc[:, "kernel_other_holidays"] = sfs2(covariates.index).max(axis=0)
    covariates = covariates.round(3)

    covariates = date_time_transforms.fit_transform(covariates)
    month_splines = periodic_spline_transformer(12, n_splines=6).fit_transform(
        covariates[["month"]]
    )
    weekday_splines = periodic_spline_transformer(7, n_splines=3).fit_transform(
        covariates[["day_of_week"]]
    )
    splines = np.concatenate((month_splines, weekday_splines), axis=1)
    spline_names = [f"spline_{i}" for i in range(splines.shape[1])]
    covariates.loc[:, spline_names] = splines
    covariates = memory_transforms.fit_transform(covariates)

    scaler = MinMaxScaler()  # StandardScaler()
    covariates = pd.DataFrame(
        data=scaler.fit_transform(covariates),
        index=covariates.index,
        columns=covariates.columns,
    )

    return covariates


# https://gist.github.com/jckantor/d100a028027c5a6b8340
class NYSECalendar(AbstractHolidayCalendar):
    """
    cdr = NYSECalendar()
    non_trading_days = cdr.holidays(datetime(2022, 1, 1), datetime(2022, 12, 31))
    """

    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2022-06-20",
            observance=nearest_workday,
        ),
        Holiday("USIndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]



def main():

    us_holidays = holidays.UnitedStates()
    # https://www.commerce.gov/hr/employees/leave/holidays
    us_holidays.observed = False
    us_holidays["2022-01-01":"2022-12-31"]
    # markets were not closed on this day (Juneteenth National Independence Day) in 2021
    if datetime.date(2021, 6, 19) in us_holidays.keys():
        del us_holidays[datetime.date(2021, 6, 19)]
    for date, name in sorted(us_holidays.items()):
        print(date, name)
    #         Thanksgiving
    #         Black Friday
    #         Cyber Monday
    #         Giving Tuesday
    #         Green Monday
    #         Free Shipping Day
    #         Hanukkah (start/end)
    #         Christmas
    #         Kwanzaa
    #         Boxing Day
    #         New Year's Eve/Day


if __name__ == "__main__":
    main()
