import holidays
import datetime
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, USMemorialDay,  USMartinLutherKingJr, USPresidentsDay, GoodFriday, USLaborDay, USThanksgivingDay, nearest_workday, USFederalHolidayCalendar
from datetime import datetime

# https://gist.github.com/jckantor/d100a028027c5a6b8340
class NYSECalendar(AbstractHolidayCalendar):
    """
    cdr = NYSECalendar()
    non_trading_days = cdr.holidays(datetime(2022, 1, 1), datetime(2022, 12, 31))
    """
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-20", observance=nearest_workday),
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
        ]


def main():
    
    us_holidays = holidays.UnitedStates()
    us_holidays.observed = False
    us_holidays['2022-01-01': '2022-12-31']
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
    
if __name__=="__main__":
    main()