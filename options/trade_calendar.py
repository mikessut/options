import calendar
import datetime
import holidays
import pytz
import yaml
import pathlib


denver_tz = pytz.timezone("America/Denver")

nyse_holidays = holidays.NYSE()

OVERRIDE_DAYS = yaml.safe_load(open(pathlib.Path(__file__).parent / 'override.yml'))


def get_3rd_friday(year, month) -> datetime.date:
    cal = calendar.Calendar(firstweekday=calendar.SUNDAY)

    month_cal = cal.monthdatescalendar(year, month)
    date = [day for week in month_cal 
                for day in week 
                    if day.weekday() == calendar.FRIDAY and day.month == month][2]
    return OVERRIDE_DAYS.get(date, date)


def get_next_3rd_friday(start: datetime.date) -> datetime.date:
    this_mon = get_3rd_friday(start.year, start.month)
    if this_mon > start:
        return this_mon
    else:
        return get_3rd_friday(*next_month(start.year, start.month))
    

def get_next_friday(start: datetime.date) -> datetime.date:
    day = start
    while day.weekday() != calendar.FRIDAY:
        day += datetime.timedelta(days=1)
    return OVERRIDE_DAYS.get(day, day)


def next_month(year, month):
    month = month + 1
    if month > 12:
        month = 1
        year += 1
    return year, month


def get_n_3rd_fridays(start: datetime.date, num):
    if isinstance(start, datetime.datetime):
        start = start.date()
    results = []
    this_month = get_3rd_friday(start.year, start.month)
    if this_month > start:
        results.append(this_month)
    year = start.year
    month = start.month

    while len(results) < num:
        year, month = next_month(year, month)
        results.append(get_3rd_friday(year, month))

    return results


def iter_trade_days(start: datetime.date, direction=1):
    assert direction in [1, -1]
    next = start
    while True:
        while (next.weekday() == calendar.SATURDAY
               or next.weekday() == calendar.SUNDAY
               or next in nyse_holidays):
            next += datetime.timedelta(days=1) * direction
        yield next
        next += datetime.timedelta(days=1) * direction


def next_trade_day(start: datetime.date):
    i = iter_trade_days(start)
    next(i)
    return next(i)


def prev_trade_day(start: datetime.date):
    i = iter_trade_days(start, -1)
    next(i)
    return next(i)


def trade_days(start: datetime.date, stop: datetime.date) -> list[datetime.date]:
    """
    Not inclusive of stop
    """
    results = []
    i = iter_trade_days(start)
    day = next(i)
    while day < stop:
        results.append(day)
        day = next(i)
    return results


def market_open(date: datetime.datetime=None) -> bool:
    """
    tz aware date

    Naive... Won't account for half days.
    """
    if date is None:
        date = pytz.utc.localize(datetime.datetime.utcnow())
    date_den = date.astimezone(denver_tz)
    if (date_den.date() not in nyse_holidays and
        (date_den.time() >= datetime.time(7, 30)) and
        (date_den.time() < datetime.time(14, 0))):
        return True
    else:
        return False
