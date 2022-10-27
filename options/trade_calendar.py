import calendar
import datetime


def get_3rd_friday(year, month):
    cal = calendar.Calendar(firstweekday=calendar.SUNDAY)

    month_cal = cal.monthdatescalendar(year, month)
    return [day for week in month_cal 
                for day in week 
                    if day.weekday() == calendar.FRIDAY and day.month == month][2]


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
