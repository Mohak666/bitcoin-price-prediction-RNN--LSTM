import datetime

now = datetime.datetime.today() + datetime.timedelta(days=1)
print(now.strftime("%Y-%m-%d"))
