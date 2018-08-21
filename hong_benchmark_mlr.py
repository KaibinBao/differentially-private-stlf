import pandas as pd
import numpy as np
import matplotlib
if not matplotlib.is_interactive(): # avoid warnings when this is used in jupyter
    matplotlib.use("AGG")
import matplotlib.pyplot as plt
import os

GEFCOM_DATA_PATH = "."

def load_elec_data():
    load_history = pd.read_csv(os.path.join(GEFCOM_DATA_PATH,"GEFCOM2012_Data","Load","Load_history.csv"), sep=',', thousands=',')
    load_history["date"] = pd.to_datetime(load_history["year"].astype(str) + "-" + load_history["month"].astype(str) + "-" + load_history["day"].astype(str), format="%Y-%m-%d")
    del load_history["year"]
    del load_history["month"]
    del load_history["day"]
    df = load_history.set_index(["zone_id","date"])
    del load_history
    df.set_axis(pd.to_timedelta([int(v[1:])-1 for v in df.columns.values], 'h'), axis=1, inplace=True)
    df = df.stack().reset_index()
    df["datetime"] = df["date"] + df["level_2"]
    del df["date"]
    del df["level_2"]
    df = df.set_index(["zone_id", "datetime"])
    df.set_axis(["kWh"], axis=1, inplace=True)
    return df


def load_temp_data():
    temp_history = pd.read_csv(os.path.join(GEFCOM_DATA_PATH,"GEFCOM2012_Data","Load","temperature_history.csv"), sep=',', thousands=',')
    temp_history["date"] = pd.to_datetime(temp_history["year"].astype(str) + "-" + temp_history["month"].astype(str) + "-" + temp_history["day"].astype(str), format="%Y-%m-%d")
    del temp_history["year"]
    del temp_history["month"]
    del temp_history["day"]
    df = temp_history.set_index(["station_id","date"])
    del temp_history
    df.set_axis(pd.to_timedelta([int(v[1:])-1 for v in df.columns.values], 'h'), axis=1, inplace=True)
    df = df.stack().reset_index()
    df["datetime"] = df["date"] + df["level_2"]
    del df["date"]
    del df["level_2"]
    df = df.set_index(["station_id", "datetime"])
    df.set_axis(["F"], axis=1, inplace=True)
    return df


def load_elec_solution():
    load_history = pd.read_csv(os.path.join(GEFCOM_DATA_PATH,"GEFCOM2012_Data","Load","Load_solution.csv"), sep=',', thousands=',')
    load_history["date"] = pd.to_datetime(load_history["year"].astype(str) + "-" + load_history["month"].astype(str) + "-" + load_history["day"].astype(str), format="%Y-%m-%d")
    del load_history["year"]
    del load_history["month"]
    del load_history["day"]
    del load_history["weight"]
    del load_history["id"]
    df = load_history.set_index(["zone_id","date"])
    del load_history
    df.set_axis(pd.to_timedelta([int(v[1:])-1 for v in df.columns.values], 'h'), axis=1, inplace=True)
    df = df.stack().reset_index()
    df["datetime"] = df["date"] + df["level_2"]
    del df["date"]
    del df["level_2"]
    df = df.set_index(["zone_id", "datetime"])
    df.set_axis(["kWh"], axis=1, inplace=True)
    return df


def load_elec_benchmark():
    load_history = pd.read_csv(os.path.join(GEFCOM_DATA_PATH,"GEFCOM2012_Data","Load","Load_benchmark.csv"), sep=',', thousands=',')
    load_history["date"] = pd.to_datetime(load_history["year"].astype(str) + "-" + load_history["month"].astype(str) + "-" + load_history["day"].astype(str), format="%Y-%m-%d")
    del load_history["year"]
    del load_history["month"]
    del load_history["day"]
    del load_history["id"]
    df = load_history.set_index(["zone_id","date"])
    del load_history
    df.set_axis(pd.to_timedelta([int(v[1:])-1 for v in df.columns.values], 'h'), axis=1, inplace=True)
    df = df.stack().reset_index()
    df["datetime"] = df["date"] + df["level_2"]
    del df["date"]
    del df["level_2"]
    df = df.set_index(["zone_id", "datetime"])
    df.set_axis(["kWh"], axis=1, inplace=True)
    return df


def load_holidays():
    holidays = pd.read_csv(os.path.join(GEFCOM_DATA_PATH,"GEFCOM2012_Data","Load","Holiday_List.csv"), sep=',')
    holidaylist = []
    for year in holidays.iloc[:,1:]:
        for v in holidays[year]:
            if type(v) == float:
                continue
            components = v.split(", ")
            dow = components[0]
            doy = components[1]
            if len(components) == 3:
                y = components[2]
            else:
                y = year
            holidaylist.append([doy + " " + y, dow])
    holidays = pd.DataFrame(holidaylist)
    holidays["datetime"] = pd.to_datetime(holidays[0], format="%B %d %Y")
    holidays["dow"] = holidays.datetime.dt.strftime("%A")
    assert(all(holidays[1] == holidays["dow"]))
    return holidays["datetime"].copy()


# Features for the GEFCom Benchmark Linear Regression described in Hongs PhD Thesis
def get_features(dateindex, T):
    featureset = OrderedDict([
            (("Tr","Trend"), (((dateindex - pd.to_datetime("2004-01-01")) // np.timedelta64(1, 'h')) + 1)),
            #("wrongtrend", np.arange(dateindex.shape[0])),
            (("T","T"), T),
            (("T","T2"), np.power(T,2)),
            (("T","T3"), np.power(T,3)),
            #("hourweekday", (dateindex.hour+1)*(dateindex.dayofweek+1)), # reformulation as class variable
            #("monthT", dateindex.month * T), # reformulation as class variable
            #("monthT2", dateindex.month * np.power(T,2)), # reformulation as class variable
            #("monthT3", dateindex.month * np.power(T,3)), # reformulation as class variable
            #("hourT", (dateindex.hour+1) * T), # reformulation as class variable
            #("hourT2", (dateindex.hour+1) * np.power(T,2)), # reformulation as class variable
            #("hourT3", (dateindex.hour+1) * np.power(T,3)), # reformulation as class variable
        ])
    for i in range(7-1): # We ommit the Saturday (dayofweek == 6) see Hong10, Formula 3.9
        featureset[("W","W"+str(i))] = (dateindex.dayofweek == i).astype("double")
    for i in range(12-1): # We ommit the December (month == 12) see Hong10, Formula 3.9
        i += 1
        featureset[("M","M"+str(i))] = (dateindex.month == i).astype("double")
    for i in range(24-1): # We ommit the 23th hour (hour == 23) see Hong10, Formula 3.9
        featureset[("H","H"+str(i))] = (dateindex.hour == i).astype("double")
    for i in range((24*7)-1):
        dow = i//24
        h = i%24
        featureset[("HW{}".format(dow),"H{}W{}".format(h,dow))] = ((dateindex.dayofweek == dow)&(dateindex.hour == h)).astype("double")
    for i in range(12-1): # We ommit the December (month == 12) see Hong10, Formula 3.9
        i += 1
        featureset[("MT","M{}T".format(i))] = (dateindex.month == i).astype("double") * T
    for i in range(12-1): # We ommit the December (month == 12) see Hong10, Formula 3.9
        i += 1
        featureset[("MT2","M{}T2".format(i))] = (dateindex.month == i).astype("double") * np.power(T,2)
    for i in range(12-1): # We ommit the December (month == 12) see Hong10, Formula 3.9
        i += 1
        featureset[("MT3","M{}T3".format(i))] = (dateindex.month == i).astype("double") * np.power(T,3)
    for i in range(24-1): # We ommit the 23th hour (hour == 23) see Hong10, Formula 3.9
        featureset[("HT","H{}T".format(i))] = (dateindex.hour == i).astype("double") * T
    for i in range(24-1): # We ommit the 23th hour (hour == 23) see Hong10, Formula 3.9
        featureset[("HT2","H{}T2".format(i))] = (dateindex.hour == i).astype("double") * np.power(T,2)
    for i in range(24-1): # We ommit the 23th hour (hour == 23) see Hong10, Formula 3.9
        featureset[("HT3","H{}T3".format(i))] = (dateindex.hour == i).astype("double") * np.power(T,3)
    return pd.DataFrame(featureset, index=dateindex)


# Load all the GEFCom Data

elecdf = load_elec_data()
tempdf = load_temp_data()
holidays = load_holidays()
elecbenchmark = load_elec_benchmark()
elecsolution = load_elec_solution()
elecdf = elecdf.reset_index().pivot(index="datetime", columns="zone_id", values="kWh")
tempdf = tempdf.reset_index().pivot(index="datetime", columns="station_id", values="F")

# Zone 21 / Station 12 is the sum of all Zones / Stations

elecdf[21] = elecdf.sum(axis=1)
tempdf[12] = tempdf.mean(axis=1)

from collections import OrderedDict
from sklearn.linear_model import LinearRegression, Ridge

forecastdateindex = pd.date_range("2008-07-01", "2008-07-07 23:00", freq='H')

backcastdateindexes = [pd.date_range("2005-03-06", "2005-03-12 23:00", freq='H'),
                       pd.date_range("2005-06-20", "2005-06-26 23:00", freq='H'),
                       pd.date_range("2005-09-10", "2005-09-16 23:00", freq='H'),
                       pd.date_range("2005-12-25", "2005-12-31 23:00", freq='H'),
                       pd.date_range("2006-02-13", "2006-02-19 23:00", freq='H'),
                       pd.date_range("2006-05-25", "2006-05-31 23:00", freq='H'),
                       pd.date_range("2006-08-02", "2006-08-08 23:00", freq='H'),
                       pd.date_range("2006-11-22", "2006-11-28 23:00", freq='H')]
def join_indexes(arr):
    result = arr[0]
    for i in range(1,len(arr)):
        result = result.union(arr[i])
    return result
backcastdateindex = join_indexes(backcastdateindexes)

# Temperature forecast
tempforecast = pd.DataFrame(index=forecastdateindex)
for station in tempdf.columns:
    pasttemps = pd.DataFrame(index=forecastdateindex)
    for y in range(1,5):
        idx = forecastdateindex - pd.DateOffset(years=y)
        pasttemps[2008-y] = tempdf.loc[idx,station].values
    pasttemps = pasttemps.mean(axis=1)
    tempforecast[station] = pasttemps

elecsolutiondf = elecsolution.reset_index().pivot(index="datetime", columns="zone_id", values="kWh")
elecbenchmarkdf = elecbenchmark.reset_index().pivot(index="datetime", columns="zone_id", values="kWh")

from sklearn import metrics

def gefcom_score(groundtruth, estimate):
    running_weight = 0.0
    running_score = 0.0
    # forecast
    forecast_gt = groundtruth.reindex(forecastdateindex)
    forecast_es = estimate.reindex(forecastdateindex)
    forecast_size = forecast_gt.shape[0]
    forecast_mse = {}
    for i in range(21):
        forecast_mse[i+1] = metrics.mean_squared_error(forecast_gt[i+1].values, forecast_es[i+1].values)
        if i == 20:
            running_score += 160 * forecast_mse[i+1] * forecast_size
            running_weight += 160 * forecast_size
        else:
            running_score += 8 * forecast_mse[i+1] * forecast_size
            running_weight += 8 * forecast_size
    #pd.Series(forecast_mse)[:20].plot.bar()
    #plt.title("Forecast")
    #plt.show()
    # backcast
    backcast_gt = groundtruth.reindex(backcastdateindex)
    backcast_es = estimate.reindex(backcastdateindex)
    backcast_size = backcast_gt.shape[0]
    backcast_mse = {}
    for i in range(21):
        backcast_mse[i+1] = metrics.mean_squared_error(backcast_gt[i+1].values, backcast_es[i+1].values)
        if i == 20:
            running_score += 20 * backcast_mse[i+1] * backcast_size
            running_weight += 20 * backcast_size
        else:
            running_score += backcast_mse[i+1] * backcast_size
            running_weight += 1 * backcast_size
    #pd.Series(backcast_mse)[:20].plot.bar()
    #plt.title("Backcast")
    #plt.show()
    return np.sqrt(running_score / running_weight)


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_train_XY(zone, station, noise_lambda=None, seed=42):
    np.random.seed(seed+zone)
    tempindex = tempdf.loc[:,station].dropna().index
    dateindex = elecdf.loc[:,zone].dropna().index.intersection(tempindex).sort_values()
    T = tempdf.loc[:,station].reindex(dateindex).values.astype("double")
    features = get_features(dateindex, T)
    target = elecdf[zone].reindex(dateindex).astype("double")
    if noise_lambda > 0.0:
        noise = np.random.laplace(loc=0.0, scale=noise_lambda, size=target.values.shape[0])
        return features, target + noise
    else:
        return features, target


def save_noised_elec_data(zonestationmap, noise_lambda, seed):
    dirname = "lambda{:.2f}_seed{:d}".format(noise_lambda, seed)
    os.makedirs(dirname, exist_ok=True)
    noised_load_history = {}
    index = None
    for i in range(20):
        zone = i+1
        _, target = get_train_XY(zone, zonestationmap[zone], noise_lambda, seed)
        noised_load_history[zone] = target.values
        index = target.index
    noised_load_history = pd.DataFrame(noised_load_history, index=index)
    noised_load_history.columns.set_names(["zone_id"], inplace=True)
    noised_load_history = pd.DataFrame({"load": noised_load_history.unstack()})
    idx = noised_load_history.index.get_level_values(1)
    noised_load_history["hour"] = idx.hour + 1
    noised_load_history = noised_load_history.reset_index()
    noised_load_history["year"] = idx.year
    noised_load_history["month"] = idx.month
    noised_load_history["day"] = idx.day
    noised_load_history = noised_load_history.set_index(["zone_id", "year", "month", "day", "hour"])["load"].unstack()
    noised_load_history.columns = noised_load_history.columns.map(lambda x: "h{}".format(x))    
    noised_load_history.to_csv(os.path.join(dirname, "Load_history.csv"))


def save_noised_elec_benchmark(elecestimate, noise_lambda, seed):
    dirname = "lambda{:.2f}_seed{:d}".format(noise_lambda, seed)
    os.makedirs(dirname, exist_ok=True)
    elecestimate.to_msgpack(os.path.join(dirname, "Load_benchmark.msgpack"))
    elecestimate = elecestimate.copy()
    elecestimate.columns.set_names(["zone_id"], inplace=True)
    elecestimate = pd.DataFrame({"load": elecestimate.unstack()})
    idx = elecestimate.index.get_level_values(1)
    elecestimate["hour"] = idx.hour + 1
    elecestimate = elecestimate.reset_index()
    elecestimate["year"] = idx.year
    elecestimate["month"] = idx.month
    elecestimate["day"] = idx.day
    elecestimate = elecestimate.set_index(["year", "month", "day", "zone_id", "hour"])["load"].unstack()
    elecestimate.columns = elecestimate.columns.map(lambda x: "h{}".format(x))
    elecestimate.columns.set_names(["id"], inplace=True)
    elecestimate = elecestimate.reset_index()
    elecestimate.index = elecestimate.index + 1
    elecestimate = elecestimate[['zone_id', 'year', 'month', 'day', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
           'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16',
           'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24']]
    elecestimate.to_csv(os.path.join(dirname, "Load_benchmark.csv"), index_label="id")


def do_noised_fitting(seed, noise_lambda):
    print(seed, noise_lambda)
    
    # Find best station
    zonestationmap = {}
    zonemodelmap = {}

    for zone in elecdf.columns[:-1]:
        rscores = []
        fitted = {}
        #print("zone {}".format(zone))
        for station in tempdf.columns[:-1]:
            features, target = get_train_XY(zone, station, noise_lambda, seed)
            fitted[station] = LinearRegression(fit_intercept=True, n_jobs=-1).fit(features.values, target.values)
            residual = fitted[station].predict(features.values) - target.values
            rscore = np.sum(np.abs(residual))
            rscores.append([station, rscore])
        rscores = np.array(rscores)
        #print(rscores)
        beststation = int(rscores[np.argmin(rscores[:,1]), 0])
        zonestationmap[zone] = beststation
        zonemodelmap[zone] = fitted[beststation]
        #print("zone {} => station {}".format(zone, beststation))
        #print()

    save_noised_elec_data(zonestationmap, noise_lambda, seed)

    # Fore and Backcast
    elecforecast = pd.DataFrame(index=forecastdateindex)
    for zone in elecdf.columns[:-1]:
        features = get_features(forecastdateindex, tempforecast[zonestationmap[zone]])
        elecforecast[zone] = zonemodelmap[zone].predict(features.values)

    elecforecast[21] = elecforecast.sum(axis=1)

    elecbackcast = pd.DataFrame(index=backcastdateindex)
    for zone in elecdf.columns[:-1]:
        features = get_features(backcastdateindex, tempdf[zonestationmap[zone]].reindex(backcastdateindex))
        elecbackcast[zone] = zonemodelmap[zone].predict(features.values)

    elecbackcast[21] = elecbackcast.sum(axis=1)

    elecestimate = pd.concat([elecbackcast, elecforecast])

    # Persist raw data
    save_noised_elec_benchmark(elecestimate, noise_lambda, seed)

    # Results
    results = {}
    MAPEresults = {}
    MSEresults = {}
    MAEresults = {}
    for i in range(21):
        MAPEresults[i+1] = mean_absolute_percentage_error(elecsolutiondf.reindex(backcastdateindex)[i+1].values,elecestimate.reindex(backcastdateindex)[i+1].values)
        MSEresults[i+1] = metrics.mean_squared_error(elecsolutiondf.reindex(backcastdateindex)[i+1].values,elecestimate.reindex(backcastdateindex)[i+1].values)
        MAEresults[i+1] = metrics.mean_absolute_error(elecsolutiondf.reindex(backcastdateindex)[i+1].values,elecestimate.reindex(backcastdateindex)[i+1].values)
    rdict = results.setdefault("backcast", {})
    rdict["mape"] = MAPEresults
    rdict["mse"] = MSEresults
    rdict["mae"] = MAEresults

    MAPEresults = {}
    MSEresults = {}
    MAEresults = {}
    for i in range(21):
        MAPEresults[i+1] = mean_absolute_percentage_error(elecsolutiondf.reindex(forecastdateindex)[i+1].values,elecestimate.reindex(forecastdateindex)[i+1].values)
        MSEresults[i+1] = metrics.mean_squared_error(elecsolutiondf.reindex(forecastdateindex)[i+1].values,elecestimate.reindex(forecastdateindex)[i+1].values)
        MAEresults[i+1] = metrics.mean_absolute_error(elecsolutiondf.reindex(forecastdateindex)[i+1].values,elecestimate.reindex(forecastdateindex)[i+1].values)
    rdict = results.setdefault("forecast", {})
    rdict["mape"] = MAPEresults
    rdict["mse"] = MSEresults
    rdict["mae"] = MAEresults
    
    results["gefcom_score"] = gefcom_score(elecsolutiondf, elecestimate)
    
    np.save("mlr_noise_{:.2f}_seed{:d}".format(noise_lambda, seed), results)

    return elecestimate, elecbackcast, elecforecast


if __name__ == "__main__":
    import argparse
    import concurrent.futures

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='23, 42, ...')
    parser.add_argument('lambdas', metavar='L', type=float, nargs='+',
        help='lambda values')

    opt = parser.parse_args()
    print(opt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        seed = opt.seed
        for noise_lambda in opt.lambdas:
            print(seed, noise_lambda)
            executor.submit(do_noised_fitting, seed, noise_lambda)