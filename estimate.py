import pandas
import matplotlib.pyplot as plt
import numpy
import scipy

def get_data():
    url_confirmed = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    raw_df_confirmed = pandas.read_csv(url_confirmed)
    df_c = raw_df_confirmed.copy()
    del df_c['Long']
    del df_c['Lat']
    df_c = df_c.set_index(['Country/Region','Province/State'])\
               .transpose() # transpose to get dates as rows
    # make a proper datetime index
    df_c = df_c.set_index(pandas.to_datetime(df_c.index))
    # sum up on a per country level
    return pandas.DataFrame(
        {c: df_c[c].sum(axis=1) for c in df_c.columns.get_level_values(0)})

def estimate_R(df, country, rolling=7, W=3, Tc=5.2, min_cases=30):
    confirmed = df[country]
    confirmed = confirmed[confirmed.index[(confirmed > min_cases)][0]:]
    deltas = (confirmed - confirmed.shift(1)).dropna()
    k = deltas.rolling(rolling, win_type='gaussian').mean(std=3).dropna().round()
    tau = 1
    r_range = numpy.linspace(0, 10, 500)
    gamma = 1/Tc
    lambdas = numpy.outer(k[:-1],  numpy.exp(tau*gamma*(r_range - 1)))
    L = scipy.stats.poisson.pmf(k[1:], lambdas.T)
    P  = L.copy()
    for i in range(L.shape[1]):
        P[:,i] = L[:,i] / L[:,i].sum()
        for j in range(1, min(W, i)):
            P[:,i] *= L[:,i-j]
            P[:,i] /= P[:,i].sum()
    lower = [r_range[numpy.argwhere(P[:,i].cumsum() >= 0.025)[0]][0] for i in range(len(k) - 1)]
    middle = [r_range[numpy.argwhere(P[:,i].cumsum() >= 0.5)[0]][0] for i in range(len(k) - 1)]
    upper = [r_range[numpy.argwhere(P[:,i].cumsum() >= 0.975)[0]][0] for i in range(len(k) - 1)]
    return pandas.DataFrame({'median': middle,
                             'lower': lower,
                             'upper': upper,
                             'cases': k[1:]}, index=k.index[1:])

def plot_country(df, country, **kwargs):
    plt.figure()
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 9
    plt.rcParams['lines.linewidth'] = 1.5
    estimates = estimate_R(df, country, **kwargs)
    estimates['median'].plot.line(marker='o', lw=0)
    plt.ylabel('$R_t$')
    plt.fill_between(estimates.index, estimates['lower'], estimates['upper'], alpha=0.6, color='grey')
    estimates['cases'].plot.line(secondary_y=True)
    plt.ylabel('new cases')
    plt.title(country)
    plt.savefig(f"results/{country}.png", dpi=300, bbox_inches='tight')
    estimates.to_csv(f"results/{country}.csv")

if __name__ == "__main__":
    import progressbar
    countries = ['Spain', 'Germany', 'Norway', 'Denmark', 'Sweden', 'Italy', 'Korea, South', 'United Kingdom']
    data = get_data()
    for c in progressbar.progressbar(countries):
        plot_country(data, c)

