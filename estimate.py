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
    if numpy.sum(df[country] > min_cases) == 0:
        return None
    # Confirmed cases per day for country
    confirmed = df[country]
    # Keep only days with confirmed cases above min_cases
    confirmed = confirmed[confirmed.index[(confirmed > min_cases)][0]:]
    # Change in confirmed cases since previous day
    deltas = (confirmed - confirmed.shift(1)).dropna()
    # Moving/rolling average over deltas with gaussian smoothing
    # (window size specified by rolling, default 7 days)
    k = deltas.rolling(rolling, win_type='gaussian').mean(std=3).dropna().round()
    tau = 1 # 
    r_range = numpy.linspace(0, 10, 500) # Sample 500 R numbers from [0,10] (R = virus spread rate)
    gamma = 1/Tc #

    # Outer product of k and e^(tau*gamma*(r_range-1))
    # Gives a m*n matrix where m=days and n=(R number samples: 500)
    # Effectively a matrix of 500 different expected delta case occurences
    # These 500 different expected occurence numbers per day are the parameters
    # for a Poisson distribution
    lambdas = numpy.outer(k[:-1],  numpy.exp(tau*gamma*(r_range - 1)))

    # Calculate the likelihoods of the moving averages from above
    # using a Poisson distribution for all sampled expected delta case
    # occurrences
    L = scipy.stats.poisson.pmf(k[1:], lambdas.T)

    P = L.copy()
    for i in range(L.shape[1]): # For each day
        # Prior probabilities of different moving averages
        # for different expected delta case occurrences
        # under Poisson distribution
        P[:,i] = L[:,i] / (L[:,i].sum() + 1e-30)
        # Calculate conditional probabilities of
        # different moving averages given
        # previously observed moving averages
        for j in range(1, min(W, i)): # At least W (default 3)
            P[:,i] *= L[:,i-j]
            P[:,i] /= (P[:,i].sum() + 1e-30)

    # Local function (hacky)
    def R_lookup(i, threshold):
        try:
            return r_range[numpy.argwhere(P[:,i].cumsum() >= threshold)[0]][0]
        except:
            return None

    lower = [R_lookup(i, 0.025) for i in range(len(k) - 1)]
    middle = [R_lookup(i, 0.5) for i in range(len(k) - 1)]
    upper = [R_lookup(i, 0.975) for i in range(len(k) - 1)]

    return pandas.DataFrame({'median': middle,
                             'lower': lower,
                             'upper': upper,
                             'cases': k[1:]}, index=k.index[1:])

def plot_country(df, country, **kwargs):
    plt.figure()
    estimates = estimate_R(df, country, **kwargs)

    # Not enough data to estimate
    if estimates is None:
        return
    
    estimates['one'] = 1
    estimates['median'].plot.line(marker='o', lw=0)
    estimates['one'].plot.line(color='grey', alpha=0.8)
    plt.ylabel('$R_t$')
    plt.fill_between(estimates.index, estimates['lower'], estimates['upper'], alpha=0.6, color='grey')

    # Add horizontal dashed helper lines for R
    for y_tick in plt.yticks()[0]:
        if y_tick == 1:
            continue
        ix = "hl"+str(y_tick)
        estimates[ix] = y_tick
        estimates[ix].plot.line(color='lightgray', alpha=.7, style='--')

    #plt.line(estimates.index, [1]*estimates.shape(1))
    estimates['cases'].plot.line(secondary_y=True)
    plt.ylabel('new cases')
    plt.title(country)
    plt.savefig(f"results/{country}.png", dpi=300, bbox_inches='tight')
    plt.close()
    estimates.to_csv(f"results/{country}.csv")

if __name__ == "__main__":
    import progressbar
    countries = ['Brazil', 'Spain', 'Germany', 'Norway', 'Denmark', 'Sweden', 'Italy', 'Korea, South',
                 'United Kingdom', 'US', 'Canada', 'Singapore', 'Thailand', 'China', 'Japan',
                 'Russia', 'Azerbaijan', 'Angola', 'Nigeria', 'Algeria', 'Venezuela', 'Libya',
                 'Tanzania', 'Argentina', 'Australia', 'India', 'Turkey', 'United Arab Emirates',
                 'Mexico', 'Netherlands', 'Nicaragua', 'Belgium', 'Ireland', 'Bahamas', 'Poland']
    data = get_data()
    for c in countries:
        assert c in data.columns.values
    for c in progressbar.progressbar(countries):
        try:
            plot_country(data, c)
        except:
            print('Something went wrong. Considering debugging')
            pass