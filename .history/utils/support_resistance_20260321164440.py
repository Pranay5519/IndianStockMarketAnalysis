from scipy.signal import find_peaks

def get_support_resistance(df):

    df = df.sort_values("Date")

    peaks, _ = find_peaks(df["Close"], distance=20)
    lows, _ = find_peaks(-df["Close"], distance=20)

    resistance = df.iloc[peaks]["Close"].sort_values(ascending=False).head(2)
    support = df.iloc[lows]["Close"].sort_values().head(2)

    return support.values, resistance.values