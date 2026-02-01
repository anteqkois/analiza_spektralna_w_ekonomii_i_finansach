#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJEKT ZALICZENIOWY: ANALIZA SPEKTRALNA
Dane: Bitcoin BTC/USD (2010-2025)
"""

# %% [markdown]
# # PROJEKT ZALICZENIOWY: ANALIZA SPEKTRALNA
#
# **Dane:** Bitcoin BTC/USD (btcusd_day.csv, 2010-2025, ~5600 obs.)
# **Transformacja:** Logarytmiczne stopy zwrotu (dla stacjonarności)

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.hp_filter import hpfilter
from astropy.timeseries import LombScargle
import warnings

warnings.filterwarnings("ignore")

print("✓ Biblioteki załadowane")

# %% [markdown]
# ## PRZYGOTOWANIE DANYCH

# %%
# Wczytanie danych
df = pd.read_csv(
    "btcusd_day.csv", usecols=["datetime", "close"], parse_dates=["datetime"]
)
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)
df = df.dropna()

print(f"Dane: {df.index[0].date()} - {df.index[-1].date()}, N={len(df)}")

# %%
# Wizualizacja cen
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Cena BTC/USD"))
fig.update_layout(
    title="Cena zamknięcia Bitcoin",
    xaxis_title="Data",
    yaxis_title="Cena (USD)",
    height=400,
)
fig.show()


# %%
# Test stacjonarności
def test_adf(series, nazwa):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"\n{nazwa}:")
    print(f"  ADF statystyka: {result[0]:.4f}")
    print(f"  p-wartość: {result[1]:.4f}")
    print(f"  Wniosek: {'STACJONARNY' if result[1] < 0.05 else 'NIESTACJONARNY'}")
    return result[1] < 0.05


# Test dla cen
is_stat_price = test_adf(df["close"], "Ceny oryginalne")

# Logarytmiczne zwroty
df["log_returns"] = np.log(df["close"]).diff()
is_stat_returns = test_adf(df["log_returns"].dropna(), "Log-zwroty")

# Wybór danych stacjonarnych
if is_stat_returns:
    data_stacjonarne = df["log_returns"].dropna()
    metoda = "Logarytmiczne zwroty"
else:
    # Filtr HP jako backup
    hp_cycle, _ = hpfilter(np.log(df["close"]).dropna(), lamb=6_400_000)
    data_stacjonarne = hp_cycle
    metoda = "Filtr HP"

print(f"\nMetoda stacjonaryzacji: {metoda}")
print(f"Liczba obserwacji: {len(data_stacjonarne)}")

# %%
# Wizualizacja danych stacjonarnych
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=data_stacjonarne.index,
        y=data_stacjonarne.values,
        mode="lines",
        name=metoda,
        line=dict(width=0.8),
    )
)
fig.add_hline(y=0, line_dash="dash", line_color="red")
fig.update_layout(
    title=f"Dane stacjonarne: {metoda}",
    xaxis_title="Data",
    yaxis_title="Wartość",
    height=400,
)
fig.show()

# %% [markdown]
# ---
# # ZADANIE 1: ANALIZA PODSTAWOWA
# ---

# %% [markdown]
# ## 1.1 DFT i periodogram: oryginalne vs scentrowane


# %%
# Funkcje pomocnicze
def compute_dft(x):
    return fft(x)


def naive_periodogram(x):
    N = len(x)
    X = fft(x)
    power = (1 / N) * np.abs(X) ** 2
    freqs = fftfreq(N, d=1.0)
    idx = freqs > 0
    return freqs[idx], power[idx]


# %%
# Dane
x_original = data_stacjonarne.values
x_centered = x_original - np.mean(x_original)

print(f"Średnia oryginalne: {np.mean(x_original):.6f}")
print(f"Średnia scentrowane: {np.mean(x_centered):.6f}")

# DFT
dft_orig = compute_dft(x_original)
dft_cent = compute_dft(x_centered)

# Periodogramy
freqs_orig, power_orig = naive_periodogram(x_original)
freqs_cent, power_cent = naive_periodogram(x_centered)

# %%
# Wizualizacja DFT
fig = make_subplots(
    rows=1, cols=2, subplot_titles=("DFT - Dane oryginalne", "DFT - Dane scentrowane")
)

fig.add_trace(
    go.Scatter(
        y=np.abs(dft_orig[: len(dft_orig) // 2]), mode="lines", name="Oryginalne"
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        y=np.abs(dft_cent[: len(dft_cent) // 2]),
        mode="lines",
        name="Scentrowane",
        line=dict(color="red"),
    ),
    row=1,
    col=2,
)

fig.update_yaxes(type="log", title_text="Amplituda", row=1, col=1)
fig.update_yaxes(type="log", title_text="Amplituda", row=1, col=2)
fig.update_xaxes(title_text="Indeks częstotliwości", row=1, col=1)
fig.update_xaxes(title_text="Indeks częstotliwości", row=1, col=2)
fig.update_layout(height=400, showlegend=False)
fig.show()

# %%
# Wizualizacja periodogramów
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Periodogram - Oryginalne", "Periodogram - Scentrowane"),
)

fig.add_trace(
    go.Scatter(x=freqs_orig, y=power_orig, mode="lines", name="Oryginalne"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=freqs_cent,
        y=power_cent,
        mode="lines",
        name="Scentrowane",
        line=dict(color="red"),
    ),
    row=1,
    col=2,
)

fig.update_yaxes(type="log", title_text="Moc", row=1, col=1)
fig.update_yaxes(type="log", title_text="Moc", row=1, col=2)
fig.update_xaxes(title_text="Częstotliwość", row=1, col=1)
fig.update_xaxes(title_text="Częstotliwość", row=1, col=2)
fig.update_layout(height=400, showlegend=False)
fig.show()

print(
    "\nKOMENTARZ: Centrowanie usuwa składową DC (f=0), ale nie wpływa na widmo dla f>0"
)

# %% [markdown]
# ## 1.2 Eksperyment z długością N

# %%
natural_period = 7  # tydzień
N_full = len(x_centered)
N_div = (N_full // natural_period) * natural_period
N_nondiv = N_div + 1

x_div = x_centered[:N_div]
x_nondiv = x_centered[:N_nondiv]

freqs_div, power_div = naive_periodogram(x_div)
freqs_nondiv, power_nondiv = naive_periodogram(x_nondiv)

print(f"N podzielne przez {natural_period}: {N_div}")
print(f"N niepodzielne: {N_nondiv}")

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(f"N={N_div} (podzielne)", f"N={N_nondiv} (niepodzielne)"),
)

fig.add_trace(
    go.Scatter(x=freqs_div, y=power_div, mode="lines", name="Podzielne"), row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=freqs_nondiv,
        y=power_nondiv,
        mode="lines",
        name="Niepodzielne",
        line=dict(color="orange"),
    ),
    row=2,
    col=1,
)

freq_nat = 1 / natural_period
fig.add_vline(x=freq_nat, line_dash="dash", line_color="red", row=1, col=1)
fig.add_vline(x=freq_nat, line_dash="dash", line_color="red", row=2, col=1)

fig.update_yaxes(type="log", title_text="Moc", row=1, col=1)
fig.update_yaxes(type="log", title_text="Moc", row=2, col=1)
fig.update_xaxes(title_text="Częstotliwość", range=[0, 0.5], row=2, col=1)
fig.update_layout(height=600, showlegend=False)
fig.show()

print(
    f"\nKOMENTARZ: N podzielne minimalizuje przeciek spektralny dla okresu {natural_period} dni"
)

# %% [markdown]
# ---
# # ZADANIE 2: PORÓWNANIE METOD
# ---

# %% [markdown]
# ## 2.1 Periodogram z 95% przedziałem ufności


# %%
def periodogram_with_ci(x, alpha=0.05):
    freqs, power = naive_periodogram(x)
    chi2_lower = stats.chi2.ppf(alpha / 2, 2)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, 2)
    ci_lower = power * (2 / chi2_upper)
    ci_upper = power * (2 / chi2_lower)
    return freqs, power, ci_lower, ci_upper


freqs_ci, power_ci, ci_lower, ci_upper = periodogram_with_ci(x_centered)

# %%
fig = go.Figure()
# Najpierw fill (pod spodem)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=ci_upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=ci_lower,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(173, 216, 230, 1.0)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
# Potem główna seria (na wierzchu)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_ci,
        mode="lines",
        name="Periodogram",
        line=dict(color="blue", width=1.5),
    )
)
fig.update_layout(
    title="Periodogram naiwny z 95% przedziałem ufności",
    xaxis_title="Częstotliwość",
    yaxis_title="Moc",
    yaxis_type="log",
    height=400,
)
fig.show()

# %% [markdown]
# ## 2.2 Periodogram Welcha

# %%
# Wariant 1: nperseg=1024, overlap=25%
nperseg1 = min(1024, len(x_centered) // 2)
freqs_welch1024, power_welch1024 = signal.welch(
    x_centered, fs=1.0, nperseg=nperseg1, noverlap=nperseg1 // 4
)

# Wariant 2: nperseg=512, overlap=50%
nperseg2 = min(512, len(x_centered) // 4)
freqs_welch512, power_welch512 = signal.welch(
    x_centered, fs=1.0, nperseg=nperseg2, noverlap=nperseg2 // 2
)

# Wariant 3: nperseg=256, overlap=75%
nperseg3 = min(256, len(x_centered) // 8)
freqs_welch256, power_welch256 = signal.welch(
    x_centered, fs=1.0, nperseg=nperseg3, noverlap=int(nperseg3 * 0.75)
)

# Wariant 4: nperseg=124, overlap=50%
nperseg4 = 124
freqs_welch124, power_welch124 = signal.welch(
    x_centered, fs=1.0, nperseg=nperseg4, noverlap=nperseg4 // 2
)

print(f"Welch wariant 1: nperseg={nperseg1}, overlap=25%")
print(f"Welch wariant 2: nperseg={nperseg2}, overlap=50%")
print(f"Welch wariant 3: nperseg={nperseg3}, overlap=75%")
print(f"Welch wariant 4: nperseg={nperseg4}, overlap=50%")

# Używamy wariantu 512 jako głównego (dla kompatybilności)
freqs_welch = freqs_welch512
power_welch = power_welch512

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        f"Welch: seg={nperseg1}, 25%",
        f"Welch: seg={nperseg2}, 50%",
        f"Welch: seg={nperseg3}, 75%",
        f"Welch: seg={nperseg4}, 50%",
    ),
)

fig.add_trace(
    go.Scatter(
        x=freqs_welch1024,
        y=power_welch1024,
        mode="lines",
        name="W1 (1024)",
        line=dict(color="#1f77b4"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=freqs_welch512,
        y=power_welch512,
        mode="lines",
        name="W2 (512)",
        line=dict(color="#d62728"),
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=freqs_welch256,
        y=power_welch256,
        mode="lines",
        name="W3 (256)",
        line=dict(color="#2ca02c"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=freqs_welch124,
        y=power_welch124,
        mode="lines",
        name="W4 (124)",
        line=dict(color="#ff7f0e"),
    ),
    row=2,
    col=2,
)

fig.update_yaxes(type="log", title_text="Moc", row=1, col=1)
fig.update_yaxes(type="log", title_text="Moc", row=1, col=2)
fig.update_yaxes(type="log", title_text="Moc", row=2, col=1)
fig.update_yaxes(type="log", title_text="Moc", row=2, col=2)
fig.update_xaxes(title_text="Częstotliwość", row=2, col=1)
fig.update_xaxes(title_text="Częstotliwość", row=2, col=2)
fig.update_layout(height=600, showlegend=False)
fig.show()

# %%
# Porównanie wszystkich wariantów Welcha
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=freqs_welch1024,
        y=power_welch1024,
        mode="lines",
        name="Welch 1024, 25%",
        line=dict(color="#1f77b4", width=0.7),
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_welch512,
        y=power_welch512,
        mode="lines",
        name="Welch 512, 50%",
        line=dict(color="#d62728", width=0.7),
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_welch256,
        y=power_welch256,
        mode="lines",
        name="Welch 256, 75%",
        line=dict(color="#2ca02c", width=0.7),
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_welch124,
        y=power_welch124,
        mode="lines",
        name="Welch 124, 50%",
        line=dict(color="#ff7f0e", width=0.7),
    )
)

fig.update_layout(
    title="Porównanie wariantów Welcha (wysoki kontrast)",
    xaxis_title="Częstotliwość",
    yaxis_title="Moc",
    yaxis_type="log",
    height=400,
)
fig.show()

# %% [markdown]
# ## 2.3 Wygładzanie oknem Daniella


# %%
def daniell_smooth(periodogram, m, levels=1):
    smoothed = periodogram.copy()
    for _ in range(levels):
        window = np.ones(2 * m + 1) / (2 * m + 1)
        smoothed = np.convolve(smoothed, window, mode="same")
    return smoothed


# m=5
m1 = 5
power_daniell_5_1 = daniell_smooth(power_ci, m=m1, levels=1)
power_daniell_5_3 = daniell_smooth(power_ci, m=m1, levels=3)

# m=10
m2 = 10
power_daniell_10_1 = daniell_smooth(power_ci, m=m2, levels=1)
power_daniell_10_3 = daniell_smooth(power_ci, m=m2, levels=3)

print(f"Okno Daniella: m={m1}, szerokość={2*m1+1}")
print(f"Okno Daniella: m={m2}, szerokość={2*m2+1}")

# Dla kompatybilności z resztą kodu
power_daniell_1 = power_daniell_5_1
power_daniell_3 = power_daniell_5_3

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        f"Daniell m={m1}, 1-poz",
        f"Daniell m={m1}, 3-poz",
        f"Daniell m={m2}, 1-poz",
        f"Daniell m={m2}, 3-poz",
    ),
)

fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_5_1,
        mode="lines",
        name="D5-1",
        line=dict(color="#1f77b4"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_5_3,
        mode="lines",
        name="D5-3",
        line=dict(color="#d62728"),
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_10_1,
        mode="lines",
        name="D10-1",
        line=dict(color="#2ca02c"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_10_3,
        mode="lines",
        name="D10-3",
        line=dict(color="#ff7f0e"),
    ),
    row=2,
    col=2,
)

fig.update_yaxes(type="log", title_text="Moc", row=1, col=1)
fig.update_yaxes(type="log", title_text="Moc", row=1, col=2)
fig.update_yaxes(type="log", title_text="Moc", row=2, col=1)
fig.update_yaxes(type="log", title_text="Moc", row=2, col=2)
fig.update_xaxes(title_text="Częstotliwość", row=2, col=1)
fig.update_xaxes(title_text="Częstotliwość", row=2, col=2)
fig.update_layout(height=600, showlegend=False)
fig.show()

# %%
# Porównanie wszystkich wariantów Daniella
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_5_1,
        mode="lines",
        name=f"Daniell m={m1}, 1-poz",
        line=dict(color="#1f77b4", width=0.7),
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_5_3,
        mode="lines",
        name=f"Daniell m={m1}, 3-poz",
        line=dict(color="#d62728", width=0.7),
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_10_1,
        mode="lines",
        name=f"Daniell m={m2}, 1-poz",
        line=dict(color="#2ca02c", width=0.7),
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_10_3,
        mode="lines",
        name=f"Daniell m={m2}, 3-poz",
        line=dict(color="#ff7f0e", width=0.7),
    )
)

fig.update_layout(
    title="Porównanie wariantów Daniella (wysoki kontrast)",
    xaxis_title="Częstotliwość",
    yaxis_title="Moc",
    yaxis_type="log",
    height=400,
)
fig.show()

# %% [markdown]
# ## 2.4 Wygładzanie manualne (uśrednianie podokresów)


# %%
def manual_average(x, n_seg=5):
    N = len(x)
    seg_len = N // n_seg
    x_trim = x[: seg_len * n_seg]
    segments = np.array_split(x_trim, n_seg)
    periodograms = [naive_periodogram(seg)[1] for seg in segments]
    freqs_avg, _ = naive_periodogram(segments[0])
    return freqs_avg, np.mean(periodograms, axis=0)


freqs_manual, power_manual = manual_average(x_centered, n_seg=5)

print("Wygładzanie manualne: 5 podokresów")

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=freqs_manual,
        y=power_manual,
        mode="lines",
        name="Manualny",
        line=dict(color="purple"),
    )
)
fig.update_layout(
    title="Periodogram - wygładzanie manualne (5 podokresów)",
    xaxis_title="Częstotliwość",
    yaxis_title="Moc",
    yaxis_type="log",
    height=400,
)
fig.show()

# %% [markdown]
# ## 2.5 Porównanie wszystkich metod

# %%
fig = go.Figure()
# 1. Naiwny
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_ci,
        mode="lines",
        name="Naiwny",
        line=dict(color="gray", width=0.7),
        opacity=0.4,
    )
)
# 2. Welch 512
fig.add_trace(
    go.Scatter(
        x=freqs_welch512,
        y=power_welch512,
        mode="lines",
        name="Welch 512",
        line=dict(color="#d62728", width=0.8),
    )
)
# 3. Welch 256
fig.add_trace(
    go.Scatter(
        x=freqs_welch256,
        y=power_welch256,
        mode="lines",
        name="Welch 256",
        line=dict(color="#2ca02c", width=0.8),
    )
)
# 4. Daniell 10 1-poz
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_10_1,
        mode="lines",
        name="Daniell 10 (1-poz)",
        line=dict(color="#1f77b4", width=0.8),
    )
)
# 5. Daniell 10 3-poz
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_10_3,
        mode="lines",
        name="Daniell 10 (3-poz)",
        line=dict(color="maroon", width=0.8),
    )
)
# 6. Daniell 5 3-poz
fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_daniell_5_3,
        mode="lines",
        name="Daniell 5 (3-poz)",
        line=dict(color="red", width=0.8),
    )
)
# 7. Manualny
fig.add_trace(
    go.Scatter(
        x=freqs_manual,
        y=power_manual,
        mode="lines",
        name="Manualny",
        line=dict(color="purple", width=0.8),
    )
)

fig.update_layout(
    title="Porównanie wybranych metod estymacji (Final)",
    xaxis_title="Częstotliwość",
    yaxis_title="Moc",
    yaxis_type="log",
    height=600,
)
fig.show()

print("\nKOMENTARZ: Welch - najlepszy kompromis wariancja/rozdzielczość")
print("Trade-off: wariancja ↓ ⟺ rozdzielczość ↓")

# %% [markdown]
# ---
# # ZADANIE 3: DANE NIEKOMPLETNE
# ---

# %% [markdown]
# ## 3.1 Symulacja braków (30%)

# %%
np.random.seed(42)
df_complete = pd.DataFrame(
    {"value": x_centered}, index=data_stacjonarne.index[: len(x_centered)]
)
n_remove = int(0.3 * len(df_complete))
indices_remove = np.random.choice(df_complete.index, size=n_remove, replace=False)
df_incomplete = df_complete.drop(indices_remove)

print(f"Usunięto {n_remove} obs. ({100*n_remove/len(df_complete):.1f}%)")
print(f"Pozostało: {len(df_incomplete)} obs.")

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        "Dane kompletne",
        "Dane: pozostałe (niebieskie) i usunięte (czerwone)",
    ),
)

fig.add_trace(
    go.Scatter(
        x=df_complete.index,
        y=df_complete["value"],
        mode="lines",
        name="Kompletne",
        line=dict(color="blue", width=0.8),
    ),
    row=1,
    col=1,
)

# Dane pozostałe (70%)
fig.add_trace(
    go.Scatter(
        x=df_incomplete.index,
        y=df_incomplete["value"],
        mode="markers",
        name="Pozostałe (70%)",
        marker=dict(size=3, color="blue"),
    ),
    row=2,
    col=1,
)

# Dane usunięte (30%)
df_removed = df_complete.loc[indices_remove]
fig.add_trace(
    go.Scatter(
        x=df_removed.index,
        y=df_removed["value"],
        mode="markers",
        name="Usunięte (30%)",
        marker=dict(size=3, color="red", symbol="x"),
    ),
    row=2,
    col=1,
)

fig.update_yaxes(title_text="Wartość", row=1, col=1)
fig.update_yaxes(title_text="Wartość", row=2, col=1)
fig.update_xaxes(title_text="Czas", row=2, col=1)
fig.update_layout(height=600, showlegend=True)
fig.show()

# %% [markdown]
# ## 3.2 Periodogram naiwny (z interpolacją)

# %%
df_interp = df_complete.copy()
df_interp.loc[indices_remove, "value"] = np.nan
df_interp["value"] = df_interp["value"].interpolate(method="linear")

freqs_interp, power_interp = naive_periodogram(df_interp["value"].values)

print("Periodogram naiwny (interpolacja liniowa)")

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=freqs_interp,
        y=power_interp,
        mode="lines",
        name="Naiwny (interpolowany)",
        line=dict(color="orange"),
    )
)
fig.update_layout(
    title="Periodogram naiwny - dane interpolowane",
    xaxis_title="Częstotliwość",
    yaxis_title="Moc",
    yaxis_type="log",
    height=400,
)
fig.show()

# %% [markdown]
# ## 3.3 Periodogram Lomba-Scargle'a

# %%
t_incomplete = (df_incomplete.index - df_incomplete.index[0]).total_seconds() / 86400
y_incomplete = df_incomplete["value"].values

ls = LombScargle(t_incomplete, y_incomplete)
freqs_ls = np.linspace(0.001, 0.5, 5000)
power_ls = ls.power(freqs_ls)

fap_level = 0.05
power_threshold = ls.false_alarm_level(fap_level)

print(f"Lomb-Scargle: próg istotności (FAP=5%) = {power_threshold:.6f}")

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=freqs_ls,
        y=power_ls,
        mode="lines",
        name="Lomb-Scargle",
        line=dict(color="green"),
    )
)
fig.add_hline(
    y=power_threshold, line_dash="dash", line_color="red", annotation_text=f"Próg 5%"
)

fig.update_layout(
    title="Periodogram Lomba-Scargle'a - dane niekompletne",
    xaxis_title="Częstotliwość",
    yaxis_title="Moc znormalizowana",
    height=400,
)
fig.show()

# %% [markdown]
# ## 3.4 Porównanie

# %%
fig = go.Figure()

# Normalizacja
power_ci_norm = power_ci / power_ci.max()
power_interp_norm = power_interp / power_interp.max()
power_ls_norm = power_ls / power_ls.max()

fig.add_trace(
    go.Scatter(
        x=freqs_ci,
        y=power_ci_norm,
        mode="lines",
        name="Naiwny (kompletne)",
        line=dict(color="blue", width=0.7),
        opacity=0.7,
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_interp,
        y=power_interp_norm,
        mode="lines",
        name="Naiwny (interpolowane)",
        line=dict(color="orange", width=0.7),
        opacity=0.7,
    )
)
fig.add_trace(
    go.Scatter(
        x=freqs_ls,
        y=power_ls_norm,
        mode="lines",
        name="Lomb-Scargle",
        line=dict(color="green", width=0.7),
        opacity=0.9,
    )
)
fig.add_hline(
    y=power_threshold / power_ls.max(),
    line_dash="dash",
    line_color="red",
    annotation_text="Próg 5%",
)

fig.update_layout(
    title="Porównanie metod dla danych niekompletnych",
    xaxis_title="Częstotliwość",
    yaxis_title="Moc znormalizowana",
    xaxis_range=[0, 0.5],
    height=500,
)
fig.show()

print("\nKOMENTARZ: Lomb-Scargle >> Naiwny dla danych z brakami")
print("Interpolacja tworzy artefakty - nie zalecana!")

# %% [markdown]
# ---
# # PODSUMOWANIE
# ---

# %%
print("\n" + "=" * 70)
print("PODSUMOWANIE PROJEKTU")
print("=" * 70)
print(
    f"""
ZADANIE 1:
✓ DFT/periodogram: centrowanie usuwa DC, nie wpływa na f>0
✓ Eksperyment N: podzielne minimalizuje przeciek spektralny

ZADANIE 2:
✓ Periodogram + 95% CI (rozkład χ²)
✓ Metoda Welcha - NAJLEPSZA (kompromis wariancja/rozdzielczość)
✓ Daniell (1 i 3 poziomy) - wygładzanie
✓ Manualny (5 podokresów) - uśrednianie

ZADANIE 3:
✓ Braki 30% - symulacja
✓ Naiwny (interpolacja) - problematyczny
✓ Lomb-Scargle - REKOMENDOWANY dla braków
✓ Test istotności 5% (FAP)

WNIOSKI:
- Stacjonarność kluczowa
- Welch dla danych regularnych
- Lomb-Scargle dla braków
- Trade-off: wariancja ↓ ⟺ rozdzielczość ↓
"""
)
print("=" * 70)
print("\n✓ PROJEKT ZAKOŃCZONY")
