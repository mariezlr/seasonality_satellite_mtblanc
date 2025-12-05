from utils import *
from seasonality_satellite_mtblanc.src.data_exploration import *
from main import *
import numpy as np
import matplotlib.pyplot as plt



def plot_random_pixel_ts():
    fig = plt.figure()

    # Trouver les indices des pixels valides
    notnan_pixels = vel_result.notnull().sum(dim='mid_date')
    valid_indices = np.argwhere(notnan_pixels.values >= 1)

    # Sélectionner le premier pixel valide (ou un aléatoire)
    y_idx, x_idx = valid_indices[0]

    # Extraire la série temporelle
    ts_orig_local = velocity.isel(y=y_idx, x=x_idx)
    ts_orig = vel_avg_detrended.isel(y=y_idx, x=x_idx)
    ts_filter = vel_lowpass_filter.isel(y=y_idx, x=x_idx)

    plt.figure(figsize=(10, 4))
    ts_orig_local.plot(label="Original", color="skyblue")
    ts_orig.plot(label="Spatial Avg", color="blue")
    ts_filter.plot(label="Filtered", color="red")

    plt.title(f'Série temporelle pour le pixel (y={y_idx}, x={x_idx})')
    plt.xlabel('Date')
    plt.ylabel('Vitesse')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    fig.savefig(f"../figures/validation_{name}.png")
    plt.close(fig)


def plot_validation_pixel_ts(x0, y0, df_data, name):

    # Extraire la série temporelle originale et filtrée à ce point avant filtrage
    ts_orig_local = velocity.sel(x=x0, y=y0, method='nearest')
    ts_orig = vel_avg_detrended.sel(x=x0, y=y0, method='nearest')
    ts_filter = vel_lowpass_filter.sel(x=x0, y=y0, method='nearest')

    # Filter timeserie between 2016 and 2022
    t_min = ts_orig_local['mid_date'].min().values
    t_max = ts_orig_local['mid_date'].max().values

    temporal_mask = (df_data['date'] >= t_min) & (df_data['date'] <= t_max)
    data_sub = df_data[temporal_mask]

    # Plot
    fig = plt.figure(figsize=(10, 4))
    ts_orig_local.plot(label="Original", color="skyblue")
    ts_orig.plot(label="Spatial Avg", color="blue")
    ts_filter.plot(label="Filtered", color="red")
    plt.plot(data_sub['date'], data_sub['velocity'], color='green', label="obs sliding velocity")


    plt.title(f'Velocity timeseries for the nearest pixel from {name} (x={x0:.2f}, y={y0:.2f})')
    plt.xlabel('Date')
    plt.ylabel(r'Velocity (m yr$^{-1}$)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    fig.savefig(f"../figures/validation_{name}.png")
    plt.close(fig)


def plot_annual_cycle_validation_point(df, name):
    df['day_of_year'] = df['date'].dt.dayofyear

    doy_mean = df.groupby('day_of_year')['velocity'].mean()

    fig, ax = plt.subplots(figsize=(11,5))
    ax.plot(doy_mean.index, doy_mean.values)
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Velocity (m/yr)")
    ax.set_title("Mean seasonal cycle of glacier sliding velocity")
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(fig_dir / f"annual_cycle_{name}.pdf")
    plt.close(fig)


def plot_xcount_ts_validation_points():
    fig, ax = plt.subplots()

    ts_xcount_arg4 = ds_merged['xcount'].sel(x=x_utm_Arg4_GPS, y=y_utm_Arg4_GPS, method='nearest')
    ts_xcount_arg4.plot(ax=ax, color="green", label="Arg4")

    ts_xcount_argG = ds_merged['xcount'].sel(x=x_utm_ArgG_GPS, y=y_utm_ArgG_GPS, method='nearest')
    ts_xcount_argG.plot(ax=ax, color="saddlebrown", label="ArgG")

    ts_xcount_argw = ds_merged['xcount'].sel(x=x_utm_Argw, y=y_utm_Argw, method='nearest')
    ts_xcount_argw.plot(ax=ax, color="blue", label="Argwheel")

    ax.axhline(100, linestyle="--", color="k")
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "arg_xcount.pdf")
    plt.close(fig)



def plot_mean_ts_all_pixels():
    fig = plt.figure()

    velocity.mean(dim=["x", "y"]).plot(label="Original mean vel", color="skyblue")
    vel_avg_detrended.mean(dim=["x", "y"]).plot(label="Spatial avg mean vel", color="blue")
    vel_lowpass_filter.mean(dim=["x", "y"]).plot(label="Filtered mean vel", color="red")
    
    plt.xlabel("Date")
    plt.ylabel(r'Velocity (m yr$^{-1}$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(fig_dir / "mean_ts_all_pixels.pdf")
    plt.close(fig)





if __name__ == "__main__":
    plot_random_pixel_ts()
#    plot_validation_pixel_ts(x0, y0, df_data, name)
#    plot_annual_cycle_validation_point(df, name)
    plot_xcount_ts_validation_points()
    plot_mean_ts_all_pixels()
