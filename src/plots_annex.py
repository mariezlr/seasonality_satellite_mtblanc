from utils import *
from data_exploration import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm


velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)
xcount = np.sqrt(ds_merged['xcount_x']**2 + ds_merged['xcount_y']**2)

# Chemin vers le fichier NetCDF
file_path = out_dir / "analysis_dataset.nc"

# Charger le Dataset
ds_analysis = xr.open_dataset(file_path)
vel_result = ds_analysis["velocity"]
vel_avg_detrended = ds_analysis["vel_detrended"]
vel_lowpass = ds_analysis["vel_lowpass"]
vel_cycle = ds_analysis["vel_cycle"]
avg_velocity = ds_analysis["avg_velocity"]
amplitude = ds_analysis["amplitude"]
melt_cycle = ds_analysis["melt_cycle"]
slope = ds_analysis["slope"]
elevation = ds_analysis["elevation"]
max_peak_doy = ds_analysis["max_peak_doy"]
min_peak_doy = ds_analysis["min_peak_doy"]
inflex_doy = ds_analysis["inflex_doy"]
result_mask = ds_analysis["mask"]


valid_vel_mask = ~vel_cycle.isnull().all(dim='cycle')
valid_indices = np.argwhere(valid_vel_mask.values)



def plot_elevation_map():
    # Créer une figure et un axe
    fig, ax = plt.subplots(figsize=(10, 8))

    print("Statistiques du dem :")
    print(f"Min: {elevation.min().values}, Max: {elevation.max().values}")
    print(f"Moyenne: {elevation.mean().values}, Écart-type: {elevation.std().values}")
    print("Nombre de NaN dans le dem :", np.isnan(elevation).sum())

    # Extraire les coordonnées x et y
    x_coords = elevation.x.values
    y_coords = elevation.y.values

    # Créer une grille de coordonnées
    x, y = np.meshgrid(x_coords, y_coords)

    # Tracer les élévations avec pcolormesh
    cax = ax.pcolormesh(x, y, elevation, cmap='terrain', shading='auto')

    # Ajouter une barre de couleur
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Élevation (m)', fontsize=12)

    # Ajouter un titre et des labels
    ax.set_title('Carte des élévations', fontsize=16)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)

    plt.tight_layout()
    fig.savefig(fig_dir / "elevation_map.png", bbox_inches='tight')
    plt.close(fig)


def plot_slope_map():
    # Créer une figure et un axe
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extraire les coordonnées x et y
    x_coords = slope.x.values
    y_coords = slope.y.values

    # Créer une grille de coordonnées
    x, y = np.meshgrid(x_coords, y_coords)

    # Tracer les élévations avec pcolormesh
    cax = ax.pcolormesh(x, y, slope, cmap='viridis', shading='auto')

    # Ajouter une barre de couleur
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Pente (°)', fontsize=12)

    # Ajouter un titre et des labels
    ax.set_title('Carte des pentes', fontsize=16)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)

    # Sauvegarder et afficher le graphique
    plt.tight_layout()
    fig.savefig(fig_dir / "slope_map.png", bbox_inches='tight')
    plt.close(fig)


def plot_slope_distribution():
    # Créer une figure et un axe
    fig, ax = plt.subplots(figsize=(10, 6))

    # Appliquer le masque pour obtenir uniquement les valeurs valides
    slope_values = slope.values.flatten()

    # Tracer un histogramme
    sns.histplot(slope_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

    # Ajouter des labels et un titre
    ax.set_title('Distribution des pentes', fontsize=16)
    ax.set_xlabel('Pente (degrés)', fontsize=14)
    ax.set_ylabel('Fréquence', fontsize=14)

    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='upper left')

    # Sauvegarder et afficher le graphique
    plt.tight_layout()
    fig.savefig(fig_dir / "slope_distribution.png", bbox_inches='tight')
    plt.close(fig)


def plot_random_pixel_ts():
    # Trouver les indices des pixels valides
    notnan_pixels = vel_result.notnull().sum(dim='mid_date')
    valid_indices = np.argwhere(notnan_pixels.values >= 1)

    # Sélectionner le premier pixel valide (ou un aléatoire)
    y_idx, x_idx = valid_indices[0]

    # Extraire la série temporelle
    ts_orig_local = velocity.isel(y=y_idx, x=x_idx)
    ts_orig = vel_avg_detrended.isel(y=y_idx, x=x_idx)
    ts_filter = vel_lowpass.isel(y=y_idx, x=x_idx)

    fig = plt.figure(figsize=(10, 4))
    ts_orig_local.plot(label="Original", color="skyblue")
    ts_orig.plot(label="Spatial Avg", color="blue")
    ts_filter.plot(label="Filtered", color="red")

    plt.title(f'Série temporelle pour le pixel (y={y_idx}, x={x_idx})')
    plt.xlabel('Date')
    plt.ylabel('Vitesse')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    fig.savefig(fig_dir / "random_pixel_ts.png")
    plt.close(fig)


def plot_validation_pixel_ts(x0, y0, df_data, name):

    # Extraire la série temporelle originale et filtrée à ce point avant filtrage
    ts_orig_local = velocity.sel(x=x0, y=y0, method='nearest')
    ts_orig = vel_avg_detrended.sel(x=x0, y=y0, method='nearest')
    ts_filter = vel_lowpass.sel(x=x0, y=y0, method='nearest')

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
    fig.savefig(fig_dir / f"validation_{name}.png")
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
    fig.savefig(fig_dir / f"annual_cycle_{name}.png")
    plt.close(fig)


def plot_xcount_ts_validation_points():
    fig, ax = plt.subplots()

    ts_xcount_arg4 = xcount.sel(x=x_utm_Arg4_GPS, y=y_utm_Arg4_GPS, method='nearest')
    ts_xcount_arg4.plot(ax=ax, color="green", label="Arg4")

    ts_xcount_argG = xcount.sel(x=x_utm_ArgG_GPS, y=y_utm_ArgG_GPS, method='nearest')
    ts_xcount_argG.plot(ax=ax, color="saddlebrown", label="ArgG")

    ts_xcount_argw = xcount.sel(x=x_utm_Argw, y=y_utm_Argw, method='nearest')
    ts_xcount_argw.plot(ax=ax, color="blue", label="Argwheel")

    ax.axhline(100, linestyle="--", color="k")
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "arg_xcount.png")
    plt.close(fig)



def plot_mean_ts_all_pixels():
    fig = plt.figure()

    velocity.mean(dim=["x", "y"]).plot(label="Original mean vel", color="skyblue")
    vel_avg_detrended.mean(dim=["x", "y"]).plot(label="Spatial avg mean vel", color="blue")
    vel_lowpass.mean(dim=["x", "y"]).plot(label="Filtered mean vel", color="red")
    
    plt.xlabel("Date")
    plt.ylabel(r'Velocity (m yr$^{-1}$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(fig_dir / "mean_ts_all_pixels.png")
    plt.close(fig)


def plot_random_pixels_avg_year():
    selected_pix = valid_indices[np.random.choice(len(valid_indices), 10, replace=False)]

    fig, ax = plt.subplots(figsize=(10,5))

    for i, (iy, ix) in enumerate(selected_pix):
        ax.plot(np.arange(1, 366), vel_cycle[:, iy, ix], label=f"Pixel {i+1}")

    ax.set_xlabel("Day of year")
    ax.set_ylabel("Velocity (m yr$^{-1}$)")
    ax.set_title("Mean seasonal cycle (±2 days, cyclic) for 10 random pixels")
    ax.grid(True)
    ax.legend(ncol=2)

    plt.tight_layout()
    fig.savefig(fig_dir / "avg_vel_random_pixels.png")
    plt.close(fig)


def plot_meteofrance_map():
    fig, ax = plt.subplots(figsize=(10,8))

    # Fond satellite
    ax.imshow(np.moveaxis(img_map, 0, -1), extent=extent_map, origin='upper')

    x = np.arange(x_min_mf, x_max_mf, 1)
    y = [m*x+b for x in x]
    ax.plot(x,y, color = "red", label="Limite du massif")

    ax.scatter(df_mtblanc["x_utm"].values, df_mtblanc["y_utm"].values, color = "red", label="Stations Météo France")

    ax.set_xlim(np.nanmin(x_1d), np.nanmax(x_1d))
    ax.set_ylim(np.nanmin(y_1d), np.nanmax(y_1d))


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=16)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(fig_dir / "map_stations_meteo.png")
    plt.close(fig)


def plot_taub_per_glacier_Elmer():
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    
    axes = axes.flatten() 
    colors = cm.tab10.colors

    for i, (glacier, info) in enumerate(glaciers.items()):
        df = pd.read_csv(out_dir / f"{glacier}.csv")
        
        # Sélectionner l'axe correspondant
        ax = axes[i]
        color = colors[i % len(colors)] 
        
        mask = df["slope"] < 40
        tau_b_vals = df["tau_b"].where(mask).values.flatten()
        slope_vals = df["slope"].where(mask).values.flatten()
        
        # Moyenne glissante
        centers, mean_tau_b = moving_average(tau_b_vals, slope_vals, step_taub_Elmer, window_taub_Elmer)
        
        # Tracer
        ax.scatter(slope_vals, tau_b_vals, color="grey", alpha=0.4, s=2)
        ax.plot(centers, mean_tau_b, color=color, linewidth=2, label=fr"Mean $\tau_b$ {glacier}")
        
        ax.set_title(glacier, fontsize=20)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=18)

    # Labels communs
    fig.text(0.5, 0.04, "Slope (°)", ha='center', fontsize=22)
    fig.text(0.04, 0.5, "Basal shear stress (MPa)", va='center', rotation='vertical', fontsize=22)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    fig.savefig(fig_dir / "tau_b_by_slope_all_glaciers.png")
    plt.close(fig)


def plot_taub_all_glaciers_Elmer():
    all_slope = []
    all_tau_b = []

    for i, (glacier, info) in enumerate(glaciers.items()):
        df = pd.read_csv(out_dir / f"{glacier}.csv")

        mask = df["slope"] < 40        
        all_slope.append(df["slope"].where(mask).values.flatten())
        all_tau_b.append(df["tau_b"].where(mask).values.flatten())

    all_slope = np.concatenate(all_slope)
    all_tau_b = np.concatenate(all_tau_b)
        
    # Moyenne glissante
    centers, mean_tau_b = moving_average(all_tau_b, all_slope, step_taub_Elmer, window_taub_Elmer)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(all_slope, all_tau_b, color="grey", alpha=0.3, s=2, label="All points")
    ax.plot(centers, mean_tau_b, color="red", linewidth=2, label=fr"Mean $\tau_b$")

    ax.set_xlabel("Slope (°)", fontsize=14)
    ax.set_ylabel("Basal shear stress (MPa)", fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(markerscale=5, fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / "tau_b_by_slope_all_glaciers_combined.png")
    plt.close(fig)


def plot_low_taub_location_Elmer():
    fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    # critères low-stress
    tau_thresh = 0.03
    slope_thresh = 5

    # Pour le mappable global
    norm = None
    sm = None

    for i, (glacier, info) in enumerate(glaciers.items()):
        df = pd.read_csv(out_dir / f"{glacier}.csv")

        ax = axes[i]
        mask_low = (df["tau_b"] < tau_thresh) & (df["slope"] < slope_thresh)

        # --- All points in grey ---
        ax.scatter(df["xcoord"], df["ycoord"], s=2, color="lightgrey", alpha=0.5)

        # --- Points with low shear stress = coloured by year ---
        sc = ax.scatter(df.loc[mask_low, "xcoord"],
                        df.loc[mask_low, "ycoord"],
                        s=6,
                        c=df.loc[mask_low, "year"],
                        cmap="turbo",
                        alpha=0.9)

        # ScalarMappable for global colorbar
        if sm is None:
            sm = sc
            norm = plt.Normalize(vmin=df.loc[mask_low, "year"].min(),
                                vmax=df.loc[mask_low, "year"].max())

        ax.set_title(glacier, fontsize=18)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Labels
    fig.text(0.5, 0.04, "x (m)", ha='center', fontsize=20)
    fig.text(0.04, 0.5, "y (m)", va='center', rotation='vertical', fontsize=20)

    # Global colobar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend='both')
    cbar.set_label("Year", fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # laisser de la place pour la colorbar
    fig.savefig(fig_dir / "map_low_taub_location_8_glaciers.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_elevation_map()
    plot_slope_map()
    plot_slope_distribution()
    plot_random_pixel_ts()
    plot_validation_pixel_ts(x_utm_Arg4_GPS, y_utm_Arg4_GPS, data_GPS_ARG4, "Arg4")
    plot_annual_cycle_validation_point(data_GPS_ARG4, "Arg4")
    plot_validation_pixel_ts(x_utm_ArgG_GPS, y_utm_ArgG_GPS, data_GPS_ARGG, "ArgG")
    plot_annual_cycle_validation_point(data_GPS_ARGG, "ArgG")
    plot_validation_pixel_ts(x_utm_Argw, y_utm_Argw, data_Argwheel, "Arg wheel")
    plot_annual_cycle_validation_point(data_Argwheel, "Arg wheel")
    plot_xcount_ts_validation_points()
    plot_mean_ts_all_pixels()
    plot_random_pixels_avg_year()
    plot_meteofrance_map()
    plot_taub_per_glacier_Elmer()
    plot_taub_all_glaciers_Elmer()
    plot_low_taub_location_Elmer()
