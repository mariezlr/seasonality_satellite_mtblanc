from data_exploration import *
from utils import *
import os
import numpy as np
import pandas as pd
import re


# Calcul du shear stress tau_b = f(u_b) avec Weertman tel que défini dans le wiki Elmer
def elmer_calcul_tau_b(u_b, C, m=3):
    return C * (u_b**(1/m))


def elmer_nodes_tau_slope(path, years, C, m=3, Hmin=20):
    all_nodes = []

    for year in years:
        # Charger les fichiers BedNode et SurfaceNode
        pattern_bed = re.compile(fr"BedNode{year}_part(\d+)\.dat")
        pattern_surf = re.compile(fr"SurfaceNode{year}_part(\d+)\.dat")

        filepaths_bed = [
            os.path.join(path, f) for f in os.listdir(path)
            if pattern_bed.match(f) and not f.endswith('.names')
        ]
        filepaths_surf = [
            os.path.join(path, f) for f in os.listdir(path)
            if pattern_surf.match(f) and not f.endswith('.names')
        ]

        dfs = []

        for fbed, fsurf in zip(filepaths_bed, filepaths_surf):
            # lecture noeuds du lit
            df_bed = pd.read_csv(fbed, sep=r'\s+', header=None, names=[
                'xcoord','ycoord','xvelbed','yvelbed','zvelbed',
                'thickbed','nodearea','load1','load2','load3',
                'normalbed1','normalbed2','normalbed3',
                'normalload','normalstress'
            ]).drop_duplicates()

            # lecture surface
            df_surf = pd.read_csv(fsurf, sep=r'\s+', header=None, names=[
                'xcoord','ycoord','xvelsurf','yvelsurf','zvelsurf',
                'thicksurf','projstress','normalstressbis',
                'normalbed1bis','normalbed2bis','normalbed3bis',
                'xgrad','ygrad','zgrad'
            ]).drop_duplicates()

            # fusion noeud
            df = pd.merge(df_bed, df_surf, on=['xcoord','ycoord'])
            df['year'] = year
            dfs.append(df)

        df_year = pd.concat(dfs, ignore_index=True)

        # Calculs noeud
        df_year = df_year[df_year['thicksurf'] >= Hmin].copy()

        df_year['vel_h_bed'] = np.sqrt(df_year['xvelbed']**2 + df_year['yvelbed']**2)
        df_year['vel_h_surf'] = np.sqrt(df_year['xvelsurf']**2 + df_year['yvelsurf']**2)

        # pente noeud
        df_year['slope'] = np.degrees(np.arctan(np.sqrt(df_year['xgrad']**2 + df_year['ygrad']**2)))

        # tau_b noeud
        df_year['tau_b'] = elmer_calcul_tau_b(df_year['vel_h_bed'], C, m)

        all_nodes.append(df_year)

    return pd.concat(all_nodes, ignore_index=True)


### If mean tau_b files do not exist for one or more of the 8 glaciers, creation of the csv file
for i, (glacier, info) in enumerate(glaciers.items()):
    output_file = out_dir / f"{glacier}.csv"

    if not output_file.exists():
        print(f"File {output_file} doesn't exist. Creation of the file...")

        df = elmer_nodes_tau_slope(
            path = elmer_base_dir / f"{glacier}/Elmer_Init/As_accurate/GlacierOut",
            years = info["years"],
            C     = info["C"],
            m     = 3
        )

        # Sauvegarde du DataFrame en CSV
        df.to_csv(output_file, index=False)
        print(f"Saved {glacier} data to {output_file}")



### If mean tau_b on the 8 glacier doesn't exist, creation of the csv file
taub_path = Path(out_dir / "tau_b_mean_8_glaciers.csv")

if not taub_path.exists():
    print(f"File {taub_path} doesn't exist. Creation of the file...")

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
    
    tau_b_mean_df = pd.DataFrame({"slope_values": centers, "tau_b_mean": mean_tau_b})
    tau_b_mean_df.to_csv(taub_path, index=False)
