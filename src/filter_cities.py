# The raw cities.csv/trips.csv/stations.csv this script reads have since been
# deleted -- the filtered output (data/interim/stations.parquet,
# station_status.parquet, trips.parquet) is already in place. This file is
# kept only as documentation of how that filtering was originally done; it
# can no longer be run.
import argparse
import os
import pandas as pd

# Some cities appear under a local-language name in this dataset (Vienna -> Wien),
# so match against a few known name variants.
TARGET_CITY_NAMES = ["Bilbao", "Vienna", "Wien", "Glasgow", "Freiburg", "Freiburg im Breisgau"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder containing cities.csv, trips.csv, stations.csv")
    parser.add_argument("--out-dir", required=True, help="Folder to write the filtered CSVs into")
    args = parser.parse_args()

    cities = pd.read_csv(f"{args.data_dir}/cities.csv")
    trips = pd.read_csv(f"{args.data_dir}/trips.csv")
    stations = pd.read_csv(f"{args.data_dir}/stations.csv")

    mask = cities["name"].str.lower().isin([c.lower() for c in TARGET_CITY_NAMES])
    target_cities = cities[mask].copy()

    print("Matched cities:")
    print(target_cities[["id", "name", "country"]])

    # Rename before merging so we don't collide with stations.csv's own "name" column (station name).
    city_lookup = target_cities[["id", "name"]].rename(columns={"name": "city_name"})
    target_ids = target_cities["id"].tolist()

    filtered_trips = trips[trips["city_id"].isin(target_ids)]
    filtered_stations = stations[stations["city_id"].isin(target_ids)]

    print("\nDepartures per city:")
    print(filtered_trips.merge(city_lookup, left_on="city_id", right_on="id").groupby("city_name").size())

    print("\nStations per city:")
    print(filtered_stations.merge(city_lookup, left_on="city_id", right_on="id").groupby("city_name").size())

    os.makedirs(args.out_dir, exist_ok=True)
    target_cities.to_csv(f"{args.out_dir}/cities_filtered.csv", index=False)
    filtered_trips.to_csv(f"{args.out_dir}/trips_filtered.csv", index=False)
    filtered_stations.to_csv(f"{args.out_dir}/stations_filtered.csv", index=False)
    print(f"\nSaved filtered CSVs to {args.out_dir}")

if __name__ == "__main__":
    main()
