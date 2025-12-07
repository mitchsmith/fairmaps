import os
from dotenv import load_dotenv
import cartogram
import pygris
import matplotlib.pyplot as plt
import pandas as pd
import pytidycensus
# Load environment variables from a .env file
load_dotenv()

# 1. Fetch the 2020 Decennial Census data for Texas Tracts
# 'geography="tract"' targets the high-resolution unit.
# 'variables=["P1_001N"]' specifies the Total Population variable.
# 'state="TX"' limits the query to Texas.
# Census API key is required. Get one at: https://api.census.gov/data/key_signup.html
# Make sure to set your CENSUS_API_KEY in the .env file or environment variables.

census_data = pytidycensus.get_decennial(
    geography="tract",
    variables=["P1_001N"],
    year=2020,
    state="TX"
)

# 2. Fetch Texas Tract geometries using pygris
texas_tracts = pygris.tracts(state="TX", year=2020)
# 3. Merge census data with geometries
texas_tracts_merged = texas_tracts.merge(census_data, on="GEOID")
# 4. Project to an appropriate CRS for Texas cartogram creation
texas_tracts_merged = texas_tracts_merged.to_crs("EPSG:3083")
# 5. Clean data: Remove tracts with missing population data
texas_tracts_merged = texas_tracts_merged[
    (texas_tracts_merged["P1_001N"] > 0) &
    (texas_tracts_merged["P1_001N"].notna())
    ]
# 6. Create a cartogram based on Total Population
cartogram_tracts = cartogram.cartogram(
    texas_tracts_merged,
    "P1_001N"
)
# 7. Plot the cartogram
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cartogram_tracts.plot(column="P1_001N", cmap="OrRd", linewidth=0.1, ax=ax, edgecolor="black")
ax.set_title("Texas Tract Cartogram - 2020 Total Population", fontsize=15)
ax.axis("off")
plt.show()

def main():
    print("Hello from fairmaps!")


if __name__ == "__main__":
    main()
