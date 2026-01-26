import os
import requests
import pandas as pd
import io

# Folder to save flags
output_folder = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\downloaded_flags"
os.makedirs(output_folder, exist_ok=True)

# Sources (CSV sources first for reliability)
SOURCES = [
    {
        "type": "csv",
        "url": "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv",
        "parser": lambda df: dict(zip(df["name"], df["alpha-2"])),
    },
    {
        "type": "csv",
        "url": "https://datahub.io/core/country-list/r/data.csv",
        "parser": lambda df: dict(zip(df["Name"], df["Code"])),
    },
    {
        "type": "api",
        "url": "https://restcountries.com/v3.1/all",
        "parser": lambda data: {
            country["name"]["common"]: country["cca2"]
            for country in data
            if "name" in country and "common" in country["name"] and "cca2" in country
        },
    },
]

def fetch_source(src):
    """Try to fetch data from a source and return a name→alpha2 dictionary."""
    try:
        r = requests.get(src["url"], timeout=10)
        r.raise_for_status()
        if src["type"] == "api":
            data = r.json()
            return src["parser"](data)
        else:
            df = pd.read_csv(io.StringIO(r.text))
            return src["parser"](df)
    except Exception as e:
        print(f"⚠️ Error in source {src['url']}: {e}")
        return None

# Try sources in order
country_map = None
for src in SOURCES:
    print(f"🔍 Attempting {src['url']}")
    mapping = fetch_source(src)
    if mapping:
        country_map = mapping
        print(f"✅ Data obtained from {src['url']} ({len(mapping)} countries)")
        break

if not country_map:
    raise RuntimeError("❌ Failed to fetch country list from all sources!")

# List of your 51 included countries (normalized, lowercase)
included_countries = {
    "albania", "algeria", "argentina", "australia", "austria", "bahrain", "belgium", "brazil", "bulgaria", "canada",
    "china", "colombia", "croatia", "denmark", "egypt", "ethiopia", "france", "germany", "greece", "india",
    "indonesia", "iran", "iraq", "ireland", "italy", "japan", "jordan", "korea, republic of", "kuwait", "lebanon",
    "libya", "mexico", "morocco", "new zealand", "nigeria", "norway", "oman", "pakistan", "portugal", "qatar",
    "romania", "russian federation", "saudi arabia", "south africa", "spain", "sudan", "sweden", "switzerland",
    "tunisia", "ukraine", "yemen"
}

def sanitize_country_name(name):
    return (
        name.replace("'", "_")
        .replace("é", "e")
        .replace("ç", "c")
        .replace("å", "a")
        .replace("ö", "o")
        .replace("ô", "o")
        .replace("ü", "u")
    )

# Build excluded_countries as all countries except your 51
excluded_countries = {
    sanitize_country_name(name.lower())
    for name in country_map.keys()
    if sanitize_country_name(name.lower()) not in included_countries
}

# Debug: Track failed downloads
failed_downloads = []

# List of flag URL templates to try for each country code
FLAG_URL_TEMPLATES = [
    "https://flagcdn.com/w320/{code}.png",
    "https://countryflagsapi.com/png/{code}",
    "https://flagpedia.net/data/flags/w580/{code}.png",
]

# Download flags for only your 51 countries
for name, code in country_map.items():
    if not isinstance(code, str) or not code or code.lower() == "nan":
        continue
    country_name = sanitize_country_name(name.lower())
    country_code = code.lower()

    # Only process included countries
    if country_name not in included_countries:
        continue
    # Skip excluded countries (shouldn't happen, but for safety)
    if country_name in excluded_countries:
        continue

    # Skip if flag already exists
    file_path = os.path.join(output_folder, f"{country_name}.png")
    if os.path.exists(file_path):
        print(f"⏩ Flag already exists: {country_name}")
        continue

    # Try each flag source in order
    flag_saved = False
    for url_template in FLAG_URL_TEMPLATES:
        url = url_template.format(code=country_code)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Saved flag: {country_name} from {url}")
                flag_saved = True
                break
            else:
                print(f"❌ Failed ({response.status_code}): {country_name} from {url}")
        except Exception as e:
            print(f"⚠️ Error downloading {country_name} from {url}: {e}")
    if not flag_saved:
        failed_downloads.append(name)

# Debug: Print failed countries
if failed_downloads:
    print("\n⚠️ The following countries failed to download:")
    for failed in failed_downloads:
        print(f"- {failed}")