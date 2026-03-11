import urllib.request
import csv
import concurrent.futures
from datetime import datetime, timedelta

gages = [
    ('07374525', 'Mississippi River at Belle Chasse, LA'),
    ('07374000', 'Mississippi River at Baton Rouge, LA'),
    ('07295100', 'Mississippi River at Tarbert Landing, MS'),
    ('07289000', 'MISSISSIPPI RIVER AT VICKSBURG, MS'),
    ('07047970', 'MISSISSIPPI RIVER AT HELENA, AR'),
    ('07032000', 'MISSISSIPPI RIVER AT MEMPHIS, TN'),
    ('07010000', 'Mississippi River at St. Louis, MO')
]

def download_chunk(site_id, start_str, end_str):
    url = f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site_id}&startDT={start_str}&endDT={end_str}&parameterCd=00060,00065&statCd=00003"
    data_dict = {}
    try:
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req, timeout=30)
        data = response.read().decode('utf-8')

        lines = data.split('\n')

        header = None
        for line in lines:
            if line.startswith('#'): continue
            if line.startswith('agency_cd'):
                header = line.strip().split('\t')
                break

        if header:
            col_indices = {}
            for i, col in enumerate(header):
                if col.endswith('_00060_00003'): col_indices['00060'] = i
                elif col.endswith('_00065_00003'): col_indices['00065'] = i
                elif col == 'datetime': col_indices['datetime'] = i

            if 'datetime' in col_indices:
                for line in lines:
                    if line.startswith('#') or line.startswith('agency_cd') or line.startswith('5s'): continue
                    if not line.strip(): continue

                    parts = line.split('\t')
                    if len(parts) <= col_indices['datetime']: continue

                    dt = parts[col_indices['datetime']]
                    if dt not in data_dict: data_dict[dt] = {}

                    for param in ['00060', '00065']:
                        if param in col_indices and len(parts) > col_indices[param]:
                            val = parts[col_indices[param]]
                            if val.strip() and val != '':
                                data_dict[dt][param] = val
    except Exception as e:
        pass
    return site_id, data_dict

def main():
    start_date = "1900-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate chunks of 10 years for each gage
    tasks = []
    for site_id, _ in gages:
        curr_start = start_dt
        while curr_start <= end_dt:
            curr_end = min(curr_start + timedelta(days=365*10), end_dt)
            tasks.append((site_id, curr_start.strftime("%Y-%m-%d"), curr_end.strftime("%Y-%m-%d")))
            curr_start = curr_end + timedelta(days=1)

    all_data = {site_id: {} for site_id, _ in gages}

    print(f"Starting {len(tasks)} download tasks...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_chunk, *t) for t in tasks]

        for future in concurrent.futures.as_completed(futures):
            site_id, data_dict = future.result()
            for dt, vals in data_dict.items():
                if dt not in all_data[site_id]:
                    all_data[site_id][dt] = {}
                all_data[site_id][dt].update(vals)

    for site_id, _ in gages:
        print(f"Loaded {len(all_data[site_id])} days for {site_id}", flush=True)

    delta = end_dt - start_dt
    dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]

    results = []
    headers = ["date", "discharge", "source_gage", "stage", "stage_source"]

    for dt in dates:
        row = {
            "date": dt,
            "discharge": "NaN",
            "source_gage": "NaN",
            "stage": "NaN",
            "stage_source": "NaN"
        }

        for site_id, site_name in gages:
            if dt in all_data[site_id] and '00060' in all_data[site_id][dt]:
                row["discharge"] = all_data[site_id][dt]['00060']
                row["source_gage"] = site_name
                break

        for site_id, site_name in gages:
            if dt in all_data[site_id] and '00065' in all_data[site_id][dt]:
                row["stage"] = all_data[site_id][dt]['00065']
                row["stage_source"] = site_name
                break

        results.append([row[h] for h in headers])

    output_file = "mississippi_river_data.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

    print(f"Successfully wrote {len(results)} days of data to {output_file}", flush=True)

if __name__ == "__main__":
    main()
