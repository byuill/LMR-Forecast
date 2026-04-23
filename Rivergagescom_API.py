import requests
import xml.etree.ElementTree as ET
import csv
import pandas as pd

# ==========================================
# ============ USER DASHBOARD ==============
# ==========================================
# Modify these hard-coded parameters to retrieve different data
TARGET_SITE = "01120"               # e.g., "rcki2"
TARGET_METHOD = "GetValues"         # "GetValues" (time series data) or "GetSiteInfo" (metadata)
TARGET_VARIABLE = "HG"              # e.g., "HP" (Pool Elevation), "HG" (River Stage)
START_DATE = "2023-03-25T00:00"     # Format: YYYY-MM-DDTHH:MM
END_DATE = "2023-04-26T23:59"       # Format: YYYY-MM-DDTHH:MM
OUTPUT_FORMAT = "both"              # Options: "csv", "df", "both"
# ==========================================

class RiverGagesAPI:
    def __init__(self):
        self.base_url = "https://rivergages.mvr.usace.army.mil/watercontrol/webservices/rest/webserviceWaterML.cfc"
        
        # The API requires ALL parameters to be present, even if unused by the method.
        # These default values act as our "dummy" data.
        self.default_params = {
            "method": "RGWML",
            "meth": "GetSites",           # Default method
            "site": "*",                  # Default site wildcard
            "location": "*",              # Default location wildcard
            "variable": "HP",             # Default dummy variable
            "beginDate": "2023-01-01T00:00", # Default dummy date
            "endDate": "2023-01-02T00:00",   # Default dummy date
            "authtoken": "RiverGages",
            "authToken": "RiverGages"     # Included both due to the email's example
        }

    def query(self, meth, site="*", location=None, variable="HP", begin_date=None, end_date=None):
        """
        Executes a query against the RiverGages API.
        Only override the parameters necessary for your specific 'meth'.
        """
        # Start with a fresh copy of the default parameters
        params = self.default_params.copy()

        # Overwrite defaults with any provided arguments
        params["meth"] = meth
        params["site"] = site
        params["location"] = location if location else site
        params["variable"] = variable

        if begin_date:
            params["beginDate"] = begin_date
        if end_date:
            params["endDate"] = end_date

        try:
            # Make the GET request
            response = requests.get(self.base_url, params=params)
            
            # Print the exact URL constructed for debugging purposes
            print(f"Request URL: {response.url}\n")
            
            # Raise an error if the request failed (e.g., 404 or 500 error)
            response.raise_for_status() 
            
            # WaterML APIs typically return XML data
            return response.text 
            
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Initialize the API helper
    api = RiverGagesAPI()

    # Execute the query using the dashboard parameters
    print(f"--- Fetching Data for Site: {TARGET_SITE} | Method: {TARGET_METHOD} ---")
    
    data = api.query(
        meth=TARGET_METHOD,
        site=TARGET_SITE,
        variable=TARGET_VARIABLE,
        begin_date=START_DATE,
        end_date=END_DATE
    )
    
    if data:
        # Print a small snippet to the console
        print(data[:500] + "...\n")
        
        output_file = "output.csv"
        try:
            # Parse the XML response
            root = ET.fromstring(data)
            rows = []
            
            # Look for XML elements that have a 'dateTime' attribute (Standard in WaterML 1.x)
            for elem in root.iter():
                if 'dateTime' in elem.attrib:
                    rows.append([elem.attrib['dateTime'], elem.text])
            
            if rows:
                if OUTPUT_FORMAT in ["df", "both"]:
                    df = pd.DataFrame(rows, columns=["DateTime", "Value"])
                    df["DateTime"] = pd.to_datetime(df["DateTime"])
                    df["Value"] = pd.to_numeric(df["Value"])
                    print(f"[SUCCESS] Created Pandas DataFrame with {len(df)} records.")
                    print(df.head())
                    
                if OUTPUT_FORMAT in ["csv", "both"]:
                    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow(["DateTime", "Value"])
                        writer.writerows(rows)
                    print(f"[SUCCESS] Extracted {len(rows)} records and saved to '{output_file}'.")
            else:
                # Fallback: if no time series data is found (e.g. GetSiteInfo method)
                with open(output_file, mode="w", encoding="utf-8") as file:
                    file.write(data)
                print(f"No time series data found. Raw response saved to '{output_file}'.")
                
        except ET.ParseError:
            # Fallback if the response isn't valid XML
            with open(output_file, mode="w", encoding="utf-8") as file:
                file.write(data)
            print(f"Data saved as raw text to '{output_file}'.")