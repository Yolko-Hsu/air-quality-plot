# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 00:35:31 2025

@author: CALab
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import re
import html
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from git import Repo

def upload_to_github():
    repo = Repo(REPO_PATH)
    
    repo.git.add(IMAGE_PATH)
    repo.index.commit(f"Update plot: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    origin = repo.remotes.origin
    origin.push()

def fetch_api_data(api_url):
    """Fetch data from API and return as JSON"""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching API data: {e}")
        return None

def process_moe_data(records, site_name):
    """Process MOENV data records into a dictionary indexed by datetime"""
    pre_merge_dict = {}
    
    for entry in records:
        date = entry['monitordate'] 
        measure_type = entry['itemengname']

        for hour in range(24):
            hour_key = f"monitorvalue{hour:02d}"
            MOE = entry.get(hour_key, None)  
            
            if MOE is not None:
                MOE_str = str(MOE).strip() 
                if '*' in MOE_str:
                    MOE = np.nan 
                else:
                    try:
                        MOE = float(MOE_str)
                    except ValueError:
                        MOE = np.nan 
                
                datetime_str = f"{date} {hour:02d}:00"
                try:
                    datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    print(f"Invalid datetime format: {datetime_str}")
                    continue  
                
                if datetime_obj not in pre_merge_dict:
                    pre_merge_dict[datetime_obj] = {}
                pre_merge_dict[datetime_obj][measure_type] = MOE
    
    return pre_merge_dict

def process_ncu_data(ncu_data):
    """Process NCU data into a dictionary with parameter values indexed by datetime"""
    ncu_dict = {}
    
    for entry in ncu_data:
        date_str = str(entry['日期時間'])
        if date_str and date_str not in ['nan', '']:
            try:
                time = datetime.strptime(date_str, '%Y/%m/%d %H:%M:%S')
                
                # Process each parameter
                parameters = ['PM25', 'THC', 'CH4', 'NMHC', 'O3', 'NO2', 'NO', 'CO', 'SO2', 'CO2', 'NOX']
                
                if time not in ncu_dict:
                    ncu_dict[time] = {}
                
                for param in parameters:
                    try:
                        param_value = float(entry.get(param, np.nan))
                        ncu_dict[time][param] = param_value
                    except (ValueError, TypeError):
                        ncu_dict[time][param] = np.nan
                
                # Calculate NOx if needed
                if 'NOX' not in ncu_dict[time] and 'NO' in ncu_dict[time] and 'NO2' in ncu_dict[time]:
                    if not np.isnan(ncu_dict[time]['NO']) and not np.isnan(ncu_dict[time]['NO2']):
                        ncu_dict[time]['NOX'] = ncu_dict[time]['NO'] + ncu_dict[time]['NO2']
                
            except ValueError:
                print(f"Skipping invalid date: {date_str}")
    
    return ncu_dict

def create_combined_dataframe(moe_dict, ncu_dict, params_mapping):
    """Create a dataframe combining MOE and NCU data for all parameters"""
    # Get all times from both datasets
    common_times = set(moe_dict.keys()) & set(ncu_dict.keys())

    if not common_times:
        raise ValueError("沒有兩個字典都有的時間點")
    
    # 2. 取出最近的最後一筆時間（最大值）
    end_time = max(common_times)
    
    # 3. 計算倒數 7 天的起始時間
    start_time = end_time - timedelta(days=7)
    
    # 4. 篩選出在這段時間內的共同時間點
    filtered_times = [t for t in common_times if start_time <= t <= end_time]
    
    # 5. 建立完整的時間範圍（每小時一次）
    full_time_range = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # 6. 初始化 DataFrame 來存放時間
    df = pd.DataFrame({'time': full_time_range})
    
    # 7. 根據共同時間點填充資料（若資料不存在，則填 NaN）
    df['moe_data'] = df['time'].map(lambda t: moe_dict.get(t, None))
    df['ncu_data'] = df['time'].map(lambda t: ncu_dict.get(t, None))

    
    # Add data for each parameter
    for moe_param, ncu_param in params_mapping.items():
        # Add MOE data
        moe_data = [float(moe_dict.get(time, {}).get(moe_param, np.nan)) for time in full_time_range]
        df[f'moe_{moe_param}'] = pd.to_numeric(moe_data, errors='coerce')
        
        # Add NCU data
        ncu_data = [float(ncu_dict.get(time, {}).get(ncu_param, np.nan)) for time in full_time_range]
        df[f'ncu_{ncu_param}'] = pd.to_numeric(ncu_data, errors='coerce')
    
    return df

def plot_air_quality_parameters(df, params_mapping, y_limits, site_name):
    """Create subplots for each parameter"""
    n_params = len(params_mapping)
    
    # Calculate grid dimensions
    n_rows = (n_params + 1) // 2  # Round up division
    n_cols = min(2, n_params)  # At most 2 columns
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows), dpi=300)
    
    # Handle single subplot case
    if n_params == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Plot each parameter
    for i, (moe_param, ncu_param) in enumerate(params_mapping.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Plot data
            ax.plot(df['time'], df[f'moe_{moe_param}'],  color='blue')
            ax.plot(df['time'], df[f'ncu_{ncu_param}'],  color='red')
            
            # Set titles and labels
            display_param = moe_param.replace('2.5', '$_{2.5}$')
            ax.set_title(f'{display_param}')
            
            # Set y-axis limits
            ylim_key = moe_param if moe_param != 'PM2.5' else 'PM25'
            if ylim_key in y_limits:
                ax.set_ylim(y_limits[ylim_key])
            
            # Set y-axis label with units
            if ylim_key == 'PM25':
                ax.set_ylabel(f'{display_param} (μg/m$^3$)')
            else:
                ax.set_ylabel(f'{display_param} (ppb)')
            
            # Format x-axis
            dfmt = mdates.DateFormatter('%m/%d')
            ax.xaxis.set_major_formatter(dfmt)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            plt.setp(ax.get_xticklabels(), ha="center")
            
            # Add legend
            ax.legend(loc='upper right')
    
    # Hide unused subplots
    for i in range(len(params_mapping), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    current_date = datetime.now().strftime('%Y-%m-%d')
    plt.suptitle(f'NAQO(red) and MOE(blue) data on {current_date}', fontsize=18, y=1.02)
    # plt.subplots_adjust(top=0.92)
    
    return fig

# Main execution
if __name__ == "__main__":
    # Define y-limits for parameters
    y_limits = {
        'O3': (0, 70),
        'PM25': (0, 70),
        'NMHC': (0, 0.6),  # For Zhongli site
        'SO2': (0, 9),
        'THC': (0, 3.9),
        'NOx': (0, 89),
        'CH4': (0, 3.9),
        'CO2': (100, 700),
        # 'NO': (0, 89),
        'NO': (0, 25),
        'CO': (0, 4)
    }
    
    # Define parameter mapping (MOE parameter name -> NCU parameter name)
    params_mapping = {
        'PM2.5': 'PM25', 
        'O3': 'O3',
        'NO': 'NO',
        'NO': 'NO',
        'CO': 'CO',
        'SO2': 'SO2',
        'THC': 'THC',
        'CH4': 'CH4',
        'NMHC': 'NMHC'
    }
    
    # First, get NMHC data from Zhongli site
    zhongli_api = 'https://data.moenv.gov.tw/api/v2/AQX_P_15?%20format=json&offset=0&limit=360&api_key=6d7ca78f-ac98-45cf-99aa-6b5d0c861542&filters=SiteName,EQ,中壢'
    zhongli_data = fetch_api_data(zhongli_api)
    
    if zhongli_data and 'records' in zhongli_data:
        zhongli_dict = process_moe_data(zhongli_data['records'], 'Zhongli')
    else:
        print("Error fetching Zhongli data")
        zhongli_dict = {}
    
    # Then get data for other parameters from Pingjen site
    pingjen_api = 'https://data.moenv.gov.tw/api/v2/AQX_P_15?%20format=json&offset=0&limit=460&api_key=6d7ca78f-ac98-45cf-99aa-6b5d0c861542&filters=SiteName,EQ,平鎮'
    pingjen_data = fetch_api_data(pingjen_api)
    
    if pingjen_data and 'records' in pingjen_data:
        pingjen_dict = process_moe_data(pingjen_data['records'], 'Pingjen')
    else:
        print("Error fetching Pingjen data")
        pingjen_dict = {}
    
    # Combine the dictionaries with Zhongli taking precedence for NMHC
    moe_dict = pingjen_dict.copy()
    
    # For each timestamp in the Zhongli data, replace or add the NMHC value
    for time, values in zhongli_dict.items():
        if 'NMHC' in values:
            if time not in moe_dict:
                moe_dict[time] = {}
            moe_dict[time]['NMHC'] = values['NMHC']
        if 'THC' in values:
            if time not in moe_dict:
                moe_dict[time] = {}
            moe_dict[time]['THC'] = values['THC']
        if 'CH4' in values:
            if time not in moe_dict:
                moe_dict[time] = {}
            moe_dict[time]['CH4'] = values['CH4']
    


    current_year = datetime.now().year
    current_month = datetime.now().month
        
    ncu_api = f'https://tortoise-fluent-rationally.ngrok-free.app/api/60min/json/{current_year}{str(current_month).zfill(2)}04'
    # ncu_api = 'https://tortoise-fluent-rationally.ngrok-free.app/api/60min/json/202504'
    response = requests.get(ncu_api)
    
    if response.status_code == 200:
        response_text = response.text.splitlines()[2:]
        response_cleaned = "\n".join(response_text)
        response_decoded = html.unescape(response_cleaned)
        response_no_html = re.sub(r'<.*?>', '', response_decoded)
        
        ncu_data = json.loads(response_no_html)
        ncu_dict = process_ncu_data(ncu_data)
    else:
        print(f"NCU API error: {response.status_code}")
        ncu_dict = {}
    
    # Create combined dataframe
    df = create_combined_dataframe(moe_dict, ncu_dict, params_mapping)
    
    # Create and display plots
    fig = plot_air_quality_parameters(df, params_mapping, y_limits, "Sites")
    # plt.show()
    
    REPO_PATH = './air-quality-plot'  # 你的 GitHub repo 路徑
    IMAGE_PATH = os.path.join(REPO_PATH, 'docs', 'air_quality_parameters_7days.png')

    fig.savefig(IMAGE_PATH, bbox_inches='tight', dpi=300)
    upload_to_github()
    
    

