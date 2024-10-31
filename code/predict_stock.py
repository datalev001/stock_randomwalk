import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model
import yfinance as yf

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def getdatabyday(tiklst, st, ed):
    download_data = pd.DataFrame([])
    interval = '1d'
    LLL = len(tiklst)
    KKK = 1
    
    for tk in tiklst:
        stock = yf.Ticker(tk)
        stock_data = \
        stock.history(start = st, end = ed, interval= '1d').reset_index()
        if len(stock_data) > 0:
            stock_data['ticker'] = tk
            download_data = pd.concat([download_data, stock_data])
            print ('daily data tickname: ', tk)
            KKK += 1
        
    download_data = download_data.sort_values(['ticker', 'Date'], ascending = [1, 0])
    z = download_data['Date'].astype(str)
    download_data['Date'] = z
    dya_c = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    download_data = download_data[dya_c]
    tm_frame = pd.DataFrame(list(set(download_data['Date'])), columns = ['Date'])
    tm_frame = tm_frame.sort_values(['Date'], ascending = False)
    tm_frame['dayseq'] = range(1, len(tm_frame) + 1)
    download_data = pd.merge(download_data, tm_frame, on= ['Date'] , how='inner')
    download_data = download_data.sort_values(['ticker', 'Date'], ascending = [False, False])
    return download_data

day_data = getdatabyday(tks, '2019-10-01', '2024-10-20')

## load data, merge day_data with sec_df_etf
stock_df = pd.read_csv('stock_fullsegment.csv')

###1) analyze by time range#########

def stock_analysis(stocks_data, start_date, end_date):

    stocks = stocks_data.copy()

    # Filter data based on the provided date range
    filtered_df = stocks[(stocks['yyyymmdd'] >= start_date) & (stocks['yyyymmdd'] <= end_date)]
    tickers = filtered_df['Ticker'].unique()
    results = []

    # Step 1: Iterate over each stock and perform the tests
    for ticker in tickers:
        stock_data = filtered_df[filtered_df['Ticker'] == ticker].sort_values(by='Date')

        if stock_data.shape[0] < 10:  # Skip stocks with insufficient data points
            continue

        # Extract the sector for the current ticker
        sector = stock_data['Sector'].iloc[0] if 'Sector' in stock_data.columns else "NA"

        # Log returns for stationarity testing
        stock_data['Log_Return'] = np.log(stock_data['Close']).diff()

        # Remove NaN and infinite values from log returns
        stock_data = stock_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_Return'])

        # Skip if stock_data is constant after transformation
        if stock_data['Log_Return'].nunique() <= 1:
            continue

        # Augmented Dickey-Fuller Test for Random Walk with adjusted p-value calculation
        try:
            # ADF test on stock log returns
            adf_result_stock = adfuller(stock_data['Log_Return'])
            adf_p_value_stock = adf_result_stock[1]

            # Generate white noise series and ADF test on it
            white_noise = np.random.normal(0, 1, len(stock_data['Log_Return']))
            adf_result_noise = adfuller(white_noise)
            adf_p_value_noise = adf_result_noise[1]

            # Adjusted p-value
            adj_adf_p_value = min(0.05*adf_p_value_stock / (adf_p_value_noise + 0.0000001), 0.99)
        except ValueError:
            continue  # Skip stocks that fail ADF test due to constant series

        # Calculate dynamic nlags based on the sample size
        dynamic_nlags = max(1, min(10, stock_data['Log_Return'].shape[0] // 2))

        # ACF and PACF Values with dynamic nlags
        acf_values = acf(stock_data['Log_Return'], nlags=dynamic_nlags)
        pacf_values = pacf(stock_data['Log_Return'], nlags=dynamic_nlags)

        # ARCH Test using GARCH(1,1) Model
        try:
            arch_model_instance = arch_model(stock_data['Log_Return'], vol='Garch', p=1, q=1)
            arch_fit = arch_model_instance.fit(disp='off')
            arch_lm_test_result = arch_fit.arch_lm_test()
            arch_p_value = arch_lm_test_result.pval  # Extract the p-value
        except Exception:
            arch_p_value = np.nan  # If GARCH fails, assign NaN

        # Store results as a dictionary
        results.append({
            'Ticker': ticker,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Sector': sector,
            'Adjusted_ADF_p_value': adj_adf_p_value,
            'ARCH_p_value': arch_p_value,
            'ACF_values': list(acf_values),
            'PACF_values': list(pacf_values)
        })

    # Step 2: Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Step 3: Generate aggregated statistics
    if not results_df.empty:
        # Count stocks that rejected the null hypothesis of ADF (i.e., non-random walk)
        non_random_walk_count = results_df[results_df['Adjusted_ADF_p_value'] < 0.05].shape[0]
        non_random_walk_percentage = (non_random_walk_count / len(results_df)) * 100

        # Count stocks with significant ARCH effect (p-value < 0.05)
        significant_arch_count = results_df[results_df['ARCH_p_value'] < 0.05].shape[0]
        significant_arch_percentage = (significant_arch_count / len(results_df)) * 100

        # Average Adjusted ADF and ARCH p-values
        avg_adf_p_value = results_df['Adjusted_ADF_p_value'].mean()
        avg_arch_p_value = results_df['ARCH_p_value'].mean()

        # Estimate the risk and predictability
        risk_description = (
            "High risk for building a predictive model" if significant_arch_percentage > 50 
            else "Moderate risk for building a predictive model"
        )
        predictability_description = (
            "Stocks are generally predictable" if non_random_walk_percentage > 50 
            else "Stocks are mostly following a random walk"
        )

        # Create the aggregated results DataFrame
        aggregated_results = [{
            'Total_Stocks_Analyzed': len(results_df),
            'Percentage_Non_Random_Walk': non_random_walk_percentage,
            'Percentage_Significant_ARCH': significant_arch_percentage,
            'Average_Adjusted_ADF_p_value': avg_adf_p_value,
            'Average_ARCH_p_value': avg_arch_p_value,
            'Risk_Description': risk_description,
            'Predictability_Description': predictability_description
        }]
        aggregated_results_df = pd.DataFrame(aggregated_results)
    else:
        # Handle case where no stocks are analyzed
        aggregated_results_df = pd.DataFrame([{
            'Total_Stocks_Analyzed': 0,
            'Percentage_Non_Random_Walk': 0.0,
            'Percentage_Significant_ARCH': 0.0,
            'Average_Adjusted_ADF_p_value': None,
            'Average_ARCH_p_value': None,
            'Risk_Description': 'No stocks analyzed',
            'Predictability_Description': 'No stocks analyzed'
        }])
    aggregated_results_df['start_date'], aggregated_results_df['end_date'] = start_date, end_date
    
    return results_df, aggregated_results_df

# Run analysis for a specific time range
aggregated_df = pd.DataFrame([])
for st, ed in [(20220101, 20231231), (20230101, 20240301), (20230601, 20240301), (20220720, 20230201), (20240301, 20240601)]:
    results_df, aggregated_results_df = stock_analysis(stock_df, start_date=st, end_date=ed)
    aggregated_df = pd.concat([aggregated_df, aggregated_results_df], ignore_index=True)

# Print the aggregated results
print(aggregated_df)

###analyze by sector###############
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model

def stock_analysis(stocks_data, start_date, end_date):

    stocks = stocks_data.copy()

    # Filter data based on the provided date range
    filtered_df = stocks[(stocks['yyyymmdd'] >= start_date) & (stocks['yyyymmdd'] <= end_date)]
    tickers = filtered_df['Ticker'].unique()
    results = []

    # Iterate over each stock and perform the tests
    for ticker in tickers:
        stock_data = filtered_df[filtered_df['Ticker'] == ticker].sort_values(by='Date')

        if stock_data.shape[0] < 10:  # Skip stocks with insufficient data points
            continue

        # Extract the sector for the current ticker
        sector = stock_data['Sector'].iloc[0] if 'Sector' in stock_data.columns else "NA"

        # Log returns for stationarity testing
        stock_data['Log_Return'] = np.log(stock_data['Close']).diff()

        # Remove NaN and infinite values from log returns
        stock_data = stock_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_Return'])

        # Skip if stock_data is constant after transformation
        if stock_data['Log_Return'].nunique() <= 1:
            continue

        # Augmented Dickey-Fuller Test for Random Walk with adjustment
        try:
            # ADF test on stock log returns
            adf_result_stock = adfuller(stock_data['Log_Return'])
            adf_p_value_stock = adf_result_stock[1]

            # Generate white noise series and ADF test on it
            white_noise = np.random.normal(0, 1, len(stock_data['Log_Return']))
            adf_result_noise = adfuller(white_noise)
            adf_p_value_noise = adf_result_noise[1]

            # Adjusted p-value
            adj_adf_p_value = min(0.05 * adf_p_value_stock / (adf_p_value_noise + 0.0000001), 0.99)
        except ValueError:
            continue  # Skip stocks that fail ADF test due to constant series

        # Calculate dynamic nlags based on the sample size
        dynamic_nlags = max(1, min(10, stock_data['Log_Return'].shape[0] // 2))

        # ACF and PACF Values with dynamic nlags
        acf_values = acf(stock_data['Log_Return'], nlags=dynamic_nlags)
        pacf_values = pacf(stock_data['Log_Return'], nlags=dynamic_nlags)

        # ARCH Test using GARCH(1,1) Model
        try:
            arch_model_instance = arch_model(stock_data['Log_Return'], vol='Garch', p=1, q=1)
            arch_fit = arch_model_instance.fit(disp='off')
            arch_lm_test_result = arch_fit.arch_lm_test()
            arch_p_value = arch_lm_test_result.pval  # Extract the p-value
        except Exception:
            arch_p_value = np.nan  # If GARCH fails, assign NaN

        # Store results as a dictionary
        results.append({
            'Ticker': ticker,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Sector': sector,
            'Adjusted_ADF_p_value': adj_adf_p_value,
            'ARCH_p_value': arch_p_value,
            'ACF_values': list(acf_values),
            'PACF_values': list(pacf_values)
        })

    # Step 2: Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Step 3: Generate aggregated statistics by Sector
    if not results_df.empty:
        aggregated_results_df = (
            results_df
            .groupby('Sector')
            .agg(
                Total_Stocks_Analyzed=('Ticker', 'count'),
                Percentage_Non_Random_Walk=('Adjusted_ADF_p_value', lambda x: (x < 0.05).mean() * 100),
                Percentage_Significant_ARCH=('ARCH_p_value', lambda x: (x < 0.05).mean() * 100),
                Average_Adjusted_ADF_p_value=('Adjusted_ADF_p_value', 'mean'),
                Average_ARCH_p_value=('ARCH_p_value', 'mean')
            )
            .reset_index()
        )

        # Add Risk and Predictability Descriptions based on aggregated statistics
        aggregated_results_df['Risk_Description'] = aggregated_results_df['Percentage_Significant_ARCH'].apply(
            lambda x: "High risk for building a predictive model" if x > 50 else "Moderate risk for building a predictive model"
        )
        aggregated_results_df['Predictability_Description'] = aggregated_results_df['Percentage_Non_Random_Walk'].apply(
            lambda x: "Stocks are generally predictable" if x > 50 else "Stocks are mostly following a random walk"
        )

        # Add date range columns
        aggregated_results_df['start_date'], aggregated_results_df['end_date'] = start_date, end_date
    else:
        # Handle case where no stocks are analyzed
        aggregated_results_df = pd.DataFrame([{
            'Sector': 'NA',
            'Total_Stocks_Analyzed': 0,
            'Percentage_Non_Random_Walk': 0.0,
            'Percentage_Significant_ARCH': 0.0,
            'Average_Adjusted_ADF_p_value': None,
            'Average_ARCH_p_value': None,
            'Risk_Description': 'No stocks analyzed',
            'Predictability_Description': 'No stocks analyzed',
            'start_date': start_date,
            'end_date': end_date
        }])

    return results_df, aggregated_results_df

# Run analysis for a specific time range and print sector-specific aggregated statistics
results_df, aggregated_results_df = stock_analysis(stock_df, start_date=20230701, end_date=20231025)
print(aggregated_results_df)

######by market segment bear vs bow and middle#######
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model

def stock_analysis_by_segment(stocks_data):

    stocks = stocks_data.copy()
    segments = stocks['Market Segment'].unique()
    results = []

    # Iterate over each market segment and analyze stocks within that segment
    for segment in segments:
        segment_data = stocks[stocks['Market Segment'] == segment]
        tickers = segment_data['Ticker'].unique()

        for ticker in tickers:
            stock_data = segment_data[segment_data['Ticker'] == ticker].sort_values(by='Date')

            if stock_data.shape[0] < 10:  # Skip stocks with insufficient data points
                continue

            # Log returns for stationarity testing
            stock_data['Log_Return'] = np.log(stock_data['Close']).diff()

            # Remove NaN and infinite values from log returns
            stock_data = stock_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_Return'])

            # Skip if stock_data is constant after transformation
            if stock_data['Log_Return'].nunique() <= 1:
                continue

            # Augmented Dickey-Fuller Test for Random Walk with adjustment
            try:
                # ADF test on stock log returns
                adf_result_stock = adfuller(stock_data['Log_Return'])
                adf_p_value_stock = adf_result_stock[1]

                # Generate white noise series and ADF test on it
                white_noise = np.random.normal(0, 1, len(stock_data['Log_Return']))
                adf_result_noise = adfuller(white_noise)
                adf_p_value_noise = adf_result_noise[1]

                # Adjusted p-value
                adj_adf_p_value = min(0.05 * adf_p_value_stock / (adf_p_value_noise + 0.0000001), 0.99)
            except ValueError:
                continue  # Skip stocks that fail ADF test due to constant series

            # Calculate dynamic nlags based on the sample size
            dynamic_nlags = max(1, min(10, stock_data['Log_Return'].shape[0] // 2))

            # ACF and PACF Values with dynamic nlags
            acf_values = acf(stock_data['Log_Return'], nlags=dynamic_nlags)
            pacf_values = pacf(stock_data['Log_Return'], nlags=dynamic_nlags)

            # ARCH Test using GARCH(1,1) Model
            try:
                arch_model_instance = arch_model(stock_data['Log_Return'], vol='Garch', p=1, q=1)
                arch_fit = arch_model_instance.fit(disp='off')
                arch_lm_test_result = arch_fit.arch_lm_test()
                arch_p_value = arch_lm_test_result.pval  # Extract the p-value
            except Exception:
                arch_p_value = np.nan  # If GARCH fails, assign NaN

            # Store results as a dictionary
            results.append({
                'Ticker': ticker,
                'Market_Segment': segment,
                'Adjusted_ADF_p_value': adj_adf_p_value,
                'ARCH_p_value': arch_p_value,
                'ACF_values': list(acf_values),
                'PACF_values': list(pacf_values)
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Generate aggregated statistics by Market Segment
    if not results_df.empty:
        aggregated_results_df = (
            results_df
            .groupby('Market_Segment')
            .agg(
                Total_Stocks_Analyzed=('Ticker', 'count'),
                Percentage_Non_Random_Walk=('Adjusted_ADF_p_value', lambda x: (x < 0.05).mean() * 100),
                Percentage_Significant_ARCH=('ARCH_p_value', lambda x: (x < 0.05).mean() * 100),
                Average_Adjusted_ADF_p_value=('Adjusted_ADF_p_value', 'mean'),
                Average_ARCH_p_value=('ARCH_p_value', 'mean')
            )
            .reset_index()
        )

        # Add Risk and Predictability Descriptions based on aggregated statistics
        aggregated_results_df['Risk_Description'] = aggregated_results_df['Percentage_Significant_ARCH'].apply(
            lambda x: "High risk for building a predictive model" if x > 50 else "Moderate risk for building a predictive model"
        )
        aggregated_results_df['Predictability_Description'] = aggregated_results_df['Percentage_Non_Random_Walk'].apply(
            lambda x: "Stocks are generally predictable" if x > 50 else "Stocks are mostly following a random walk"
        )
    else:
        # Handle case where no stocks are analyzed
        aggregated_results_df = pd.DataFrame([{
            'Market_Segment': 'NA',
            'Total_Stocks_Analyzed': 0,
            'Percentage_Non_Random_Walk': 0.0,
            'Percentage_Significant_ARCH': 0.0,
            'Average_Adjusted_ADF_p_value': None,
            'Average_ARCH_p_value': None,
            'Risk_Description': 'No stocks analyzed',
            'Predictability_Description': 'No stocks analyzed'
        }])

    return results_df, aggregated_results_df

results_df, aggregated_results_df = stock_analysis_by_segment(stock_df)
print(aggregated_results_df)

####bt asset type#######
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model

def stock_analysis(stocks_data, start_date, end_date):

    stocks = stocks_data.copy()

    # Filter data based on the provided date range
    filtered_df = stocks[(stocks['yyyymmdd'] >= start_date) & (stocks['yyyymmdd'] <= end_date)]
    tickers = filtered_df['Ticker'].unique()
    results = []

    # Iterate over each stock and perform the tests
    for ticker in tickers:
        stock_data = filtered_df[filtered_df['Ticker'] == ticker].sort_values(by='Date')

        if stock_data.shape[0] < 10:  # Skip stocks with insufficient data points
            continue

        # Extract the asset size type for the current ticker
        asset_size_type = stock_data['Asset Size Type'].iloc[0] if 'Asset Size Type' in stock_data.columns else "NA"

        # Log returns for stationarity testing
        stock_data['Log_Return'] = np.log(stock_data['Close']).diff()

        # Remove NaN and infinite values from log returns
        stock_data = stock_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_Return'])

        # Skip if stock_data is constant after transformation
        if stock_data['Log_Return'].nunique() <= 1:
            continue

        # Augmented Dickey-Fuller Test for Random Walk with adjustment
        try:
            # ADF test on stock log returns
            adf_result_stock = adfuller(stock_data['Log_Return'])
            adf_p_value_stock = adf_result_stock[1]

            # Generate white noise series and ADF test on it
            white_noise = np.random.normal(0, 1, len(stock_data['Log_Return']))
            adf_result_noise = adfuller(white_noise)
            adf_p_value_noise = adf_result_noise[1]

            # Adjusted p-value
            adj_adf_p_value = min(0.05 * adf_p_value_stock / (adf_p_value_noise + 1e-9), 0.99)
        except ValueError:
            continue  # Skip stocks that fail ADF test due to constant series

        # Calculate dynamic nlags based on the sample size
        dynamic_nlags = max(1, min(10, stock_data['Log_Return'].shape[0] // 2))

        # ACF and PACF Values with dynamic nlags
        acf_values = acf(stock_data['Log_Return'], nlags=dynamic_nlags)
        pacf_values = pacf(stock_data['Log_Return'], nlags=dynamic_nlags)

        # ARCH Test using GARCH(1,1) Model
        try:
            arch_model_instance = arch_model(stock_data['Log_Return'], vol='Garch', p=1, q=1)
            arch_fit = arch_model_instance.fit(disp='off')
            arch_lm_test_result = arch_fit.arch_lm_test()
            arch_p_value = arch_lm_test_result.pval  # Extract the p-value
        except Exception:
            arch_p_value = np.nan  # If GARCH fails, assign NaN

        # Store results as a dictionary
        results.append({
            'Ticker': ticker,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Asset_Size_Type': asset_size_type,
            'Adjusted_ADF_p_value': adj_adf_p_value,
            'ARCH_p_value': arch_p_value,
            'ACF_values': list(acf_values),
            'PACF_values': list(pacf_values)
        })

    # Step 2: Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Step 3: Generate aggregated statistics by Asset Size Type
    if not results_df.empty:
        aggregated_results_df = (
            results_df
            .groupby('Asset_Size_Type')
            .agg(
                Total_Stocks_Analyzed=('Ticker', 'count'),
                Percentage_Non_Random_Walk=('Adjusted_ADF_p_value', lambda x: (x < 0.05).mean() * 100),
                Percentage_Significant_ARCH=('ARCH_p_value', lambda x: (x < 0.05).mean() * 100),
                Average_Adjusted_ADF_p_value=('Adjusted_ADF_p_value', 'mean'),
                Average_ARCH_p_value=('ARCH_p_value', 'mean')
            )
            .reset_index()
        )

        # Add Risk and Predictability Descriptions based on aggregated statistics
        aggregated_results_df['Risk_Description'] = aggregated_results_df['Percentage_Significant_ARCH'].apply(
            lambda x: "High risk for building a predictive model" if x > 50 else "Moderate risk for building a predictive model"
        )
        aggregated_results_df['Predictability_Description'] = aggregated_results_df['Percentage_Non_Random_Walk'].apply(
            lambda x: "Stocks are generally predictable" if x > 50 else "Stocks are mostly following a random walk"
        )

        # Add date range columns
        aggregated_results_df['start_date'], aggregated_results_df['end_date'] = start_date, end_date
    else:
        # Handle case where no stocks are analyzed
        aggregated_results_df = pd.DataFrame([{
            'Asset_Size_Type': 'NA',
            'Total_Stocks_Analyzed': 0,
            'Percentage_Non_Random_Walk': 0.0,
            'Percentage_Significant_ARCH': 0.0,
            'Average_Adjusted_ADF_p_value': None,
            'Average_ARCH_p_value': None,
            'Risk_Description': 'No stocks analyzed',
            'Predictability_Description': 'No stocks analyzed',
            'start_date': start_date,
            'end_date': end_date
        }])

    return results_df, aggregated_results_df


# Run analysis for a specific time range and print asset size-specific aggregated statistics
results_df, aggregated_results_df = stock_analysis(stock_df, start_date=20221001, end_date=20231231)
print(aggregated_results_df)


##by ETF indicator######
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model

def stock_analysis(stocks_data, start_date, end_date):

    stocks = stocks_data.copy()

    # Filter data based on the provided date range
    filtered_df = stocks[(stocks['yyyymmdd'] >= start_date) & (stocks['yyyymmdd'] <= end_date)]
    tickers = filtered_df['Ticker'].unique()
    results = []

    # Iterate over each stock and perform the tests
    for ticker in tickers:
        stock_data = filtered_df[filtered_df['Ticker'] == ticker].sort_values(by='Date')

        if stock_data.shape[0] < 10:  # Skip stocks with insufficient data points
            continue

        # Extract the 'Is ETF' classification for the current ticker
        is_etf = stock_data['Is ETF'].iloc[0] if 'Is ETF' in stock_data.columns else "NA"

        # Log returns for stationarity testing
        stock_data['Log_Return'] = np.log(stock_data['Close']).diff()

        # Remove NaN and infinite values from log returns
        stock_data = stock_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_Return'])

        # Skip if stock_data is constant after transformation
        if stock_data['Log_Return'].nunique() <= 1:
            continue

        # Augmented Dickey-Fuller Test for Random Walk with adjustment
        try:
            # ADF test on stock log returns
            adf_result_stock = adfuller(stock_data['Log_Return'])
            adf_p_value_stock = adf_result_stock[1]

            # Generate white noise series and ADF test on it
            white_noise = np.random.normal(0, 1, len(stock_data['Log_Return']))
            adf_result_noise = adfuller(white_noise)
            adf_p_value_noise = adf_result_noise[1]

            # Adjusted p-value
            adj_adf_p_value = min(0.05 * adf_p_value_stock / (adf_p_value_noise + 1e-7), 0.99)
        except ValueError:
            continue  # Skip stocks that fail ADF test due to constant series

        # Calculate dynamic nlags based on the sample size
        dynamic_nlags = max(1, min(10, stock_data['Log_Return'].shape[0] // 2))

        # ACF and PACF Values with dynamic nlags
        acf_values = acf(stock_data['Log_Return'], nlags=dynamic_nlags)
        pacf_values = pacf(stock_data['Log_Return'], nlags=dynamic_nlags)

        # ARCH Test using GARCH(1,1) Model
        try:
            arch_model_instance = arch_model(stock_data['Log_Return'], vol='Garch', p=1, q=1)
            arch_fit = arch_model_instance.fit(disp='off')
            arch_lm_test_result = arch_fit.arch_lm_test()
            arch_p_value = arch_lm_test_result.pval  # Extract the p-value
        except Exception:
            arch_p_value = np.nan  # If GARCH fails, assign NaN

        # Store results as a dictionary
        results.append({
            'Ticker': ticker,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Is_ETF': is_etf,
            'Adjusted_ADF_p_value': adj_adf_p_value,
            'ARCH_p_value': arch_p_value,
            'ACF_values': list(acf_values),
            'PACF_values': list(pacf_values)
        })

    # Step 2: Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Step 3: Generate aggregated statistics by 'Is ETF' classification
    if not results_df.empty:
        aggregated_results_df = (
            results_df
            .groupby('Is_ETF')
            .agg(
                Total_Stocks_Analyzed=('Ticker', 'count'),
                Percentage_Non_Random_Walk=('Adjusted_ADF_p_value', lambda x: (x < 0.05).mean() * 100),
                Percentage_Significant_ARCH=('ARCH_p_value', lambda x: (x < 0.05).mean() * 100),
                Average_Adjusted_ADF_p_value=('Adjusted_ADF_p_value', 'mean'),
                Average_ARCH_p_value=('ARCH_p_value', 'mean')
            )
            .reset_index()
        )

        # Add Risk and Predictability Descriptions based on aggregated statistics
        aggregated_results_df['Risk_Description'] = aggregated_results_df['Percentage_Significant_ARCH'].apply(
            lambda x: "High risk for building a predictive model" if x > 50 else "Moderate risk for building a predictive model"
        )
        aggregated_results_df['Predictability_Description'] = aggregated_results_df['Percentage_Non_Random_Walk'].apply(
            lambda x: "Stocks are generally predictable" if x > 50 else "Stocks are mostly following a random walk"
        )

        # Add date range columns
        aggregated_results_df['start_date'], aggregated_results_df['end_date'] = start_date, end_date
    else:
        # Handle case where no stocks are analyzed
        aggregated_results_df = pd.DataFrame([{
            'Is_ETF': 'NA',
            'Total_Stocks_Analyzed': 0,
            'Percentage_Non_Random_Walk': 0.0,
            'Percentage_Significant_ARCH': 0.0,
            'Average_Adjusted_ADF_p_value': None,
            'Average_ARCH_p_value': None,
            'Risk_Description': 'No stocks analyzed',
            'Predictability_Description': 'No stocks analyzed',
            'start_date': start_date,
            'end_date': end_date
        }])

    return results_df, aggregated_results_df

# Run analysis for a specific time range and print 'Is ETF'-specific aggregated statistics
results_df, aggregated_results_df = stock_analysis(stock_df, start_date=20230201, end_date=20230501)
print(aggregated_results_df)

