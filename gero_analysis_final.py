import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # for the logistic function
import seaborn as sns
import pandas as pd
from io import BytesIO
from PIL import Image
import base64
import io

# Parameters
default_total_older_adults = 55000000 # Supply the number
default_list_price = 618
default_discounts = {'PBMs': 0.55 * default_list_price}
default_penetration_rates = [0.25, 0.5, 0.75]
default_patent_duration = 15
default_annual_inflation = 0.025
default_start_year = 2030
default_end_year = default_start_year + default_patent_duration
default_peak_time = 5
default_slow_growth_duration = 5

extended_years = list(range(default_start_year, default_end_year + 1))
market_share = {'PBMs': 1}  # Assuming 100% market share for PBMs

def plot_to_html(fig, plot_size):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=1200)  # Increase DPI for better resolution
    buf.seek(0)
    img = Image.open(buf)
    img_data = io.BytesIO()
    img.save(img_data, format='PNG')
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8').replace('\n', '')
    img_tag = f'<img src="data:image/png;base64,{img_base64}" width="{int(plot_size*100)}"/>'
    return img_tag

def adoption_multiplier(t, max_rate, peak_time, slowed_growth_duration):
    if not (0 < max_rate < 1):
        raise ValueError("max_rate must be between 0 and 1")
    s = np.log(max_rate / (1 - max_rate))
    t_50 = default_start_year + default_patent_duration / 2
    k = s / (default_peak_time + slowed_growth_duration - 2 * t_50)
    return expit(k * (t - t_50))

def calculate_spending(start_year, year, condition, peak_time, slow_growth_duration, reduction_percentages, total_older_adults, discounts, annual_inflation):
    # Get the reduction percentage for the given condition
    reduction_percentage = reduction_percentages[condition]

    # Calculate the adoption multiplier for the given year and reduction percentage
    adopt_mult = adoption_multiplier(year, reduction_percentage, peak_time, slow_growth_duration)

    # Calculate the effective spending based on market shares and discounts
    effective_spending = (market_share['PBMs'] * discounts['PBMs'])

    # Adjust for inflation
    inflation_adjustment = (1 + annual_inflation) ** (year - start_year)
    effective_spending *= inflation_adjustment

    # Calculate the spending
    spending = total_older_adults * reduction_percentage * adopt_mult * effective_spending

    return spending

def calculate_consumer_revenue(start_year, year, max_rate, peak_time, slow_growth_duration, total_older_adults, list_price, annual_inflation):
    # Calculate the adoption multiplier for the given year
    adopt_mult = adoption_multiplier(year, max_rate, peak_time, slow_growth_duration)

    # Calculate the consumer revenue based on list price and adoption multiplier
    consumer_revenue = total_older_adults * adopt_mult * list_price

    # Adjust for inflation
    inflation_adjustment = (1 + annual_inflation) ** (year - start_year)
    consumer_revenue *= inflation_adjustment

    return consumer_revenue

def main():
    st.title('Aging Drug Analysis')

    # Create two columns: one for sliders, one for plots
    sliders_col, plots_col = st.columns(2)

    # List of diseases
    list_of_diseases = ["Alzheimer's", "Arthritis", "AFib", "Cancer", "CKD", "COPD", "Diabetes", "HF", "Hyperlipidemia", "Hypertension", "IHD", "Osteoporosis", "Stroke"]

    # User input for parameters in the sliders column
    with sliders_col:
        plot_size = st.slider('Plot Size', min_value=0.1, max_value=15.0, value=5.0, key='plot_size')
        plot_total = st.checkbox('Plot Total Spending', value=True, key='plot_total')
        diseases = st.multiselect('Diseases', options=list_of_diseases, default=[list_of_diseases[0]], key='diseases')
        reduction_rate = st.number_input('Reduction Rate', min_value=0.0, max_value=1.0, value=0.2, key='reduction_rate')
        total_older_adults = st.number_input('Total Older Adults', min_value=0, value=55000000, key='total_older_adults')
        list_price = st.number_input('List Price', min_value=0, value=618, key='list_price')
        discount = st.number_input('Discount for PBMs', min_value=0.0, max_value=1.0, value=0.55, key='discount')
        discounts = {'PBMs': discount * list_price}
        penetration_rates = st.multiselect('Penetration Rates', options=[0.25, 0.5, 0.75], default=[0.25, 0.5, 0.75], key='penetration_rates')
        patent_duration = st.slider('Patent Duration', min_value=0, max_value=30, value=15, key='patent_duration')
        annual_inflation = st.slider('Annual Inflation', min_value=0.0, max_value=0.1, value=0.025, key='annual_inflation')
        start_year = st.slider('Start Year', min_value=2000, max_value=2050, value=2030, key='start_year')
        peak_time = st.slider('Peak Time', min_value=0, max_value=10, value=5, key='peak_time')
        slow_growth_duration = st.slider('Slow Growth Duration', min_value=0, max_value=10, value=5, key='slow_growth_duration')

    # Define extended_years here, after the sliders
    end_year = start_year + patent_duration
    extended_years = list(range(start_year, end_year + 1))

    # Adjusted parameters
    adjusted_peak_time = peak_time - 2
    adjusted_slow_growth_duration = slow_growth_duration + 2

    # Load the data
    data_path = '/Users/alexkesin/adjusted_spending_data.csv'
    spending_data = pd.read_csv(data_path)

    # Calculate the mean of each column (condition)
    # Select only numeric columns and calculate the mean of each column (condition)
    reduction_percentages = spending_data.select_dtypes(include=[np.number]).mean()

    # Normalize the reduction percentages to be between 0 and 1, and add a small constant
    epsilon = 1e-6  # small constant
    max_value = max(reduction_percentages)
    reduction_percentages = {k: epsilon + (v / max_value) * (1 - 2*epsilon) for k, v in reduction_percentages.items()}

    # Calculate the revenues, spendings, and combined_revenue here, after the sliders
    revenues = {(rate, condition): [calculate_spending(start_year, year, condition, adjusted_peak_time, adjusted_slow_growth_duration, reduction_percentages, total_older_adults, discounts, annual_inflation) 
                    for year in extended_years] 
                for rate in penetration_rates for condition in list_of_diseases}

    spendings = {(rate, condition): [calculate_spending(start_year, year, condition, adjusted_peak_time, adjusted_slow_growth_duration, reduction_percentages, total_older_adults, discounts, annual_inflation) 
                                    for year in extended_years] 
                for rate in penetration_rates for condition in list_of_diseases}

    combined_revenue = {(rate, condition): [(revenues[(rate, condition)][i] + spendings[(rate, condition)][i]) 
                                            for i in range(len(extended_years))] 
                        for rate in penetration_rates for condition in list_of_diseases}

    # Calculate total revenues, total spendings, and total combined_revenue for all diseases
    total_revenues = [sum(revenues[(rate, condition)][i] for condition in list_of_diseases for rate in penetration_rates) for i in range(len(extended_years))]
    total_spendings = [sum(spendings[(rate, condition)][i] for condition in list_of_diseases for rate in penetration_rates) for i in range(len(extended_years))]
    total_combined_revenue = [(total_revenues[i] + total_spendings[i]) / 2 for i in range(len(extended_years))]

    consumer_revenues = {rate: [calculate_consumer_revenue(start_year, year, rate, peak_time, slow_growth_duration, total_older_adults, list_price, annual_inflation) 
                            for year in extended_years] 
                        for rate in penetration_rates}

    # Create a color map
    color_map = {disease: color for disease, color in zip(list_of_diseases, sns.color_palette("hls", len(list_of_diseases)))}

    # Set the style of the plots
    sns.set_theme()

    # Plot the final consumer revenue
    fig, ax = plt.subplots(figsize=(10, 7.5))  # Use a fixed size for the plot
    final_consumer_revenue = consumer_revenues[penetration_rates[-1]]  # Get the final consumer revenue
    rev_in_billion = [r / 1e9 for r in final_consumer_revenue]  # Convert revenue to billions
    ax.plot(extended_years, rev_in_billion, label=f'Penetration Rate: {penetration_rates[-1]*100:.0f}%')
    ax.set_title('Consumer Revenue Over Time for An Aging Drug', fontsize=20)
    ax.set_xlabel('Year', fontsize=15)
    ax.set_ylabel('Revenue ($ billions)', fontsize=15)  # Adjust y-axis label
    ax.set_yscale('log')  # Make y-axis logarithmic
    ax.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax.grid(True)

    fig.tight_layout()  # Adjust the layout of the plot
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=1200) # Increase DPI for better resolution
    buf.seek(0)
    plots_col.markdown(plot_to_html(fig, plot_size), unsafe_allow_html=True)

    # Plot the Medicare spending
    fig, ax = plt.subplots(figsize=(10, 7.5))  # Use a fixed size for the plot
    for (rate, condition), spend in spendings.items():
        if condition in diseases:
            spend_in_billion = [s / 1e9 for s in spend]  # Convert spending to billions
            ax.plot(extended_years, spend_in_billion, label=f'Condition: {condition}, Reduction Rate: {rate*100:.0f}%', color=color_map[condition])

    ax.set_title('Potential Medicare Spending Over Time for An Aging Drug', fontsize=20)
    ax.set_xlabel('Year', fontsize=15)
    ax.set_ylabel('Spending ($ billions)', fontsize=15)  # Adjust y-axis label
    ax.set_yscale('log')  # Make y-axis logarithmic
    ax.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax.grid(True)

    fig.tight_layout()  # Adjust the layout of the plot
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=1200)  # Increase DPI for better resolution
    buf.seek(0)
    plots_col.markdown(plot_to_html(fig, plot_size), unsafe_allow_html=True)

    # Plot the combined revenue
    fig, ax = plt.subplots(figsize=(10, 7.5))  # Use a fixed size for the plot
    final_combined_revenue = combined_revenue[(penetration_rates[-1], diseases[-1])]  # Get the final combined revenue
    rev_in_billion = [r / 1e9 for r in final_combined_revenue]  # Convert revenue to billions
    ax.plot(extended_years, rev_in_billion, label=f'Condition: {diseases[-1]}, Penetration Rate: {penetration_rates[-1]*100:.0f}%', color=color_map[diseases[-1]])

    ax.set_title('Combined Revenue Over Time for An Aging Drug', fontsize=20)
    ax.set_xlabel('Year', fontsize=15)
    ax.set_ylabel('Revenue ($ billions)', fontsize=15)  # Adjust y-axis label
    ax.set_yscale('log')  # Make y-axis logarithmic
    ax.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax.grid(True)

    fig.tight_layout()  # Adjust the layout of the plot
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=1200)  # Increase DPI for better resolution
    buf.seek(0)
    plots_col.markdown(plot_to_html(fig, plot_size), unsafe_allow_html=True)

if __name__ == "__main__":
    main()