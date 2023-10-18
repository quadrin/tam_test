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
    fig.savefig(buf, format="PNG", dpi=300)  # Increase DPI for better resolution
    buf.seek(0)
    img = Image.open(buf)
    img_data = io.BytesIO()
    img.save(img_data, format='PNG')
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8').replace('\n', '')
    img_tag = f'<img src="data:image/png;base64,{img_base64}" width="{int(plot_size*100)}"/>'
    return img_tag

def reduction_to_penetration(reduction):
    # Parameters of the sigmoid function
    max_penetration = 0.9
    steepness = 10

    # Sigmoid function
    penetration = max_penetration / (1 + np.exp(-steepness * (reduction - 0.5)))

    return penetration

def adoption_multiplier(t, max_rate, peak_time, slowed_growth_duration):
    if not (0 < max_rate < 1):
        raise ValueError("max_rate must be between 0 and 1")
    s = np.log(max_rate / (1 - max_rate))
    t_50 = default_start_year + default_patent_duration / 2
    k = s / (default_peak_time + slowed_growth_duration - 2 * t_50)
    adopt_mult = expit(k * (t - t_50))

    return max(0, adopt_mult)  # Ensure that the adoption multiplier is not less than 0

def calculate_spending(start_year, year, state, condition, adjusted_peak_time, adjusted_slow_growth_duration, reduction_rate, total_older_adults, discounts, annual_inflation, spending_data, total_spending_per_condition, total_spending_all_conditions):     
    # Get the spending data for the given state and condition
    original_spending = total_spending_per_condition[condition]
    print(f"Original spending for {condition}: {original_spending}")

    reduction_rate = 1 - reduction_rate

    # Convert reduction_rate to penetration_rate
    penetration_rate = reduction_to_penetration(reduction_rate)
    print(f"Penetration rate for {condition}: {penetration_rate}")

    # Calculate the adoption multiplier for the given year
    adopt_mult = adoption_multiplier(year, penetration_rate, adjusted_peak_time, adjusted_slow_growth_duration)
    print(f"Adoption multiplier for {condition} in year {year}: {adopt_mult}")

    # Calculate the new spending based on the spending data, adoption multiplier, and penetration rate
    new_spending = original_spending * adopt_mult 
    print(f"New spending for {condition} in year {year}: {new_spending}")

    # Adjust for inflation
    inflation_adjustment = (1 + annual_inflation) ** (year - start_year)
    new_spending *= inflation_adjustment
    print(f"New spending for {condition} in year {year} after inflation adjustment: {new_spending}")

    original_spending *= inflation_adjustment

    # Calculate the difference between the original spending and the new spending
    spending_difference = original_spending - new_spending
    print(f"Spending difference for {condition} in year {year}: {spending_difference}")

    return spending_difference

def calculate_consumer_revenue(start_year, year, max_rate, peak_time, slow_growth_duration, total_older_adults, list_price, annual_inflation):
    # Calculate the adoption multiplier for the given year
    adopt_mult = adoption_multiplier(year, max_rate, peak_time, slow_growth_duration)

    # Calculate the consumer revenue based on list price, adoption multiplier, and consumer penetration rate
    consumer_revenue = total_older_adults * adopt_mult * list_price * max_rate

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
        reduction_rate = st.number_input('Spending Reduction Rate (for payers)', min_value=0.0, max_value=1.0, value=0.2, key='reduction_rate')
        consumer_percentage = st.slider('Consumer Revenue as Percentage of Total Revenue', min_value=0.0, max_value=1.0, value=0.5, key='percentage')
        spending_percentage = 1 - consumer_percentage
        list_price = st.number_input('Consumer Aging Drug List Price ($)', min_value=0, value=618, key='list_price')
        penetration_rate = st.slider('Consumer Penetration Rate at Peak Adoption', min_value=0.0, max_value=1.0, value=0.25, key='penetration_rate')
        total_older_adults = st.number_input('Total Older Adults', min_value=0, value=55000000, key='total_older_adults')
        discount = st.number_input('Discount for Payers (Medicare/PBMs)', min_value=0.0, max_value=1.0, value=0.55, key='discount')
        discounts = {'PBMs': discount}
        patent_duration = st.slider('Years of Patent Duration', min_value=0, max_value=30, value=15, key='patent_duration')
        annual_inflation = st.slider('Annual Inflation Rate', min_value=0.0, max_value=0.1, value=0.025, key='annual_inflation')
        start_year = st.slider('Start Year', min_value=2023, max_value=2050, value=2030, key='start_year')
        peak_time = st.slider('Years Until Peak Revenue', min_value=0, max_value=10, value=5, key='peak_time')
        slow_growth_duration = st.slider('Years under Slow Adoption Period', min_value=0, max_value=10, value=5, key='slow_growth_duration')

    # Define extended_years here, after the sliders
    end_year = start_year + patent_duration
    extended_years = list(range(start_year, end_year + 1))

    # Adjusted parameters
    adjusted_peak_time = peak_time - 2
    adjusted_slow_growth_duration = slow_growth_duration + 2

    # Load the data
    data_path = '/Users/alexkesin/adjusted_spending_data.csv'
    
    spending_data = pd.read_csv(data_path)

    state_rows = spending_data['State'].notna() & ~spending_data['State'].str.contains("Total|Average|spending", na=False, case=False)
   
    suspicious_rows = spending_data[~state_rows]
    print(suspicious_rows)
   
    spending_data = spending_data[state_rows]

    # Calculate total spending per condition
    total_spending_per_condition = spending_data.sum(numeric_only=True)

    # Calculate total spending for all conditions
    total_spending_all_conditions = total_spending_per_condition.sum()

    # Calculate the mean of each column (condition)
    # Select only numeric columns and calculate the mean of each column (condition)
    # Calculate the mean of each column (condition)
    reduction_percentages = spending_data.select_dtypes(include=[np.number]).mean()

    # Normalize the reduction percentages to be between 0 and 1, and add a small constant
    epsilon = 1e-6  # small constant
    max_value = max(reduction_percentages)
    reduction_percentages = {k: epsilon + (v / max_value) * (1 - 2*epsilon) for k, v in reduction_percentages.items()}

    # Calculate the final consumer revenue using the selected consumer penetration rate
    final_consumer_revenue = [calculate_consumer_revenue(start_year, year, penetration_rate, peak_time, slow_growth_duration, total_older_adults, list_price, annual_inflation) for year in extended_years]

    # Calculate total revenues, total spendings, and total combined_revenue for all diseases
    total_spendings = [sum(calculate_spending(start_year, year, state, condition, adjusted_peak_time, adjusted_slow_growth_duration, reduction_rate, total_older_adults, discounts, annual_inflation, spending_data, total_spending_per_condition, total_spending_all_conditions) 
                            for state in spending_data['State'] 
                            for condition in diseases) 
                            for year in extended_years]
    total_combined_revenue = [final_consumer_revenue[i] * consumer_percentage + total_spendings[i] * (1 - consumer_percentage) for i in range(len(extended_years))]
   
    # Create a color map
    color_map = {disease: color for disease, color in zip(list_of_diseases, sns.color_palette("hls", len(list_of_diseases)))}

    # Set the style of the plots
    sns.set_theme()

    # Plot the final consumer revenue
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5))  # Use a fixed size for the plot
    rev_in_billion = [r / 1e9 for r in final_consumer_revenue]  # Convert revenue to billions
    ax1.plot(extended_years, rev_in_billion, color='blue')  # Removed label argument
    ax2.plot(extended_years, rev_in_billion, color='blue')  # Removed label argument

    ax1.set_title('Consumer Revenue Over Time for An Aging Drug (Log Scale)', fontsize=20)
    ax1.set_xlabel('Year', fontsize=15)
    ax1.set_ylabel('Revenue ($ billions)', fontsize=15)  # Adjust y-axis label
    ax1.set_yscale('log')  # Make y-axis logarithmic
    ax1.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax1.grid(True)

    ax2.set_title('Consumer Revenue Over Time for An Aging Drug (Normal Scale)', fontsize=20)
    ax2.set_xlabel('Year', fontsize=15)
    ax2.set_ylabel('Revenue ($ billions)', fontsize=15)  # Adjust y-axis label
    ax2.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax2.grid(True)

    fig.tight_layout()  # Adjust the layout of the plot
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=300) # Increase DPI for better resolution
    buf.seek(0)
    plots_col.markdown(plot_to_html(fig, plot_size), unsafe_allow_html=True)

    # Calculate the Medicare spending
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5))  # Use a fixed size for the plot
    for condition in diseases:
        spendings = []
        for year in extended_years:
            # Calculate the spending for the current year
            spending = total_spending_per_condition[condition]
            spendings.append(spending)
        spend_in_billion = [s / 1e9 for s in spendings]  # Convert spending to billions
        ax1.plot(extended_years, spend_in_billion, label=f'Condition: {condition}')
        if plot_total:
            total_spend_in_billion = [s / 1e9 for s in total_spendings]  # Convert total spending to billions
            ax1.plot(extended_years, total_spend_in_billion, label='Total DiseaseSpending', color='black', linestyle='--')
        ax2.plot(extended_years, spend_in_billion, label=f'Condition: {condition}')

    if plot_total:
        total_spend_in_billion = [s / 1e9 for s in total_spendings]  # Convert total spending to billions
        ax1.plot(extended_years, total_spend_in_billion, label='Total Disease Spending', color='black', linestyle='--')
        ax2.plot(extended_years, total_spend_in_billion, label='Total Disease Spending', color='black', linestyle='--')

    ax1.set_title('Potential Medicare Spending Over Time for An Aging Drug (Log Scale)', fontsize=20)
    ax1.set_xlabel('Year', fontsize=15)
    ax1.set_ylabel('Spending ($ billions)', fontsize=15)  # Adjust y-axis label
    ax1.set_yscale('log')  # Make y-axis logarithmic
    ax1.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax1.grid(True)

    ax2.set_title('Potential Medicare Spending Over Time for An Aging Drug (Normal Scale)', fontsize=20)
    ax2.set_xlabel('Year', fontsize=15)
    ax2.set_ylabel('Spending ($ billions)', fontsize=15)  # Adjust y-axis label
    ax2.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax2.grid(True)

    fig.tight_layout()  # Adjust the layout of the plot
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=300)  # Increase DPI for better resolution
    buf.seek(0)
    plots_col.markdown(plot_to_html(fig, plot_size), unsafe_allow_html=True)

    # Create a combined revenue figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5))  # Use a fixed size for the plot

    # Plot the combined revenue on the first subplot (logarithmic scale)
    ax1.set_title('Total Revenue Over Time for An Aging Drug (Log Scale)', fontsize=20)
    ax1.set_xlabel('Year', fontsize=15)
    ax1.set_ylabel('Revenue ($ billions)', fontsize=15)  # Adjust y-axis label
    ax1.set_yscale('log')  # Make y-axis logarithmic
    ax1.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax1.grid(True)

    if plot_total:
        total_combined_rev_in_billion = [r / 1e9 for r in total_combined_revenue]  # Convert total combined revenue to billions
        ax1.plot(extended_years, total_combined_rev_in_billion, label='Total Combined Revenue', color='black', linestyle='--')
        ax2.plot(extended_years, total_combined_rev_in_billion, label='Total Combined Revenue', color='black', linestyle='--')

    # Plot the combined revenue on the second subplot (normal scale)
    ax2.set_title('Total Revenue Over Time for An Aging Drug (Normal Scale)', fontsize=20)
    ax2.set_xlabel('Year', fontsize=15)
    ax2.set_ylabel('Revenue ($ billions)', fontsize=15)  # Adjust y-axis label
    ax2.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend to the right
    ax2.grid(True)

    fig.tight_layout()  # Adjust the layout of the plot
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=300)  # Increase DPI for better resolution
    buf.seek(0)
    plots_col.markdown(plot_to_html(fig, plot_size), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
