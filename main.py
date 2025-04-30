import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO
from collections import Counter
import re
from replies import get_all_replies_with_sentiment
from report import generate_improvement_report
import io
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
from datetime import datetime
import plotly.io as pio
from PIL import Image

# Access Twitter API key from Streamlit secrets
twitter_api_key = st.secrets["twitter"]["bearer_token"]

# Access Groq API key from Streamlit secrets
groq_api_key = st.secrets["groq"]["api_key"]

def generate_dashboard_report(data, sentiment_counts, category_sentiment_data, positive_ratios=None, 
                             improvement_report=None, selected_companies=None):
    """
    Generate a comprehensive PDF report from dashboard data
    
    Parameters:
    - data: DataFrame containing sentiment analysis data
    - sentiment_counts: Series or dict containing sentiment distribution counts
    - category_sentiment_data: DataFrame with category-wise sentiment distribution
    - positive_ratios: DataFrame with company positive sentiment ratios (optional)
    - improvement_report: String containing AI-generated improvement recommendations (optional)
    - selected_companies: List of selected companies for comparison (optional)
    
    Returns:
    - BytesIO object containing the PDF report
    """
    # Create PDF with A4 dimensions
    class PDF(FPDF):
        def header(self):
            # Add company logo (placeholder)
            self.image('pinelabs_3.png', 10, 8, 30)
            # Add report title
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Sentiment Analysis Dashboard Report', 0, 1, 'C')
            # Add date
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
            # Add line break
            self.ln(5)
        
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Arial italic 8
            self.set_font('Arial', 'I', 8)
            # Add page number
            self.cell(0, 10, f'Page {self.page_no()}/2', 0, 0, 'C')
            
    # Initialize PDF object
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add executive summary section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Calculate key metrics
    total_reviews = len(data)
    if isinstance(sentiment_counts, dict):
        pos_count = sentiment_counts.get('positive', 0)
        neg_count = sentiment_counts.get('negative', 0)
        neu_count = sentiment_counts.get('neutral', 0)
    else:
        pos_count = sentiment_counts.get('positive', 0) if not sentiment_counts.empty else 0
        neg_count = sentiment_counts.get('negative', 0) if not sentiment_counts.empty else 0
        neu_count = sentiment_counts.get('neutral', 0) if not sentiment_counts.empty else 0
        
    pos_percent = (pos_count / total_reviews * 100) if total_reviews > 0 else 0
    neg_percent = (neg_count / total_reviews * 100) if total_reviews > 0 else 0
    neu_percent = (neu_count / total_reviews * 100) if total_reviews > 0 else 0
    
    # Add summary text
    pdf.multi_cell(0, 5, f"This report provides a comprehensive analysis of {total_reviews} reviews across "
                      f"{len(data['source'].unique())} sources. The overall sentiment distribution shows "
                      f"{pos_percent:.1f}% positive, {neg_percent:.1f}% negative, and {neu_percent:.1f}% neutral reviews.", 0)
    pdf.ln(5)
    
    # Create matplotlib figure for sentiment distribution (pie chart)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [pos_count, neu_count, neg_count]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Overall Sentiment Distribution')
    
    # Save pie chart to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Convert BytesIO to a temporary file
    temp_img_path = "temp_logo.png"
    with open(temp_img_path, "wb") as f:
        f.write(buf.getvalue())

    # Use the file path instead of BytesIO
    pdf.image(temp_img_path, x=10, y=70, w=90)

    # Clean up the temporary file
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    # Create category sentiment bar chart if data is available
    if category_sentiment_data is not None and not category_sentiment_data.empty:
        # Prepare data for category chart
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Get top 5 categories
        top_categories = category_sentiment_data.head(5)
        categories = top_categories['category'].tolist()
        
        # Create stacked bar chart
        neg_vals = top_categories['negative_percentage'].tolist() if 'negative_percentage' in top_categories.columns else [0] * len(categories)
        neu_vals = top_categories['neutral_percentage'].tolist() if 'neutral_percentage' in top_categories.columns else [0] * len(categories)
        pos_vals = top_categories['positive_percentage'].tolist() if 'positive_percentage' in top_categories.columns else [0] * len(categories)
        
        ax.barh(categories, pos_vals, color='#4CAF50', label='Positive')
        ax.barh(categories, neu_vals, left=pos_vals, color='#FFC107', label='Neutral')
        ax.barh(categories, neg_vals, left=[p+n for p,n in zip(pos_vals, neu_vals)], color='#F44336', label='Negative')
        
        ax.set_xlabel('Percentage (%)')
        ax.set_title('Sentiment Distribution by Top Categories')
        ax.legend(loc='lower right')
        
        # Save category chart to buffer
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf2.seek(0)
        
        # Add category chart to PDF
        
        temp_img_path = "temp_logo_2.png"
        with open(temp_img_path, "wb") as f:
            f.write(buf2.getvalue())

        # Use the file path instead of BytesIO
        pdf.image(temp_img_path, x=100, y=70, w=90)

        # Clean up the temporary file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
    
    # Add company comparison section if available
    if positive_ratios is not None and not positive_ratios.empty and selected_companies:
        pdf.ln(85)  # Move down after charts
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Company Comparison', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Create company comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Sort companies by positive percentage
        sorted_companies = positive_ratios.sort_values(by='positive_percentage', ascending=False)
        companies = sorted_companies['company'].tolist()
        pos_pcts = sorted_companies['positive_percentage'].tolist()
        
        # Create horizontal bar chart
        bars = ax.barh(companies, pos_pcts, color='#4CAF50')
        
        # Add value labels to bars
        for i, v in enumerate(pos_pcts):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
        
        ax.set_xlabel('Positive Sentiment (%)')
        ax.set_title('Positive Sentiment Comparison by Company')
        
        # Save company chart to buffer
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf3.seek(0)
        
        # Add company chart to PDF
        temp_img_path = "temp_logo_3.png"
        with open(temp_img_path, "wb") as f:
            f.write(buf3.getvalue())

        # Use the file path instead of BytesIO
        pdf.image( temp_img_path, x=10, y=170, w=180)

        # Clean up the temporary file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
    else:
        pdf.ln(85)  # Move down after charts
    
    # Add second page with improvement recommendations
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Key Findings & Recommendations', 0, 1)
    
    # Add key metrics in a table
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'Key Metrics:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Create a simple table for key metrics
    col_width = pdf.w / 4
    pdf.set_fill_color(240, 240, 240)
    
    # Table headers
    pdf.cell(col_width, 10, 'Metric', 1, 0, 'C', 1)
    pdf.cell(col_width, 10, 'Value', 1, 0, 'C', 1)
    pdf.cell(col_width * 2, 10, 'Interpretation', 1, 1, 'C', 1)
    
    # Total reviews
    pdf.cell(col_width, 10, 'Total Reviews', 1, 0, 'L')
    pdf.cell(col_width, 10, f'{total_reviews}', 1, 0, 'C')
    pdf.cell(col_width * 2, 10, 'Sample size for analysis', 1, 1, 'L')
    
    # Positive ratio
    pdf.cell(col_width, 10, 'Positive Ratio', 1, 0, 'L')
    pdf.cell(col_width, 10, f'{pos_percent:.1f}%', 1, 0, 'C')
    status = 'Good' if pos_percent > 60 else 'Average' if pos_percent > 40 else 'Needs improvement'
    pdf.cell(col_width * 2, 10, f'{status}', 1, 1, 'L')
    
    # Positive to Negative ratio
    pos_neg_ratio = pos_count / max(1, neg_count)
    pdf.cell(col_width, 10, 'Pos:Neg Ratio', 1, 0, 'L')
    pdf.cell(col_width, 10, f'{pos_neg_ratio:.2f}', 1, 0, 'C')
    status = 'Good' if pos_neg_ratio > 2 else 'Average' if pos_neg_ratio > 1 else 'Needs improvement'
    pdf.cell(col_width * 2, 10, f'{status} - Target is >2.0', 1, 1, 'L')
    
    # Add AI-generated improvement recommendations
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'Improvement Recommendations:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if improvement_report:
        # Process the HTML from improvement report to plain text
        from html.parser import HTMLParser
        
        class HTMLFilter(HTMLParser):
            text = ""
            def handle_data(self, data):
                self.text += data
        
        html_filter = HTMLFilter()
        html_filter.feed(improvement_report)
        plain_text = html_filter.text
        
        # Add the improvement text
        pdf.multi_cell(0, 5, plain_text, 0)
    else:
        # Generate generic recommendations based on data
        pdf.multi_cell(0, 5, "Based on the sentiment analysis, we recommend focusing on improving the most mentioned negative aspects in customer feedback. Key areas to address include:", 0)
        pdf.ln(2)
        
        # Try to find top negative categories
        top_negative_cats = []
        try:
            if category_sentiment_data is not None and not category_sentiment_data.empty:
                # Find categories with highest negative percentage
                neg_cats = category_sentiment_data.sort_values(by='negative_percentage' if 'negative_percentage' in category_sentiment_data.columns else 'total', ascending=False).head(3)
                top_negative_cats = neg_cats['category'].tolist() if 'category' in neg_cats.columns else []
        except:
            pass
        
        # Add bullet points for recommendations
        if top_negative_cats:
            for i, cat in enumerate(top_negative_cats):
                pdf.cell(10, 5, chr(149), 0, 0)  # bullet character
                pdf.cell(0, 5, f"Improve {cat} by addressing customer pain points and enhancing service quality", 0, 1)
        else:
            # Generic recommendations
            pdf.cell(10, 5, chr(149), 0, 0)  # bullet character
            pdf.cell(0, 5, "Enhance customer service response time and quality of resolution", 0, 1)
            pdf.cell(10, 5, chr(149), 0, 0)
            pdf.cell(0, 5, "Address product quality concerns highlighted in negative feedback", 0, 1)
            pdf.cell(10, 5, chr(149), 0, 0)
            pdf.cell(0, 5, "Improve communication channels for better customer engagement", 0, 1)
    
    # Add conclusion
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'Conclusion:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Generate dynamic conclusion based on sentiment ratio
    conclusion_text = ""
    if pos_percent > 70:
        conclusion_text = f"With {pos_percent:.1f}% positive sentiment, overall customer satisfaction is high. Focus on maintaining strengths while addressing specific improvement areas to further enhance customer experience."
    elif pos_percent > 50:
        conclusion_text = f"With {pos_percent:.1f}% positive sentiment, customer satisfaction is moderate. There are significant opportunities to improve by addressing the negative feedback areas highlighted in this report."
    else:
        conclusion_text = f"With only {pos_percent:.1f}% positive sentiment, there are critical areas requiring immediate attention. We recommend a comprehensive review of the highlighted issues and development of an action plan to address customer concerns."
    
    pdf.multi_cell(0, 5, conclusion_text, 0)
    
    # Add company-specific note if applicable
    if selected_companies and 'PineLabs' in selected_companies:
        pdf.ln(5)
        pdf.multi_cell(0, 5, "Note: For PineLabs specifically, we recommend focusing on the areas where competitor analysis shows significant gaps compared to industry benchmarks.", 0)
    
    # Convert PDF to BytesIO
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    
    return pdf_output


# Function to add the download report button to your Streamlit app
def add_report_download_button(data, sentiment_counts, category_sentiment_data, positive_ratios=None, 
                              improvement_report=None, selected_companies=None):
    """
    Adds a download report button to the Streamlit app that generates a comprehensive PDF report
    
    Parameters:
    - data: DataFrame containing sentiment analysis data
    - sentiment_counts: Series or dict containing sentiment distribution counts
    - category_sentiment_data: DataFrame with category-wise sentiment distribution
    - positive_ratios: DataFrame with company positive sentiment ratios (optional)
    - improvement_report: String containing AI-generated improvement recommendations (optional)
    - selected_companies: List of selected companies for comparison (optional)
    """
    try:
        # Generate the PDF report
        pdf_bytes = generate_dashboard_report(
            data, 
            sentiment_counts, 
            category_sentiment_data, 
            positive_ratios, 
            improvement_report, 
            selected_companies
        )
        
        # Create download button for the report
        st.download_button(
            label="📊 Download Report",
            data=pdf_bytes,
            file_name=f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            help="Download a comprehensive 2-page report with key insights from this dashboard"
        )
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.exception(e)

# Set page configuration
st.set_page_config(
    page_title="Pine Labs Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ✅ Password Protection Page – Pine Labs Style
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

if not st.session_state["password_correct"]:
    # Show Pine Labs logo
    st.markdown("<div style='padding-top: 60px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        try:
            st.image("pinelabs_3.png", width=160)
        except:
            st.markdown("<h2 style='text-align:center;color:#345c49;'>Pine Labs</h2>", unsafe_allow_html=True)

    # Show instruction
    st.markdown("""
    <div style='text-align:center; color:#345c49; font-size: 1.3rem; font-weight: 600; margin-top: 20px;'>
        🔐 Enter Password to Access the Application
    </div>
    """, unsafe_allow_html=True)

    # Password box
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        password = st.text_input("", type="password", placeholder="Enter password", key="password_input")
        login = st.button("ENTER")

    if login:
        if password == st.secrets["auth"]["password"]:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("❌ Incorrect Password. Please try again.")

    st.stop()  # Prevent rest of app from showing

# Custom CSS with improved styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --pine-green: #345c49;
        --pine-light: #f0f5f2;
        --pine-accent: #4a8572;
        --pine-hover: #2a4a3b;
        --positive: #4CAF50;
        --neutral: #FFC107;
        --negative: #F44336;
        --white: #ffffff;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Background and general styling */
    .main {
        background-color: var(--pine-light);
        padding: 1.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--pine-green);
        color: var(--white);
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 6px var(--shadow);
    }
    
    .stButton>button:hover {
        background-color: var(--pine-hover);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px var(--shadow);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Card styling */
    .metrics-card {
        background-color: var(--white);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 6px 12px var(--shadow);
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        transition: all 0.3s ease;
        border-left: 5px solid var(--pine-green);
    }
    
    .metrics-card:hover {
        box-shadow: 0 10px 20px var(--shadow);
        transform: translateY(-5px);
    }
    
    /* Metric values */
    .metric-title {
        color: var(--pine-green);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #333;
        padding: 8px 0;
        position: relative;
        display: inline-block;
    }
    
    .metric-value:after {
        content: '';
        position: absolute;
        width: 40%;
        height: 3px;
        background-color: var(--pine-green);
        bottom: 0;
        left: 30%;
        border-radius: 2px;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: var(--white);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 12px var(--shadow);
        margin-bottom: 24px;
        transition: all 0.3s ease;
        border-top: 5px solid var(--pine-green);
    }
    
    .chart-container:hover {
        box-shadow: 0 10px 20px var(--shadow);
    }
    
    /* Typography */
    h1 {
        color: var(--pine-green);
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        text-shadow: 1px 1px 2px var(--shadow);
    }
    
    h2, h3 {
        color: var(--pine-green);
        margin-bottom: 1rem;
    }
    
    .highlight {
        color: var(--pine-green);
        font-weight: bold;
    }
    
    /* Lists */
    ul {
        padding-left: 20px;
        margin-bottom: 20px;
    }
    
    li {
        margin-bottom: 12px;
        position: relative;
        padding-left: 25px;
    }
    
    
    /* About section */
    .about-section p {
        line-height: 1.8;
        margin-bottom: 20px;
        font-size: 1.05rem;
    }
    
    /* Chart titles */
    .chart-title {
        color: var(--pine-green);
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 15px;
        padding: 0;
        display: flex;
        align-items: center;
    }
    
    .chart-title:before {
        content: '📊';
        margin-right: 10px;
    }
    
    /* Data table */
    .dataframe {
        width: 100% !important;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Custom badge for sentiment */
    .sentiment-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        color: white;
        display: inline-block;
        text-align: center;
        min-width: 100px;
    }
    
    .badge-positive {
        background-color: var(--positive);
    }
    
    .badge-neutral {
        background-color: var(--neutral);
        color: #333;
    }
    
    .badge-negative {
        background-color: var(--negative);
    }
    
    /* Progress indicators */
    .stProgress > div > div > div > div {
        background-color: var(--pine-green);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-radius: 4px 4px 0 0;
        border-top: 4px solid var(--pine-green);
        border-right: 1px solid #ccc;
        border-left: 1px solid #ccc;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        border-radius: 0 0 4px 4px;
        border-right: 1px solid #ccc;
        border-bottom: 1px solid #ccc;
        border-left: 1px solid #ccc;
        padding: 16px;
    }
    
    /* Select box */
    div[data-baseweb="select"] > div {
        border-radius: 8px;
        border-color: var(--pine-green);
    }
    
    /* Number input */
    .stNumberInput [aria-describedby] {
        border-color: var(--pine-green);
        border-radius: 8px;
    }
    
    /* Sidebar refinements */
    .css-6qob1r {
        background-color: var(--pine-light);
    }
    
    /* Tooltip and interactive elements */
    div[data-testid="stTooltipIcon"] {
        color: var(--pine-green);
    }

    /* Company color badges */
    .company-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        color: white;
        display: inline-block;
        text-align: center;
        min-width: 100px;
        margin-right: 10px;
    }
    
    .badge-pinelabs {
        background-color: #345c49;
    }
    
    .badge-razorpay {
        background-color: #3366CC;
    }
    
    .badge-paytm {
        background-color: #0F4A8A;
    }
        
</style>
""", unsafe_allow_html=True)

# Function to encode image to base64
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to create a sentiment badge
def get_sentiment_badge(sentiment):
    if sentiment.lower() == 'positive':
        return f'<div class="sentiment-badge badge-positive">Positive</div>'
    elif sentiment.lower() == 'neutral':
        return f'<div class="sentiment-badge badge-neutral">Neutral</div>'
    elif sentiment.lower() == 'negative':
        return f'<div class="sentiment-badge badge-negative">Negative</div>'
    else:
        return sentiment

# Function to create a company badge
def get_company_badge(company):
    if company.lower() == 'pinelabs':
        return f'<div class="company-badge badge-pinelabs">Pine Labs</div>'
    elif company.lower() == 'razorpay':
        return f'<div class="company-badge badge-razorpay">Razorpay</div>'
    elif company.lower() == 'paytm':
        return f'<div class="company-badge badge-paytm">Paytm</div>'
    else:
        return company

# Function to get company-specific color
def get_company_color(company):
    if company.lower() == 'pinelabs':
        return '#345c49'
    elif company.lower() == 'razorpay':
        return '#3366CC'
    elif company.lower() == 'paytm':
        return '#0F4A8A'
    else:
        return '#999999'  # Default color

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'info'
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'cards'
    if 'selected_company' not in st.session_state:
        st.session_state.selected_company = 'All'
        
    # Information page
    if st.session_state.page == 'info':
        # Display Pine Labs logo with animation
        st.markdown("<div style='padding-top: 5px;'></div>", unsafe_allow_html=True)

        # Center the image with proper spacing using columns
        col1, col2, col3 = st.columns([2, 2, 1])
        with col2:
            try:
                st.image("pinelabs_3.png", width=200)
            except:
                # Fallback to text if image fails to load
                st.markdown("""
                <h1 style='text-align: center; color: #345c49; 
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.1); 
                    animation: fadeIn 1.5s ease-in-out;'>
                    Pine Labs
                </h1>
                <style>
                @keyframes fadeIn {
                    0% { opacity: 0; transform: translateY(-20px); }
                    100% { opacity: 1; transform: translateY(0); }
                }
                </style>
                """, unsafe_allow_html=True)

        # Add some space after the logo
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        
        # App title with animation
        st.markdown("""
        <h1 style='text-align: center; animation: slideIn 1s ease-out;'>
            Sentimeter
        </h1>
        <style>
        @keyframes slideIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Cards with animations
        st.markdown("""
        <div class="metrics-card about-section" style="animation: fadeIn 1.2s ease-in-out;">
            <h2>About this Application</h2>
            <p>Real-time social media sentiment dashboard to monitor brand perception for Pine Labs and key competitors. The dashboard will analyze comments from official channels across platforms like X (Twitter), LinkedIn, YouTube, App Store, and Play Store. In pilot with X right now.</p>
            <h3>Key Features:</h3>
            <ul>
                <li>Comparative analysis between Pine Labs, Razorpay, and Paytm</li>
                <li>Real-time sentiment analysis from multiple sources</li>
                <li>Interactive visualization of sentiment trends over time</li>
                <li>Word cloud analysis to identify common themes in customer feedback</li>
                <li>Comprehensive metrics for data-driven decision making</li>
                <li>Benchmarking capabilities to compare performance against competitors</li>
            </ul>
        </div>
        <style>
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Input section for days with improved styling
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="metrics-card" style="animation: fadeIn 1.6s ease-in-out;">
                <div class="metric-title">Select number of past days for analysis</div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])  # Adjust these numbers to change column widths
            with col2:
                days = st.selectbox("",
                                options=[5,10,20,30], 
                                index=0,
                                key="days_input",
                                help="Select the number of days for which you want to analyze sentiment data")
            
            # Button with loading animation
            col1, col2, col3 = st.columns([1.2, 2, 1])
            with col2:
                if st.button("Perform Sentiment Analysis", key="analyze_button", 
                            help="Click to fetch and analyze sentiment data for the selected days"):
                    with st.spinner("Fetching and analyzing data..."):
                        # Add a progress bar for better UX
                        progress_bar = st.progress(0)
                        for i in range(101):
                            # Update progress bar
                            progress_bar.progress(i)
                            if i == 25:
                                st.markdown("""
                                <div class="loading">
                                    <p>Connecting to data sources...</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif i == 50:
                                st.markdown("""
                                <div class="loading">
                                    <p>Analyzing sentiment patterns...</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif i == 75:
                                st.markdown("""
                                <div class="loading">
                                    <p>Preparing visualization data...</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            import time
                            time.sleep(0.02)  # Simulate processing time
                            
                        try:
                            # Get sentiment data
                            st.session_state.data = get_all_replies_with_sentiment(twitter_api_key,groq_api_key, days)
                            st.session_state.page = 'dashboard'
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error fetching data: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Dashboard page
    elif st.session_state.page == 'dashboard':
        st.info("ℹ️ Need help understanding the metrics and charts? See the **Dashboard Guide** in the sidebar 👈")
        # ✅ Dashboard-only sidebar
        with st.sidebar:
            st.header("📘 Dashboard Guide")
            st.markdown("""
            - **Total Comments**: Number of user comments collected in the selected time frame.
            - **Sentiment Distribution**: Percentages of positive, neutral, and negative comments.
            - **Average Sentiment Score**: A 0–1 score indicating overall positivity.
            - **Positive vs Negative Ratio**: Ratio of positive comments to negative ones (>1 means more positive).
            - **Company Comparison**: Side‑by‑side sentiment metrics for PineLabs, Razorpay, and Paytm.
            - **Sentiment Trend Over Time**: How sentiment shifts day‑by‑day (or week/month) in the period.
            - **Top Reviews**: Highlights of the best and worst PineLabs reviews.
            - **Word Cloud**: Most common words in negative feedback to spot pain‑points.
            - **Day of Week Analysis**: Which weekdays/weekends see better or worse sentiment.
            - **Competitor Trend**: Comparative positive‑sentiment trend lines for all companies.
            """)
            # Create a top navigation bar
        col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])
        with col1:
            if st.button("← Back", help="Return to the information page"):
                st.session_state.page = 'info'
                st.rerun()
        
        with col2:
            # Add company filter
            company_options = ['All', 'PineLabs', 'Razorpay', 'Paytm']
            selected_company = st.selectbox(
                "Filter by Company",
                options=company_options,
                index=company_options.index(st.session_state.selected_company),
                help="Select a specific company to focus on or 'All' for comparison"
            )
            
            if selected_company != st.session_state.selected_company:
                st.session_state.selected_company = selected_company
                st.rerun()

        with col4:
            data = st.session_state.data
            if 'improvement_report' not in st.session_state:
                st.session_state.improvement_report = None
            
            if 'positive_ratios' not in st.session_state:
                st.session_state.positive_ratios = None
            
            if 'selected_companies' not in st.session_state:
                st.session_state.selected_companies = None
                
            # Since sentiment counts is calculated early, we can get it directly
            sentiment_counts = data['sentiment'].value_counts()
            
            # Category sentiment data preparation needs to match what you calculate later
            category_sentiment = None
            try:
                # Try to calculate category sentiment similar to how it's done in the dashboard
                if 'category' in data.columns:
                    category_sentiment = data.groupby('category')['sentiment'].value_counts().unstack().fillna(0)
                    category_sentiment['total'] = category_sentiment.sum(axis=1)
                    
                    # Calculate percentages
                    for col in ['positive', 'negative', 'neutral']:
                        if col in category_sentiment.columns:
                            category_sentiment[f'{col}_percentage'] = (category_sentiment[col] / category_sentiment['total']) * 100
                    
                    # Sort by total and reset index
                    category_sentiment = category_sentiment.sort_values(by='total', ascending=False).reset_index()
            except Exception as e:
                st.error(f"Error preparing category data for report: {str(e)}")
                
            # Now create the download button
            add_report_download_button(
                data, 
                sentiment_counts, 
                category_sentiment, 
                st.session_state.positive_ratios, 
                st.session_state.improvement_report, 
                st.session_state.selected_companies
            )

        with col5:
            # Toggle view mode
            if st.button("Toggle View" + (" 📊" if st.session_state.view_mode == "cards" else " 🗃️"), 
                        help="Switch between card view and table view"):
                st.session_state.view_mode = "tables" if st.session_state.view_mode == "cards" else "cards"
                st.rerun()
            
        # Title with animation
        st.markdown("""
        <h1 style='text-align: center; animation: fadeDown 0.8s ease-out;'>
            Sentimeter Dashboard
        </h1>
        <style>
        @keyframes fadeDown {
            0% { opacity: 0; transform: translateY(-10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Get data
        data = st.session_state.data
        
        # Error handling
        if data is None or len(data) == 0:
            st.error("No data available for analysis. Please go back and try again.")
            return
        
        # Ensure datetime format for 'at' column
        if 'at' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['at']):
            try:
                data['at'] = pd.to_datetime(data['at'])
            except:
                st.warning("Could not convert 'at' column to datetime. Time-based analysis may be affected.")
        
        # Filter data based on selected company
        filtered_data = data
        if st.session_state.selected_company != 'All':
            filtered_data = data[data['source'] == st.session_state.selected_company]
            if len(filtered_data) == 0:
                st.warning(f"No data available for {st.session_state.selected_company}. Showing all data instead.")
                filtered_data = data
        
        # Calculate metrics for filtered data
        total_comments = len(filtered_data)
        sentiment_counts = filtered_data['sentiment'].value_counts()
        sentiment_percentages = sentiment_counts / total_comments * 100
        avg_sentiment_score = round(filtered_data['score'].mean(), 2)
        
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        pos_neg_ratio = round(positive_count / negative_count, 2) if negative_count > 0 else "N/A"
        positive_pct = round(sentiment_percentages.get('positive', 0), 1)
        
        # Date range info
        if 'at' in filtered_data.columns and pd.api.types.is_datetime64_any_dtype(filtered_data['at']):
            min_date = filtered_data['at'].min().strftime('%b %d, %Y')
            max_date = filtered_data['at'].max().strftime('%b %d, %Y')
            date_range = f"{min_date} to {max_date}"
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 20px; animation: fadeIn 1s ease-out;'>
                <span style='background-color: #345c49; color: white; padding: 8px 16px; border-radius: 20px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                Data Period: {date_range}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # Create tabs for different dashboard sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trends Analysis", "🔍 Content Analysis", "📝 Report","📋 Raw Data"])
        
        with tab1:
            # Top metrics row with animations
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metrics-card" style="animation: slideUp 0.8s ease-out;">
                    <div class="metric-title">Total Comments</div>
                    <div class="metric-value">{total_comments:,}</div>
                </div>
                <style>
                @keyframes slideUp {{
                    0% {{ opacity: 0; transform: translateY(20px); }}
                    100% {{ opacity: 1; transform: translateY(0); }}
                }}
                </style>
                """, unsafe_allow_html=True)

            # For Sentiment Distribution  
            with col2:
                st.markdown(f"""
                <div class="metrics-card" style="animation: slideUp 1s ease-out;">
                    <div class="metric-title">Sentiment Distribution</div>
                    <div class="metric-value">{positive_pct}% Positive</div>
                </div>
                """, unsafe_allow_html=True)

            # For Average Sentiment Score
            with col3:
                # Add color coding based on score
                score_color = "#4CAF50" if avg_sentiment_score > 0.6 else "#FFC107" if avg_sentiment_score > 0.4 else "#F44336"
                st.markdown(f"""
                <div class="metrics-card" style="animation: slideUp 1.2s ease-out;">
                    <div class="metric-title">Average Sentiment Score</div>
                    <div class="metric-value" style="color: {score_color};">{avg_sentiment_score}/1</div>
                </div>
                """, unsafe_allow_html=True)

            # For Positive vs Negative Ratio
            with col4:
                ratio_color = "#4CAF50" if pos_neg_ratio != "N/A" and pos_neg_ratio > 1 else "#F44336"
                st.markdown(f"""
                <div class="metrics-card" style="animation: slideUp 1.4s ease-out;">
                    <div class="metric-title">Positive vs Negative Ratio</div>
                    <div class="metric-value" style="color: {ratio_color};">{pos_neg_ratio}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Company Comparison Metrics when viewing all companies
            if st.session_state.selected_company == 'Compare All':
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                st.markdown("""
                    <div class="chart-container" style="animation: fadeIn 1.6s ease-out;">
                        <h3 class="chart-title">Company Comparison</h3>
                    """, unsafe_allow_html=True)
                
                # Calculate company-wise metrics
                company_metrics = []
                
                for company in ['PineLabs', 'Razorpay', 'Paytm']:
                    company_data = data[data['source'] == company]
                    if len(company_data) > 0:
                        company_sentiment = company_data['sentiment'].value_counts()
                        company_positive = company_sentiment.get('positive', 0)
                        company_negative = company_sentiment.get('negative', 0)
                        company_neutral = company_sentiment.get('neutral', 0)
                        total_company = len(company_data)
                        
                        company_metrics.append({
                            'company': company,
                            'total': total_company,
                            'positive': company_positive,
                            'negative': company_negative,
                            'neutral': company_neutral,
                            'positive_pct': round(company_positive / total_company * 100, 1) if total_company > 0 else 0,
                            'avg_score': round(company_data['score'].mean(), 2),
                            'pos_neg_ratio': round(company_positive / company_negative, 2) if company_negative > 0 else "N/A"
                        })
                
                # Display metrics in a comparative table
                if company_metrics:
                    metrics_df = pd.DataFrame(company_metrics)
                    
                    # Create visual comparison charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Positive sentiment percentage comparison
                        fig = px.bar(
                            metrics_df,
                            x='company',
                            y='positive_pct',
                            color='company',
                            color_discrete_map={
                                'PineLabs': '#345c49',
                                'Razorpay': '#3366CC',
                                'Paytm': '#0F4A8A'
                            },
                            labels={'positive_pct': 'Positive Sentiment (%)', 'company': 'Company'},
                            title="Positive Sentiment Percentage by Company"
                        )
                        fig.update_layout(
                            xaxis_title="Company",
                            yaxis_title="Positive Sentiment (%)",
                            legend_title="Company",
                            margin=dict(t=40, b=0, l=0, r=0),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Average sentiment score comparison
                        fig = px.bar(
                            metrics_df,
                            x='company',
                            y='avg_score',
                            color='company',
                            color_discrete_map={
                                'PineLabs': '#345c49',
                                'Razorpay': '#3366CC',
                                'Paytm': '#0F4A8A'
                            },
                            labels={'avg_score': 'Average Sentiment Score', 'company': 'Company'},
                            title="Average Sentiment Score by Company"
                        )
                        fig.update_layout(
                            xaxis_title="Company",
                            yaxis_title="Average Score",
                            legend_title="Company",
                            margin=dict(t=40, b=0, l=0, r=0),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display metrics table
                    st.markdown("<h4>Company Metrics Comparison</h4>", unsafe_allow_html=True)
                    
                    # Format the metrics table with badges
                    metrics_display = metrics_df.copy()
                    metrics_display['company'] = metrics_display['company'].apply(get_company_badge)
                    metrics_display.columns = ['Company', 'Total Comments', 'Positive', 'Negative', 'Neutral', 
                                             'Positive %', 'Avg Score', 'Pos/Neg Ratio']
                    
                    st.write(metrics_display.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Find leader in each metric and provide insights
                    best_positive_pct = metrics_df.loc[metrics_df['positive_pct'].idxmax()]
                    best_avg_score = metrics_df.loc[metrics_df['avg_score'].idxmax()]
                    
                    st.markdown(f"""
                    <div style="background-color: #f0f5f2; padding: 15px; border-radius: 10px; margin-top: 15px;">
                        <h4 style="color: #345c49; margin-top: 0;">📊 Competitive Insights</h4>
                        <ul>
                            <li><b>{best_positive_pct['company']}</b> leads in positive sentiment percentage at <b>{best_positive_pct['positive_pct']}%</b></li>
                            <li><b>{best_avg_score['company']}</b> has the highest average sentiment score of <b>{best_avg_score['avg_score']}</b></li>
                            <li>Pine Labs ranks <b>#{metrics_df['positive_pct'].rank(ascending=False).loc[metrics_df['company'] == 'PineLabs'].iloc[0]}</b> in positive sentiment</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.info("Not enough data to generate company comparison.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Sentiment Distribution (removed Channel Analysis)
            col1, _ = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="chart-container" style="animation: fadeIn 1.6s ease-out;">
                        <h3 class="chart-title">Sentiment Distribution</h3>
                    """, unsafe_allow_html=True)
                
                # Pie chart for sentiment distribution with error handling
                try:
                    fig = px.pie(
                        names=sentiment_counts.index.str.capitalize(),
                        values=sentiment_counts.values,
                        color=sentiment_counts.index,
                        color_discrete_map={'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'},
                        hole=0.4
                    )
                    fig.update_layout(
                        margin=dict(t=0, b=0, l=0, r=0),
                        legend_title="Sentiment"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating sentiment pie chart: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Time-based sentiment trend - simplified version for Overview tab
            st.markdown("""
                <div class="chart-container" style="animation: fadeIn 2s ease-out;">
                    <h3 class="chart-title">Sentiment Trend Over Time</h3>
                """, unsafe_allow_html=True)
            
            try:
                if 'at' in filtered_data.columns and pd.api.types.is_datetime64_any_dtype(filtered_data['at']):
                    # Group by date and sentiment
                    filtered_data['date'] = filtered_data['at'].dt.date
                    time_sentiment = filtered_data.groupby(['date', 'sentiment']).size().unstack().fillna(0)
                    
                    if not time_sentiment.empty:
                        # Create the trend line chart
                        fig = go.Figure()
                        
                        if 'positive' in time_sentiment.columns:
                            fig.add_trace(go.Scatter(
                                x=time_sentiment.index,
                                y=time_sentiment['positive'],
                                name='Positive',
                                mode='lines+markers',
                                line=dict(color='#4CAF50', width=3),
                                marker=dict(size=8)
                            ))
                        
                        if 'neutral' in time_sentiment.columns:
                            fig.add_trace(go.Scatter(
                                x=time_sentiment.index,
                                y=time_sentiment['neutral'],
                                name='Neutral',
                                mode='lines+markers',
                                line=dict(color='#FFC107', width=3),
                                marker=dict(size=8)
                            ))
                        
                        if 'negative' in time_sentiment.columns:
                            fig.add_trace(go.Scatter(
                                x=time_sentiment.index,
                                y=time_sentiment['negative'],
                                name='Negative',
                                mode='lines+markers',
                                line=dict(color='#F44336', width=3),
                                marker=dict(size=8)
                            ))
                        
                        fig.update_layout(
                            xaxis_title='Date',
                            yaxis_title='Number of Comments',
                            legend_title='Sentiment',
                            hovermode='x unified',
                            margin=dict(t=0, b=0, l=0, r=0)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add trend insights
                        recent_trend = time_sentiment.iloc[-3:] if len(time_sentiment) >= 3 else time_sentiment
                        
                        if not recent_trend.empty:
                            recent_positive = recent_trend['positive'].mean() if 'positive' in recent_trend.columns else 0
                            overall_positive = time_sentiment['positive'].mean() if 'positive' in time_sentiment.columns else 0
                            
                            trend_direction = "improving" if recent_positive > overall_positive else "declining"
                            
                            st.markdown(f"""
                            <div style="background-color: #f0f5f2; padding: 15px; border-radius: 10px; margin-top: 10px;">
                                <h4 style="color: #345c49; margin-top: 0;">📈 Trend Insights</h4>
                                <p>Recent past 3 days sentiment trend is <b>{trend_direction}</b> compared to the overall period.</p>
                                <p>See the "Trends Analysis" tab for more detailed time-based analysis.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Not enough time-series data to generate trend.")
                else:
                    st.info("Time data not available or not in the correct format.")
            except Exception as e:
                st.error(f"Error creating trend chart: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Call to action for other tabs
            st.markdown("""
            <div style="text-align: center; margin-top: 30px; animation: pulse 2s infinite ease-in-out;">
                <p style="color: #345c49; font-size: 1.2rem; font-weight: bold;">
                    Explore more insights in the other tabs above!
                </p>
            </div>
            <style>
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            </style>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<h3>Sentiment Trends Over Time</h3>", unsafe_allow_html=True)
            
            # Time-based filtering options
            col1, col2 = st.columns(2)
            with col1:
                time_grouping = st.selectbox(
                    "Time Grouping",
                    options=["Daily", "Weekly", "Monthly"],
                    index=0,
                    help="Select how to group the time series data"
                )
            
            with col2:
                chart_type = st.selectbox(
                    "Chart Type",
                    options=["Line", "Area", "Bar"],
                    index=0,
                    help="Select the type of chart for visualization"
                )
            
            # Time-based trend analysis with more options and detail
            try:
                if 'at' in filtered_data.columns and pd.api.types.is_datetime64_any_dtype(filtered_data['at']):
                    # Apply time grouping
                    if time_grouping == "Daily":
                        filtered_data['time_group'] = filtered_data['at'].dt.date
                    elif time_grouping == "Weekly":
                        filtered_data['time_group'] = filtered_data['at'].dt.to_period('W').apply(lambda x: x.start_time)
                    else:  # Monthly
                        filtered_data['time_group'] = filtered_data['at'].dt.to_period('M').apply(lambda x: x.start_time)
                    
                    # Group by time period and sentiment
                    time_sentiment = filtered_data.groupby(['time_group', 'sentiment']).size().unstack().fillna(0)

                    # Capitalize sentiment column names for the legend
                    time_sentiment.columns = [col.capitalize() for col in time_sentiment.columns]

                    if not time_sentiment.empty:
                        # Define updated color map with capitalized keys
                        color_map = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
                        
                        # Create the trend chart based on selection
                        if chart_type == "Line":
                            fig = px.line(
                                time_sentiment, 
                                labels={'value': 'Number of Comments', 'time_group': 'Date'},
                                color_discrete_map=color_map
                            )
                        elif chart_type == "Area":
                            fig = px.area(
                                time_sentiment,
                                labels={'value': 'Number of Comments', 'time_group': 'Date'},
                                color_discrete_map=color_map
                            )
                        else:  # Bar
                            fig = px.bar(
                                time_sentiment,
                                labels={'value': 'Number of Comments', 'time_group': 'Date'},
                                color_discrete_map=color_map
                            )

                        fig.update_layout(
                            title=f"{time_grouping} Sentiment Trend",
                            xaxis_title='Date',
                            yaxis_title='Number of Comments',
                            legend_title='Sentiment',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        time_sentiment.columns = [col[0].lower() + col[1:] if col else '' for col in time_sentiment.columns]


                        # Add advanced trend analysis
                        if len(time_sentiment) > 1:
                            # Calculate moving averages for smoother trend detection
                            window_size = min(3, len(time_sentiment))
                            if 'positive' in time_sentiment.columns:
                                time_sentiment['positive_ma'] = time_sentiment['positive'].rolling(window=window_size).mean()
                            if 'negative' in time_sentiment.columns:
                                time_sentiment['negative_ma'] = time_sentiment['negative'].rolling(window=window_size).mean()
                            
                            # Get first half and second half averages to detect overall trend
                            half_point = len(time_sentiment) // 2
                            
                            # Initialize variables with default values
                            first_half_pos = 0
                            second_half_pos = 0
                            positive_trend = 0
                            positive_trend_pct = 0
                            first_half_neg = 0
                            second_half_neg = 0
                            negative_trend = 0
                            negative_trend_pct = 0
                            
                            # First half and second half for positive sentiment
                            if 'positive' in time_sentiment.columns and half_point > 0:
                                first_half_pos = time_sentiment['positive'].iloc[:half_point].mean()
                                second_half_pos = time_sentiment['positive'].iloc[half_point:].mean()
                                positive_trend = second_half_pos - first_half_pos
                                positive_trend_pct = (positive_trend / first_half_pos * 100) if first_half_pos > 0 else 0
                            
                            # First half and second half for negative sentiment
                            if 'negative' in time_sentiment.columns and half_point > 0:
                                first_half_neg = time_sentiment['negative'].iloc[:half_point].mean()
                                second_half_neg = time_sentiment['negative'].iloc[half_point:].mean()
                                negative_trend = second_half_neg - first_half_neg
                                negative_trend_pct = (negative_trend / first_half_neg * 100) if first_half_neg > 0 else 0
                            
                            # Insights box
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                positive_icon = "📈" if positive_trend > 0 else "📉"
                                positive_trend_desc = "increased" if positive_trend > 0 else "decreased"
                                
                                st.markdown(f"""
                                <div class="chart-container">
                                    <h4>Positive Sentiment Trend</h4>
                                    <p>{positive_icon} Positive sentiment has <b>{positive_trend_desc} by {abs(positive_trend_pct):.1f}%</b> 
                                    from the first half to the second half of the period.</p>
                                    <p>Average positive comments in recent period: <b>{second_half_pos:.1f}</b></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                # For negative, we want the opposite meaning (decrease is good)
                                negative_icon = "📉" if negative_trend < 0 else "📈"
                                negative_trend_color = "#4CAF50" if negative_trend < 0 else "#F44336"
                                negative_trend_desc = "decreased" if negative_trend < 0 else "increased"
                                
                                st.markdown(f"""
                                <div class="chart-container">
                                    <h4>Negative Sentiment Trend</h4>
                                    <p>{negative_icon} Negative sentiment has <b style="color: {negative_trend_color}">
                                    {negative_trend_desc} by {abs(negative_trend_pct):.1f}%</b> 
                                    from the first half to the second half of the period.</p>
                                    <p>Average negative comments in recent period: <b>{second_half_neg:.1f}</b></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add overall sentiment trend score
                            overall_trend = positive_trend_pct - negative_trend_pct
                            
                            trend_color = "#4CAF50" if overall_trend > 0 else "#F44336" if overall_trend < 0 else "#FFC107"
                            trend_desc = "Improving" if overall_trend > 5 else "Declining" if overall_trend < -5 else "Stable"
                            
                            st.markdown(f"""
                            <div class="chart-container">
                                <h4>Overall Sentiment Trend Analysis</h4>
                                <p>The overall sentiment trend is <b style="color: {trend_color}">{trend_desc}</b>.</p>
                                <p>This analysis compares the first half of the time period to the second half.</p>
                                <ul>
                                    <li>A positive trend indicates improving sentiment over time</li>
                                    <li>A negative trend indicates declining sentiment over time</li>
                                    <li>A stable trend indicates consistent sentiment levels</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Peak analysis
                            if len(time_sentiment) >= 3:
                                try:
                                    peak_positive_day = "N/A"
                                    peak_positive_count = 0
                                    peak_negative_day = "N/A"
                                    peak_negative_count = 0
                                    
                                    # Find peak positive and negative days
                                    if 'positive' in time_sentiment.columns:
                                        peak_positive_day = time_sentiment['positive'].idxmax()
                                        peak_positive_count = time_sentiment.loc[peak_positive_day, 'positive']
                                    
                                    if 'negative' in time_sentiment.columns:
                                        peak_negative_day = time_sentiment['negative'].idxmax()
                                        peak_negative_count = time_sentiment.loc[peak_negative_day, 'negative']
                                    
                                    st.markdown(f"""
                                    <div class="chart-container">
                                        <h4>Peak Analysis</h4>
                                        <p>📆 <b>Peak positive day:</b> {peak_positive_day} with {peak_positive_count:.0f} positive comments</p>
                                        <p>📆 <b>Peak negative day:</b> {peak_negative_day} with {peak_negative_count:.0f} negative comments</p>
                                        <p>These peaks may correspond to significant events or product changes.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error in peak analysis: {str(e)}")
                        
                    else:
                        st.info("Not enough time-series data to generate trend.")
                else:
                    st.info("Time data not available or not in the correct format.")
            except Exception as e:
                st.error(f"Error creating detailed trend chart: {str(e)}")
            
            # Add competitor trend comparison if 'All' companies selected
            if st.session_state.selected_company == 'All':
                st.markdown("<h3>Competitor Trend Comparison</h3>", unsafe_allow_html=True)
                
                try:
                    # Create comparative time trend by company
                    if 'at' in data.columns and 'source' in data.columns:
                        # Apply time grouping
                        if time_grouping == "Daily":
                            data['time_group'] = data['at'].dt.date
                        elif time_grouping == "Weekly":
                            data['time_group'] = data['at'].dt.to_period('W').apply(lambda x: x.start_time)
                        else:  # Monthly
                            data['time_group'] = data['at'].dt.to_period('M').apply(lambda x: x.start_time)
                        
                        # Calculate positive sentiment percentage by company and time
                        company_trends = []
                        
                        for company in ['PineLabs', 'Razorpay', 'Paytm']:
                            company_data = data[data['source'] == company]
                            if len(company_data) > 0:
                                # Group by time and calculate positive percentage
                                time_sentiment = company_data.groupby(['time_group', 'sentiment']).size().unstack().fillna(0)
                                
                                if not time_sentiment.empty and 'positive' in time_sentiment.columns:
                                    time_sentiment['total'] = time_sentiment.sum(axis=1)
                                    time_sentiment['positive_pct'] = (time_sentiment['positive'] / time_sentiment['total'] * 100).round(1)
                                    
                                    for date, row in time_sentiment.iterrows():
                                        company_trends.append({
                                            'Date': date,
                                            'Company': company,
                                            'Positive_Percentage': row['positive_pct'] if 'positive_pct' in row else 0
                                        })
                        
                        if company_trends:
                            # Create DataFrame from the collected data
                            trends_df = pd.DataFrame(company_trends)
                            
                            # Create comparative trend chart
                            fig = px.line(
                                trends_df,
                                x='Date',
                                y='Positive_Percentage',
                                color='Company',
                                color_discrete_map={
                                    'PineLabs': '#345c49',
                                    'Razorpay': '#3366CC',
                                    'Paytm': '#0F4A8A'
                                },
                                labels={'Positive_Percentage': 'Positive Sentiment (%)', 'Date': 'Date'},
                                title=f"Comparative Positive Sentiment Trend ({time_grouping})"
                            )
                            
                            fig.update_layout(
                                xaxis_title='Date',
                                yaxis_title='Positive Sentiment (%)',
                                legend_title='Company',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate trend change by company
                            company_insights = []
                            
                            for company in ['PineLabs', 'Razorpay', 'Paytm']:
                                company_trend = trends_df[trends_df['Company'] == company]
                                
                                if len(company_trend) > 1:
                                    half_point = len(company_trend) // 2
                                    first_half = company_trend['Positive_Percentage'].iloc[:half_point].mean()
                                    second_half = company_trend['Positive_Percentage'].iloc[half_point:].mean()
                                    
                                    trend_change = second_half - first_half
                                    trend_change_pct = (trend_change / first_half * 100) if first_half > 0 else 0
                                    
                                    company_insights.append({
                                        'Company': company,
                                        'First_Half': first_half,
                                        'Second_Half': second_half,
                                        'Trend_Change': trend_change,
                                        'Trend_Change_Pct': trend_change_pct
                                    })
                            
                            if company_insights:
                                # Create insights box
                                insights_df = pd.DataFrame(company_insights)
                                
                                # Find company with best improvement
                                best_improvement = insights_df.loc[insights_df['Trend_Change'].idxmax()]
                                worst_trend = insights_df.loc[insights_df['Trend_Change'].idxmin()]
                                
                                # PineLabs specific insight - handle if PineLabs is not in the data
                                pine_insight = None
                                if 'PineLabs' in insights_df['Company'].values:
                                    pine_insight = insights_df[insights_df['Company'] == 'PineLabs'].iloc[0]
                                
                                st.markdown("""
                                <div class="chart-container">
                                    <h4>Competitive Trend Insights</h4>
                                </div>
                                """, unsafe_allow_html=True)

                            # Create columns for each company insight
                            cols = st.columns(len(company_insights))

                            # Define company color function
                            def get_company_color(company):
                                colors = {
                                    'PineLabs': '#345c49',
                                    'Razorpay': '#3366CC',
                                    'Paytm': '#0F4A8A'
                                }
                                return colors.get(company, '#777777')
                                
                            # Display insights for each company in separate columns
                            for i, insight in enumerate(company_insights):
                                with cols[i]:
                                    trend_icon = "📈" if insight['Trend_Change'] > 0 else "📉" if insight['Trend_Change'] < 0 else "⟷"
                                    trend_color = "#4CAF50" if insight['Trend_Change'] > 0 else "#F44336" if insight['Trend_Change'] < 0 else "#FFC107"
                                    
                                    st.markdown(f"""
                                    <div style="background-color: {get_company_color(insight['Company'])}1A; padding: 15px; border-radius: 10px;">
                                        <h4 style="color: {get_company_color(insight['Company'])};">{insight['Company']}</h4>
                                        <p>{trend_icon} <b style="color: {trend_color}">
                                        {'+' if insight['Trend_Change'] > 0 else ''}{insight['Trend_Change']:.1f}% points</b></p>
                                        <p>First half average: {insight['First_Half']:.1f}%</p>
                                        <p>Second half average: {insight['Second_Half']:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("Not enough data to generate comparative trends.")
                    else:
                        st.info("Source and time data not available.")
                except Exception as e:
                    st.error(f"Error creating competitor trend comparison: {str(e)}")
            
            # Day of week analysis
            st.markdown("<h3>Day of Week Analysis</h3>", unsafe_allow_html=True)
            
            try:
                if 'at' in filtered_data.columns and pd.api.types.is_datetime64_any_dtype(filtered_data['at']):
                    # Extract day of week
                    filtered_data['day_of_week'] = filtered_data['at'].dt.day_name()
                    
                    # Order days correctly
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    # Group by day of week and sentiment
                    dow_sentiment = filtered_data.groupby(['day_of_week', 'sentiment']).size().unstack().fillna(0)
                    
                    if not dow_sentiment.empty:
                        # Reindex to ensure correct day order
                        dow_sentiment = dow_sentiment.reindex(day_order)
                        
                        # Calculate positive percentage
                        dow_sentiment['total'] = dow_sentiment.sum(axis=1)
                        if 'positive' in dow_sentiment.columns:
                            dow_sentiment['positive_pct'] = (dow_sentiment['positive'] / dow_sentiment['total'] * 100).round(1)
                        
                        # Create a grouped bar chart for day of week analysis
                        fig = go.Figure()
                        
                        if 'positive' in dow_sentiment.columns:
                            fig.add_trace(go.Bar(
                                x=dow_sentiment.index,
                                y=dow_sentiment['positive'],
                                name='Positive',
                                marker_color='#4CAF50'
                            ))
                        
                        if 'neutral' in dow_sentiment.columns:
                            fig.add_trace(go.Bar(
                                x=dow_sentiment.index,
                                y=dow_sentiment['neutral'],
                                name='Neutral',
                                marker_color='#FFC107'
                            ))
                        
                        if 'negative' in dow_sentiment.columns:
                            fig.add_trace(go.Bar(
                                x=dow_sentiment.index,
                                y=dow_sentiment['negative'],
                                name='Negative',
                                marker_color='#F44336'
                            ))
                        
                        fig.update_layout(
                            title='Sentiment by Day of Week',
                            xaxis_title='Day of Week',
                            yaxis_title='Number of Comments',
                            barmode='group',
                            legend_title='Sentiment'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a line chart for positive percentage by day of week
                        if 'positive_pct' in dow_sentiment.columns:
                            fig = px.line(
                                x=dow_sentiment.index,
                                y=dow_sentiment['positive_pct'],
                                markers=True,
                                labels={'x': 'Day of Week', 'y': 'Positive Sentiment (%)'},
                                title='Positive Sentiment Percentage by Day of Week'
                            )
                            
                            fig.update_traces(line=dict(color='#345c49', width=3), marker=dict(size=10))
                            
                            fig.update_layout(
                                xaxis_title='Day of Week',
                                yaxis_title='Positive Sentiment (%)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Initialize default values
                            best_day = "N/A"
                            best_day_pct = 0
                            worst_day = "N/A"
                            worst_day_pct = 0
                            weekday_positive_pct = 0
                            weekend_positive_pct = 0
                            
                            # Check if we have data to calculate
                            if not dow_sentiment['positive_pct'].empty:
                                best_day = dow_sentiment['positive_pct'].idxmax()
                                best_day_pct = dow_sentiment.loc[best_day, 'positive_pct']
                                
                                worst_day = dow_sentiment['positive_pct'].idxmin()
                                worst_day_pct = dow_sentiment.loc[worst_day, 'positive_pct']
                            
                            # Calculate weekday vs weekend comparison
                            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                            weekends = ['Saturday', 'Sunday']
                            
                            weekday_data = dow_sentiment.loc[dow_sentiment.index.isin(weekdays)]
                            weekend_data = dow_sentiment.loc[dow_sentiment.index.isin(weekends)]
                            
                            if not weekday_data.empty and 'positive_pct' in weekday_data.columns:
                                weekday_positive_pct = weekday_data['positive_pct'].mean()
                            
                            if not weekend_data.empty and 'positive_pct' in weekend_data.columns:
                                weekend_positive_pct = weekend_data['positive_pct'].mean()
                            
                            weekday_weekend_diff = weekday_positive_pct - weekend_positive_pct
                            
                            st.markdown(f"""
                            <div class="chart-container">
                                <h4>Day of Week Insights</h4>
                                <ul>
                                    <li>Best day for positive sentiment: <b>{best_day}</b> at <b>{best_day_pct:.1f}%</b> positive</li>
                                    <li>Worst day for positive sentiment: <b>{worst_day}</b> at <b>{worst_day_pct:.1f}%</b> positive</li>
                                    <li>Weekday average positive: <b>{weekday_positive_pct:.1f}%</b></li>
                            <li>Weekend average positive: <b>{weekend_positive_pct:.1f}%</b></li>
                            <li>Sentiment is <b>{abs(weekday_weekend_diff):.1f}%</b> {'higher' if weekday_weekend_diff > 0 else 'lower'} on weekdays compared to weekends</li>
                                </ul>
                                
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Not enough data to generate day of week analysis.")
                else:
                    st.info("Time data not available or not in the correct format.")
            except Exception as e:
                st.error(f"Error creating day of week analysis: {str(e)}")
        
        with tab3:
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            # Update title to "Review Analysis"
            st.markdown("""
                <div class="chart-container">
                    <h3 class="chart-title">Review Analysis</h3>
                """, unsafe_allow_html=True)
            
            # Create two columns for positive and negative reviews
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 5 positive reviews
                st.markdown("""
                    <div style="background-color: #f0f5f2; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                        <h4 style="color: #4CAF50; margin-top: 0;">🌟 Top 5 Positive Reviews</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Filter positive reviews and sort by score in descending order
                    positive_reviews = data[(data['sentiment'] == 'positive') & (data['source'] == 'PineLabs')].sort_values(by='score', ascending=False).head(5)
                    
                    # Display each positive review with score and source
                    for i, (_, comment) in enumerate(positive_reviews.iterrows(), 1):
                        # Format date if available
                        review_date = comment['at'].strftime('%b %d, %Y') if 'at' in comment and pd.notna(comment['at']) else "N/A"
                        
                        # Create card for each review
                        st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; 
                            border-left: 4px solid #4CAF50; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <span style="font-weight: bold; color: #345c49;">Review #{i}</span>
                                <span style="color: #666;">Score: <b>{comment['score']:.2f}</b></span>
                            </div>
                            <p style="margin-bottom: 10px;">{comment['review']}</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #666;">
                                <span>Source: {comment.get('source', 'N/A')}</span>
                                <span>Date: {review_date}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying positive reviews: {str(e)}")
            
            with col2:
                # Top 5 negative reviews
                st.markdown("""
                    <div style="background-color: #f0f5f2; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                        <h4 style="color: #F44336; margin-top: 0;">👎 Top 5 Negative Reviews</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Filter negative reviews and sort by score in ascending order
                    negative_reviews = data[(data['sentiment'] == 'negative') & (data['source'] == 'PineLabs')].sort_values(by='score', ascending=False).head(5)
                    
                    # Display each negative review with score and source
                    for i, (_, comment) in enumerate(negative_reviews.iterrows(), 1):
                        # Format date if available
                        review_date = comment['at'].strftime('%b %d, %Y') if 'at' in comment and pd.notna(comment['at']) else "N/A"
                        
                        # Create card for each review
                        st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; 
                            border-left: 4px solid #F44336; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <span style="font-weight: bold; color: #345c49;">Review #{i}</span>
                                <span style="color: #666;">Score: <b>{comment['score']:.2f}</b></span>
                            </div>
                            <p style="margin-bottom: 10px;">{comment['review']}</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #666;">
                                <span>Source: {comment.get('source')}</span>
                                <span>Date: {review_date}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying negative reviews: {str(e)}")
            
            # Word Cloud for negative comments with improved cleaning and visualization
            st.markdown("""
                <div class="chart-container">
                    <h3 class="chart-title">Word Cloud - Negative Comments</h3>
                </div>
            """, unsafe_allow_html=True)

            try:
                # Filter only negative comments
                negative_comments = data[(data['sentiment'] == 'negative') & (data['source'] == 'PineLabs')]
                
                if len(negative_comments) > 0:
                    # Join all text from negative comments - use 'review' column instead of 'text'
                    all_text = ' '.join(negative_comments['review'].astype(str).fillna('').tolist())
                    
                    # Enhanced text cleaning
                    # Remove URLs
                    all_text = re.sub(r'http\S+', '', all_text)
                    # Remove email addresses
                    all_text = re.sub(r'\S*@\S*\s?', '', all_text)
                    # Remove special characters and numbers
                    all_text = re.sub(r'[^\w\s]', '', all_text)
                    all_text = re.sub(r'\d+', '', all_text)
                    all_text = all_text.lower()
                    
                    # Expanded stop words list
                    stop_words = set([
                        'the', 'and', 'is', 'in', 'it', 'to', 'that', 'of', 'for', 'on', 'with', 
                        'as', 'this', 'by', 'be', 'are', 'was', 'were', 'at', 'from', 'has', 'have', 
                        'had', 'a', 'an', 'i', 'my', 'we', 'you', 'your', 'our', 'they', 'their', 
                        'app', 'pine', 'labs', 'would', 'could', 'should', 'will', 'can', 'just',
                        'not', 'but', 'or', 'so', 'what', 'when', 'where', 'who', 'how', 'which',
                        'there', 'here', 'than', 'then', 'them', 'these', 'those', 'some', 'such',
                        'very', 'much', 'many', 'any', 'all', 'one', 'two', 'three', 'first', 'last'
                    ])
                    
                    # Filter out stop words and short words
                    filtered_words = [word for word in all_text.split() if word not in stop_words and len(word) > 2]
                    cleaned_text = ' '.join(filtered_words)
                    
                    if cleaned_text.strip():
                        # Generate word cloud with improved styling
                        wordcloud = WordCloud(
                            width=800,
                            height=400,
                            background_color='white',
                            max_words=100,
                            colormap='Reds',
                            contour_width=1,
                            contour_color='#F44336',
                            collocations=False,
                            prefer_horizontal=0.9,  # Prefer horizontal text (90%)
                            relative_scaling=0.5,   # Balance frequency vs. random placement
                            min_font_size=8,
                            max_font_size=150,
                            mask=None,              # No mask for cleaner look
                            random_state=42         # For reproducibility
                        ).generate(cleaned_text)
                        
                        # Convert to image with improved styling
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        plt.tight_layout(pad=0)
                        
                        # Add a subtle background color to the figure
                        fig.patch.set_facecolor('#f9f9f9')
                        
                        # Save to buffer for display
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight', facecolor='#f9f9f9')
                        buf.seek(0)
                        
                        # Display with container width
                        st.image(buf, use_container_width=True)
                        
                        # Get top negative words with more detailed analysis
                        word_freq = Counter(filtered_words)
                        top_negative_words = word_freq.most_common(10)
                        
                        # Display top words with improved styling
                        st.markdown("""
                        <div style="background-color: #f8f1f1; padding: 15px; border-radius: 10px; margin-top: 15px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            <h4 style="color: #345c49; margin-top: 0; font-weight: 600;">🔍 Top Words in Negative Reviews</h4>
                            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
                        """, unsafe_allow_html=True)
                        
                        # Create gradient of red colors for the tags
                        color_start = [244, 67, 54]  # #F44336
                        color_end = [198, 40, 40]    # #C62828
                        
                        for i, (word, count) in enumerate(top_negative_words):
                            # Calculate gradient color based on position
                            ratio = i / len(top_negative_words)
                            r = int(color_start[0] * (1-ratio) + color_end[0] * ratio)
                            g = int(color_start[1] * (1-ratio) + color_end[1] * ratio)
                            b = int(color_start[2] * (1-ratio) + color_end[2] * ratio)
                            color = f"rgb({r},{g},{b})"
                            
                            # Show word with count
                            st.markdown(f"""
                            <div style="background-color: {color}; color: white; padding: 8px 15px; 
                                border-radius: 20px; font-size: 0.9em; display: flex; align-items: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                                <span style="margin-right: 8px; font-weight: 500;">{word}</span>
                                <span style="background-color: white; color: {color}; border-radius: 50%; 
                                    width: 25px; height: 25px; display: flex; align-items: center; 
                                    justify-content: center; font-size: 0.8em; font-weight: bold;">
                                    {count}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        # Add simple frequency insight
                        most_common = top_negative_words[0] if top_negative_words else (None, 0)
                        if most_common[0]:
                            percentage = (most_common[1] / len(filtered_words)) * 100
                            st.markdown(f"""
                            <div style="margin-top: 15px; font-style: italic; color: #666;">
                            The word "<b>{most_common[0]}</b>" appears {most_common[1]} times, 
                            representing approximately {percentage:.1f}% of all significant words in negative reviews.
                            </div>
                            """, unsafe_allow_html=True)
                            
                    else:
                        st.info("Not enough text in negative comments to generate a word cloud.")
                else:
                    st.info("No negative comments available for word cloud generation.")
            except Exception as e:
                st.error(f"Error generating negative comments word cloud: {str(e)}")
                # Add debug information
                st.markdown("<details><summary>Debug info</summary>", unsafe_allow_html=True)
                st.exception(e)
                st.markdown("</details>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown("<h3>Comparative Analysis</h3>", unsafe_allow_html=True)
            
            if 'source' in data.columns and 'category' in data.columns:
                # Filters for comparative analysis
                st.markdown("<h4>Comparison Settings</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    compare_metrics = st.multiselect(
                        "Comparison Metrics",
                        options=["Overall Sentiment", "Positive Ratio", "Topic Distribution"],
                        default=["Overall Sentiment", "Topic Distribution"],
                        help="Select metrics to compare across companies"
                    )
                
                with col2:
                    # Get unique companies from source column
                    available_companies = sorted(data['source'].unique().tolist())
                    compare_companies = st.multiselect(
                        "Companies to Compare",
                        options=available_companies,
                        default=available_companies[:3] if len(available_companies) >= 3 else available_companies,
                        help="Select companies to include in comparison"
                    )
                st.session_state.selected_companies = compare_companies
                # Make sure we have at least one company selected
                if len(compare_companies) > 0:
                    # Filter data for selected companies
                    compare_data = data[data['source'].isin(compare_companies)]
                    
                    # Overall Sentiment Comparison
                    if "Overall Sentiment" in compare_metrics:
                        st.markdown("<h4>Overall Sentiment Comparison</h4>", unsafe_allow_html=True)
                        
                        try:
                            # Group by company and sentiment
                            company_sentiment = compare_data.groupby(['source', 'sentiment']).size().unstack().fillna(0)
                            
                            if not company_sentiment.empty:
                                # Create stacked bar chart for sentiment comparison
                                company_sentiment_long = company_sentiment.reset_index().melt(
                                    id_vars=['source'],
                                    value_vars=company_sentiment.columns,
                                    var_name='sentiment',
                                    value_name='count'
                                )
                                company_sentiment_long['sentiment'] = company_sentiment_long['sentiment'].str.capitalize()
                                fig = px.bar(
                                    company_sentiment_long,
                                    x='source',
                                    y='count',
                                    color='sentiment',
                                    color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'},
                                    title='Sentiment Distribution by Company',
                                    labels={'source': 'Company', 'count': 'Number of Comments', 'sentiment': 'Sentiment'}
                                )
                                
                                fig.update_layout(barmode='stack')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Calculate percentage distribution
                                company_totals = company_sentiment.sum(axis=1)
                                company_sentiment_pct = company_sentiment.div(company_totals, axis=0) * 100
                                
                                # Create normalized stacked bar chart (percentage)
                                company_sentiment_pct_long = company_sentiment_pct.reset_index().melt(
                                    id_vars=['source'],
                                    value_vars=company_sentiment_pct.columns,
                                    var_name='sentiment',
                                    value_name='percentage'
                                )
                                company_sentiment_pct_long['sentiment'] = company_sentiment_pct_long['sentiment'].str.capitalize()
                                fig = px.bar(
                                    company_sentiment_pct_long,
                                    x='source',
                                    y='percentage',
                                    color='sentiment',
                                    color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'},
                                    title='Sentiment Distribution by Company (Percentage)',
                                    labels={'source': 'Company', 'percentage': 'Percentage (%)', 'sentiment': 'Sentiment'}
                                )
                                
                                fig.update_layout(barmode='stack', yaxis=dict(ticksuffix='%'))
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Not enough data for sentiment comparison.")
                        except Exception as e:
                            st.error(f"Error creating sentiment comparison: {str(e)}")
                    
                    # Positive Ratio Comparison
                    if "Positive Ratio" in compare_metrics:
                        st.markdown("<h4>Positive Sentiment Ratio Comparison</h4>", unsafe_allow_html=True)
                        
                        try:
                            # Calculate positive ratio for each company
                            positive_ratios = []
                            
                            for company in compare_companies:
                                company_data = compare_data[compare_data['source'] == company]
                                
                                if len(company_data) > 0:
                                    sentiment_counts = company_data['sentiment'].value_counts()
                                    total_comments = len(company_data)
                                    
                                    positive_count = sentiment_counts.get('positive', 0)
                                    negative_count = sentiment_counts.get('negative', 0)
                                    neutral_count = sentiment_counts.get('neutral', 0)
                                    
                                    positive_ratio = (positive_count / total_comments * 100) if total_comments > 0 else 0
                                    negative_ratio = (negative_count / total_comments * 100) if total_comments > 0 else 0
                                    
                                    # Calculate positive to negative ratio (handle division by zero)
                                    positive_to_negative = (positive_count / max(negative_count, 1)) if negative_count > 0 else positive_count
                                    
                                    positive_ratios.append({
                                        'company': company,
                                        'positive_percentage': positive_ratio,
                                        'negative_percentage': negative_ratio,
                                        'positive_to_negative_ratio': positive_to_negative,
                                        'total_comments': total_comments
                                    })
                            
                            if positive_ratios:
                                # Create DataFrame for visualization
                                ratios_df = pd.DataFrame(positive_ratios)
                                
                                # Sort by positive percentage
                                ratios_df = ratios_df.sort_values(by='positive_percentage', ascending=False)
                                st.session_state.positive_ratios = ratios_df
                                # Create horizontal bar chart
                                fig = px.bar(
                                    ratios_df,
                                    y='company',
                                    x='positive_percentage',
                                    orientation='h',
                                    color='company',
                                    title='Positive Sentiment Percentage by Company',
                                    labels={'company': 'Company', 'positive_percentage': 'Positive Sentiment (%)'},
                                    text='positive_percentage'
                                )
                                
                                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis=dict(autorange="reversed"))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Create positive to negative ratio chart
                                fig = px.bar(
                                    ratios_df,
                                    y='company',
                                    x='positive_to_negative_ratio',
                                    orientation='h',
                                    color='company',
                                    title='Positive to Negative Ratio by Company',
                                    labels={'company': 'Company', 'positive_to_negative_ratio': 'Positive to Negative Ratio'},
                                    text='positive_to_negative_ratio'
                                )
                                
                                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis=dict(autorange="reversed"))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Create comparison table
                                st.markdown("<h4>Sentiment Metrics Comparison</h4>", unsafe_allow_html=True)
                                
                                # Format the table data
                                comparison_table = ratios_df.copy()
                                comparison_table['positive_percentage'] = comparison_table['positive_percentage'].round(1).astype(str) + '%'
                                comparison_table['negative_percentage'] = comparison_table['negative_percentage'].round(1).astype(str) + '%'
                                comparison_table['positive_to_negative_ratio'] = comparison_table['positive_to_negative_ratio'].round(2).astype(str)
                                comparison_table['total_comments'] = comparison_table['total_comments'].astype(int).astype(str)
                                
                                # Rename columns for display
                                comparison_table.columns = ['Company', 'Positive %', 'Negative %', 'Positive/Negative Ratio', 'Total Comments']
                                
                                # Display the table
                                st.dataframe(comparison_table, hide_index=True, use_container_width=True)
                                
                                # Find the best performer
                                best_company = ratios_df.iloc[0]['company']
                                best_positive_pct = ratios_df.iloc[0]['positive_percentage']
                                
                                # Find PineLabs ranking if it exists in the compared companies
                                pine_info = ""
                                if 'PineLabs' in ratios_df['company'].values:
                                    pine_rank = ratios_df[ratios_df['company'] == 'PineLabs'].index[0] + 1
                                    pine_pct = ratios_df[ratios_df['company'] == 'PineLabs']['positive_percentage'].values[0]
                                    pine_info = f"""<li>Pine Labs ranks <b>#{pine_rank}</b> with <b>{pine_pct:.1f}%</b> positive sentiment</li>"""
                                
                                # Create insights box
                                st.markdown(f"""
                                <div style="background-color: #f0f5f2; padding: 15px; border-radius: 10px; margin-top: 10px;">
                                    <h4 style="color: #345c49; margin-top: 0;">Comparative Insights</h4>
                                    <ul>
                                        <li><b>{best_company}</b> has the highest positive sentiment at <b>{best_positive_pct:.1f}%</b></li>
                                        {pine_info}
                                        <li>The industry average positive sentiment is <b>{ratios_df['positive_percentage'].mean():.1f}%</b></li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("Not enough data for positive ratio comparison.")
                        except Exception as e:
                            st.error(f"Error creating positive ratio comparison: {str(e)}")
                    
                    
                    # Topic Distribution Comparison (Using the category column)
                    # Topic Distribution Comparison (Using the category column)
                if "Topic Distribution" in compare_metrics:
                    st.markdown("<h4>Topic Distribution Comparison</h4>", unsafe_allow_html=True)
                    
                    try:
                        # Create topic distribution data from the actual category column, now including sentiment
                        topic_sentiment_data = []
                        
                        for company in compare_companies:
                            company_data = compare_data[compare_data['source'] == company]
                            
                            if len(company_data) > 0:
                                # Group by category and sentiment
                                category_sentiment_counts = company_data.groupby(['category', 'sentiment']).size().reset_index(name='count')
                                
                                # Get total counts per category to calculate percentages
                                category_totals = company_data['category'].value_counts().reset_index()
                                category_totals.columns = ['category', 'total']
                                
                                # Merge to calculate percentages
                                category_sentiment_counts = category_sentiment_counts.merge(category_totals, on='category')
                                category_sentiment_counts['percentage'] = (category_sentiment_counts['count'] / category_sentiment_counts['total']) * 100
                                
                                # Add company information
                                category_sentiment_counts['company'] = company
                                
                                # Add to the main dataframe
                                topic_sentiment_data.append(category_sentiment_counts[['company', 'category', 'sentiment', 'percentage']])
                        
                        if topic_sentiment_data:
                            # Combine all company data
                            topic_sentiment_df = pd.concat(topic_sentiment_data, ignore_index=True)
                            
                            # Create another view: Topic by Sentiment Chart (multi-level bar chart)
                            st.markdown("<h5>Category Performance Comparison by Sentiment</h5>", unsafe_allow_html=True)
                            
                            # Create selector for specific category to analyze across companies
                            selected_category = st.selectbox(
                                "Select category to compare across companies:",
                                options=sorted(topic_sentiment_df['category'].unique()),
                                index=0
                            )
                            
                            # Filter for selected category
                            category_comparison = topic_sentiment_df[topic_sentiment_df['category'] == selected_category]
                            category_comparison['sentiment'] = category_comparison['sentiment'].str.capitalize()
                            if not category_comparison.empty:
                                fig = px.bar(
                                    category_comparison,
                                    x='company',
                                    y='percentage',
                                    color='sentiment',
                                    color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'},
                                    title=f'Sentiment Distribution for "{selected_category}" Category Across Companies',
                                    labels={'company': 'Company', 'percentage': 'Percentage (%)', 'sentiment': 'Sentiment'},
                                    text='percentage'
                                )
                                
                                fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                                fig.update_layout(barmode='stack', yaxis=dict(ticksuffix='%'))
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Create heatmap with sentiment breakdown
                            st.markdown("<h5>Sentiment by Topic Heatmap</h5>", unsafe_allow_html=True)
                            
                            # Let user select which sentiment to view in the heatmap
                            selected_sentiment = st.radio(
                                "Select sentiment to view:",
                                options=[ 'All','Positive', 'Negative', 'Neutral'],
                                horizontal=True,
                                index=0
                            )
                            
                            # Filter based on selected sentiment
                            if selected_sentiment != 'All':
                                heatmap_data = topic_sentiment_df[topic_sentiment_df['sentiment'] == selected_sentiment]
                                title_prefix = selected_sentiment.capitalize()
                            else:
                                # For 'all', we need to pivot differently to show all sentiments
                                heatmap_data = topic_sentiment_df
                                title_prefix = 'Overall'
                            
                            # Create the appropriate heatmap based on selection
                            if selected_sentiment != 'All':
                                # Pivot the data for heatmap
                                topic_pivot = heatmap_data.pivot(index='company', columns='category', values='percentage').fillna(0)
                                
                                # Create heatmap
                                fig = px.imshow(
                                    topic_pivot,
                                    text_auto='.1f',
                                    labels=dict(x="Category", y="Company", color=f"{title_prefix} Sentiment (%)"),
                                    color_continuous_scale='Viridis',
                                    title=f'{title_prefix} Sentiment Percentage by Category and Company'
                                )
                                
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Create a more complex heatmap showing sentiment ratio
                                # Calculate the sentiment ratio (positive - negative) for each category and company
                                sentiment_ratio = topic_sentiment_df.pivot_table(
                                    index=['company', 'category'], 
                                    columns='sentiment', 
                                    values='percentage',
                                    aggfunc='sum'
                                ).fillna(0).reset_index()
                                
                                # Calculate net sentiment (positive - negative)
                                sentiment_ratio['net_sentiment'] = sentiment_ratio['positive'] - sentiment_ratio['negative']
                                
                                # Pivot to get company vs category with net sentiment values
                                net_sentiment_pivot = sentiment_ratio.pivot(
                                    index='company',
                                    columns='category',
                                    values='net_sentiment'
                                ).fillna(0)
                                
                                # Create heatmap with diverging color scale centered at 0
                                fig = px.imshow(
                                    net_sentiment_pivot,
                                    text_auto='.1f',
                                    labels=dict(x="Category", y="Company", color="Net Sentiment (%)"),
                                    color_continuous_scale='RdBu',
                                    color_continuous_midpoint=0,
                                    title='Net Sentiment (Positive - Negative) by Category and Company'
                                )
                                
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Generate dynamic topic insights based on the data
                            insights = []
                            
                            # Find categories with highest positive sentiment for each company
                            for company in compare_companies:
                                company_topics = topic_sentiment_df[(topic_sentiment_df['company'] == company) & 
                                                                (topic_sentiment_df['sentiment'] == 'positive')]
                                if not company_topics.empty:
                                    top_topic = company_topics.loc[company_topics['percentage'].idxmax()]
                                    insights.append(f"<li><b>{company}</b> shows highest positive sentiment in <b>{top_topic['category']}</b> ({top_topic['percentage']:.1f}%)</li>")
                            
                            # Find categories with highest negative sentiment for each company
                            for company in compare_companies:
                                company_topics = topic_sentiment_df[(topic_sentiment_df['company'] == company) & 
                                                                (topic_sentiment_df['sentiment'] == 'negative')]
                                if not company_topics.empty:
                                    worst_topic = company_topics.loc[company_topics['percentage'].idxmax()]
                                    insights.append(f"<li><b>{company}</b> has highest negative sentiment in <b>{worst_topic['category']}</b> ({worst_topic['percentage']:.1f}%)</li>")
                            
                            # Find categories where PineLabs differs most from others in positive sentiment
                            if 'PineLabs' in compare_companies:
                                pine_data = topic_sentiment_df[(topic_sentiment_df['company'] == 'PineLabs') & 
                                                            (topic_sentiment_df['sentiment'] == 'positive')]
                                other_companies = [c for c in compare_companies if c != 'PineLabs']
                                
                                if other_companies and not pine_data.empty:
                                    other_data = topic_sentiment_df[(topic_sentiment_df['company'].isin(other_companies)) & 
                                                                (topic_sentiment_df['sentiment'] == 'positive')]
                                    
                                    for category in pine_data['category'].unique():
                                        pine_pct = pine_data[pine_data['category'] == category]['percentage'].values[0]
                                        other_avg = other_data[other_data['category'] == category]['percentage'].mean() if not other_data[other_data['category'] == category].empty else 0
                                        
                                        diff = pine_pct - other_avg
                                        if abs(diff) >= 5:  # Only include significant differences
                                            if diff > 0:
                                                insights.append(f"<li>PineLabs has <b>{diff:.1f}%</b> higher positive sentiment in <b>{category}</b> compared to competitors</li>")
                                            else:
                                                insights.append(f"<li>PineLabs has <b>{abs(diff):.1f}%</b> lower positive sentiment in <b>{category}</b> compared to competitors</li>")
                            
                            # Topic comparison insights
                            st.markdown(f"""
                            <div style="background-color: #f0f5f2; padding: 15px; border-radius: 10px; margin-top: 10px;">
                                <h4 style="color: #345c49; margin-top: 0;">Topic and Sentiment Insights</h4>
                                <ul>
                                    {"".join(insights[:7])}  <!-- Limit to top 7 insights -->
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.info("Not enough data for topic distribution comparison.")
                    except Exception as e:
                        st.error(f"Error creating topic distribution comparison: {str(e)}")
                    
                    # Summary comparison and improvement recommendations
                    st.markdown("<h4>Competitive Analysis Summary for PineLabs</h4>", unsafe_allow_html=True)
                    
                    # Check if PineLabs exists in the data
                    if 'PineLabs' in compare_companies:
                        try:
                            # Get PineLabs data
                            pine_data = compare_data[compare_data['source'] == 'PineLabs']
                            
                            if not pine_data.empty:
                                # Get top 5 negative comments for PineLabs
                                # Using 'at' column for sorting if available, otherwise use index
                                if 'at' in pine_data.columns:
                                    pine_negative = pine_data[pine_data['sentiment'] == 'negative'].sort_values(by='at', ascending=False).head(5)
                                else:
                                    pine_negative = pine_data[pine_data['sentiment'] == 'negative'].head(5)
                                
                                # Prepare the negative comments for the improvement report
                                # Use 'review' column if available, otherwise try 'text' column
                                if 'review' in pine_negative.columns:
                                    negative_comments = pine_negative['review'].tolist()
                                elif 'text' in pine_negative.columns:
                                    negative_comments = pine_negative['text'].tolist()
                                else:
                                    negative_comments = []
                                
                                # Create SWOT analysis
                                col1, col2 = st.columns(2)
                                
                                # Dynamically find strengths and weaknesses based on the data
                                strengths = []
                                weaknesses = []
                                
                                # Check sentiment by category for PineLabs
                                if 'category' in pine_data.columns:
                                    pine_category_sentiment = pine_data.groupby('category')['sentiment'].value_counts().unstack().fillna(0)
                                    
                                    if not pine_category_sentiment.empty and 'positive' in pine_category_sentiment.columns:
                                        # Calculate positive ratio by category
                                        pine_category_sentiment['total'] = pine_category_sentiment.sum(axis=1)
                                        pine_category_sentiment['positive_ratio'] = (pine_category_sentiment['positive'] / pine_category_sentiment['total']) * 100
                                        
                                        # Sort by positive ratio
                                        pine_category_sentiment = pine_category_sentiment.sort_values(by='positive_ratio', ascending=False)
                                        
                                        # Top 3 categories as strengths
                                        for category in pine_category_sentiment.head(2).index:
                                            ratio = pine_category_sentiment.loc[category, 'positive_ratio']
                                            strengths.append(f"<li>Strong performance in <b>{category}</b> with {ratio:.1f}% positive sentiment</li>")
                                        
                                        # Bottom 3 categories as weaknesses
                                        for category in pine_category_sentiment.tail(2).index:
                                            ratio = pine_category_sentiment.loc[category, 'positive_ratio']
                                            weaknesses.append(f"<li>Lower satisfaction in <b>{category}</b> with only {ratio:.1f}% positive sentiment</li>")
                                
                                # If we couldn't get strengths/weaknesses from data, provide defaults
                                if not strengths:
                                    strengths = [
                                        "<li>Data insufficient to automatically determine strengths</li>",
                                        "<li>Please review the sentiment analysis for more insights</li>"
                                    ]
                                
                                if not weaknesses:
                                    weaknesses = [
                                        "<li>Data insufficient to automatically determine weaknesses</li>",
                                        "<li>Please review the sentiment analysis for more insights</li>"
                                    ]
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; height: 100%;">
                                        <h5 style="color: #2e7d32; margin-top: 0;">Strengths</h5>
                                        <ul>
                                            {"".join(strengths[:2])}
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; height: 100%;">
                                        <h5 style="color: #c62828; margin-top: 0;">Weaknesses</h5>
                                        <ul>
                                            {"".join(weaknesses[:2])}
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Generate improvement report if function exists and we have negative comments
                                if negative_comments:
                                    st.markdown("<h4>AI-Generated Improvement Recommendations</h4>", unsafe_allow_html=True)
                                    
                                    with st.spinner("Generating improvement recommendations..."):
                                        try:
                                            # Check if generate_improvement_report function exists in the global namespace
                                            if 'generate_improvement_report' in globals():
                                                # Call generate_improvement_report function with negative comments
                                                improvement_report = generate_improvement_report(negative_comments,groq_api_key)
                                                st.session_state.improvement_report = improvement_report

                                                # Display the improvement report
                                                st.markdown(f"""
                                                <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
                                                    {improvement_report}
                                                </div>
                                                """, unsafe_allow_html=True)
                                            else:
                                                # Fallback to displaying the negative comments
                                                st.warning("Improvement report generation function not available. Displaying negative comments instead.")
                                                st.markdown("""
                                                <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
                                                    <h4 style="margin-top: 0;">Top Negative Comments for Improvement</h4>
                                                    <p>Please review these comments to identify areas for improvement:</p>
                                                    <ul>
                                                """, unsafe_allow_html=True)
                                                
                                                for comment in negative_comments:
                                                    st.markdown(f"<li>{comment}</li>", unsafe_allow_html=True)
                                                
                                                st.markdown("</ul></div>", unsafe_allow_html=True)
                                        except Exception as e:
                                            st.error(f"Error generating improvement report: {str(e)}")
                                            
                                            # Fallback to displaying the negative comments
                                            st.markdown("""
                                            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
                                                <h4 style="margin-top: 0;">Top Negative Comments for Improvement</h4>
                                                <p>Please review these comments to identify areas for improvement:</p>
                                                <ul>
                                            """, unsafe_allow_html=True)
                                            
                                            for comment in negative_comments:
                                                st.markdown(f"<li>{comment}</li>", unsafe_allow_html=True)
                                            
                                            st.markdown("</ul></div>", unsafe_allow_html=True)
                            else:
                                st.warning("No PineLabs data available for analysis.")
                        except Exception as e:
                            st.error(f"Error generating company analysis: {str(e)}")
                    else:
                        st.warning("PineLabs is not selected for comparison. Please include PineLabs to see detailed analysis.")
                else:
                    st.warning("Please select at least one company for comparison.")
            else:
                st.warning("Required data columns (source, category) not available for company comparison.")
        
        with tab5:
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            # Raw data display
            st.markdown("""
                <div class="chart-container">
                    <h3 class="chart-title">Raw Data</h3>
                """, unsafe_allow_html=True)
            
            # Add filter functionality
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                category_filter = st.multiselect(
                    "Filter by Category",
                    options=data['category'].unique(),
                    default=list(data['category'].unique()),
                    help="Select categories to include"
                )
            
            with col2:
                source_filter = st.multiselect(
                    "Filter by Source",
                    options=data['source'].unique(),
                    default=list(data['source'].unique()),
                    help="Select sources to include"
                )
            
            with col3:
                sentiment_filter_raw = st.multiselect(
                    "Filter by Sentiment",
                    options=data['sentiment'].unique(),
                    default=list(data['sentiment'].unique()),
                    help="Select sentiment categories to include"
                )
            
            # Add download button in a new row
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Data",
                    csv,
                    "sentiment_data.csv",
                    "text/csv",
                    key='download-csv',
                    help="Download the data as CSV file"
                )
            
            # Apply filters to raw data
            filtered_raw_data = data.copy()
            
            if sentiment_filter_raw:
                filtered_raw_data = filtered_raw_data[filtered_raw_data['sentiment'].isin(sentiment_filter_raw)]
            
            if category_filter:
                filtered_raw_data = filtered_raw_data[filtered_raw_data['category'].isin(category_filter)]
            
            if source_filter:
                filtered_raw_data = filtered_raw_data[filtered_raw_data['source'].isin(source_filter)]
            
            # Display data with pagination
            if not filtered_raw_data.empty:
                # Add pagination
                items_per_page = 10
                total_pages = (len(filtered_raw_data) - 1) // items_per_page + 1
                
                col1, col2, col3 = st.columns([2, 3, 2])
                with col2:
                    page = st.number_input(
                        f"Page (1-{total_pages})",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        help="Navigate between pages of data"
                    )
                
                # Calculate slice for current page
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(filtered_raw_data))
                
                # Display page information
                st.markdown(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_raw_data)} records")
                
                # Prepare the data for display
                display_data = filtered_raw_data.iloc[start_idx:end_idx].copy()
                
                # Format the data for better display
                if 'at' in display_data.columns and pd.api.types.is_datetime64_any_dtype(display_data['at']):
                    display_data['at'] = display_data['at'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Add sentiment badges
                if 'sentiment' in display_data.columns:
                    display_data['sentiment'] = display_data['sentiment'].apply(get_sentiment_badge)
                
                # Limit text length for better display
                if 'text' in display_data.columns:
                    display_data['text'] = display_data['text'].str.slice(0, 150) + '...'
                
                # Round score to 2 decimal places
                if 'score' in display_data.columns:
                    display_data['score'] = display_data['score'].round(2)
                
                # Display the data with the sentiment column rendered as HTML
                st.write(display_data.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Navigation buttons for pagination
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                with col1:
                    if st.button("◀◀ First", disabled=(page == 1)):
                        st.session_state.page_num = 1
                        st.rerun()
                
                with col2:
                    if st.button("◀ Previous", disabled=(page == 1)):
                        st.session_state.page_num = max(1, page - 1)
                        st.rerun()
                
                with col3:
                    if st.button("Next ▶", disabled=(page == total_pages)):
                        st.session_state.page_num = min(total_pages, page + 1)
                        st.rerun()
                
                with col4:
                    if st.button("Last ▶▶", disabled=(page == total_pages)):
                        st.session_state.page_num = total_pages
                        st.rerun()
            else:
                st.info("No data matching your filters.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        # Run the app
if __name__ == "__main__":
    main()