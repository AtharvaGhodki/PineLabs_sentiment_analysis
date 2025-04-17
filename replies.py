from google_play_scraper import reviews_all
from app_store_scraper import AppStore
from time import sleep
import pandas as pd
import datetime
from sentiment import predict_sentiment
from report import categorize_comment
import os
import requests
from dotenv import load_dotenv

def get_playstore_reviews(past_days,source="PineLabs"):
    if source=="PineLabs":
        app_id="com.pinelabs.pinelabsone"
    elif source=="Razorpay":
        app_id ="com.razorpay.payments.app"
    else:
        app_id ="net.one97.paytm"
    data = reviews_all(app_id, lang="en", country="us")
    df = pd.DataFrame(data)
    df["source"] = "Play Store"
    df=df[["content","at","source"]].rename(columns={"content": "review"})
    # Make sure 'at' column is timezone-aware
    df["at"] = pd.to_datetime(df["at"], utc=True)
    # Filter reviews based on past_days
    end_date = datetime.datetime.now(datetime.UTC)
    start_date = end_date - datetime.timedelta(days=past_days)
    df = df[df["at"] >= start_date]
    return df

def get_apple_store_reviews(past_days,source="PineLabs"):
    if source=="PineLabs":
        app_id = 6444654068  # Replace with the correct Pine Labs App ID
        app_name = "Pine Labs" # You need to provide the app name here
    else:
        app_id = 1497250144
        app_name="Razorpay"   
    app = AppStore(country="in", app_name=app_name, app_id=app_id) # Pass app_name to AppStore
    app.review(how_many=50)
    df = pd.DataFrame(app.reviews)
    df["source"] = "Apple Store"
    df = df[["review", "date", "source"]]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(by="date", ascending=False).reset_index(drop=True)
    # Filter reviews based on past_days
    end_date = datetime.datetime.now(datetime.UTC)
    start_date = end_date - datetime.timedelta(days=past_days)
    df = df[df["date"] >= start_date]
    df = df.rename(columns={"date": "at"})
    return df

# Access environment variables
X_api = "6d3b9ae245474bc09b0f121932a19234"

def get_twitter_comments(past_days,source):
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=past_days)  # Ensure timezone-aware  
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    all_reviews = []
    cursor = None  # Initialize cursor for pagination

    while True:
        url = "https://api.twitterapi.io/twitter/tweet/advanced_search"  # Verify API URL
        
        querystring = {
            "queryType": "Latest",
            "query": f"to:{source} since:{start_date_str} until:{end_date_str}"
        }
        if cursor:
            querystring["cursor"] = cursor  # Handle pagination

        headers = {"X-API-Key":X_api}

        # API request to fetch replies
        response = requests.get(url, headers=headers, params=querystring)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break  # Stop execution if API request fails

        response_data = response.json()

        # Check if the response contains tweets
        if "tweets" in response_data:
            for tweet in response_data['tweets']:
                review = tweet.get('text', '')
                created_at = tweet.get('createdAt', '')

                # Convert created_at to datetime (if provided)
                if created_at:
                    created_at = datetime.datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y").replace(tzinfo=datetime.timezone.utc)
                    created_at_str = created_at.strftime('%Y-%m-%d %H:%M:%S%z')
                else:
                    created_at_str = None

                all_reviews.append({
                    'review': review,
                    'at': created_at_str,
                    'source': source,
                })
        
        # Handle pagination
        cursor = response_data.get('next_cursor')
        if not cursor:
            break  # Exit loop if no more pages

    # Convert the list of reviews to a DataFrame
    df = pd.DataFrame(all_reviews)
    return df    
    
def get_all_replies_with_sentiment(past_days=50):
    """Fetch or load replies, apply sentiment analysis, and return the final DataFrame."""
    
    if os.path.exists('all_replies_with_sentiment.csv'):
        # Load existing data
        df_combined = pd.read_csv('all_replies_with_sentiment.csv')
        return df_combined
    else:
        # Fetch new data
        df1 = get_twitter_comments(past_days,"PineLabs")
        df2 = get_twitter_comments(past_days,"Razorpay")
        df3 = get_twitter_comments(past_days,"Paytm")
        #df2 = get_playstore_reviews(past_days,source)
        #df3 = get_apple_store_reviews(past_days,source)
        
        # Combine and save
        df_combined = pd.concat([df1,df2,df3], ignore_index=True)
    
    # Apply sentiment analysis
    df_combined[['sentiment', 'score']] = df_combined['review'].apply(lambda x: pd.Series(predict_sentiment(x)))
    df_combined[['category']] = df_combined['review'].apply(
    lambda x: pd.Series(categorize_comment(x)['predicted_category'])
)
    # Save final dataset
    #df_combined.to_csv('all_replies_with_sentiment.csv', index=False)
    
    return df_combined    