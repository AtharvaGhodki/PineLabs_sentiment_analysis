import time
import pandas as pd
import datetime
from sentiment import predict_sentiment
from report import categorize_comment
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# def get_playstore_reviews(past_days,source="PineLabs"):
#     if source=="PineLabs":
#         app_id="com.pinelabs.pinelabsone"
#     elif source=="Razorpay":
#         app_id ="com.razorpay.payments.app"
#     else:
#         app_id ="net.one97.paytm"
#     data = reviews_all(app_id, lang="en", country="us")
#     df = pd.DataFrame(data)
#     df["source"] = "Play Store"
#     df=df[["content","at","source"]].rename(columns={"content": "review"})
#     # Make sure 'at' column is timezone-aware
#     df["at"] = pd.to_datetime(df["at"], utc=True)
#     # Filter reviews based on past_days
#     end_date = datetime.datetime.now(datetime.UTC)
#     start_date = end_date - datetime.timedelta(days=past_days)
#     df = df[df["at"] >= start_date]
#     return df

# def get_apple_store_reviews(past_days,source="PineLabs"):
#     if source=="PineLabs":
#         app_id = 6444654068  # Replace with the correct Pine Labs App ID
#         app_name = "Pine Labs" # You need to provide the app name here
#     else:
#         app_id = 1497250144
#         app_name="Razorpay"   
#     app = AppStore(country="in", app_name=app_name, app_id=app_id) # Pass app_name to AppStore
#     app.review(how_many=50)
#     df = pd.DataFrame(app.reviews)
#     df["source"] = "Apple Store"
#     df = df[["review", "date", "source"]]
#     df["date"] = pd.to_datetime(df["date"], utc=True)
#     df = df.sort_values(by="date", ascending=False).reset_index(drop=True)
#     # Filter reviews based on past_days
#     end_date = datetime.datetime.now(datetime.UTC)
#     start_date = end_date - datetime.timedelta(days=past_days)
#     df = df[df["date"] >= start_date]
#     df = df.rename(columns={"date": "at"})
#     return df


def get_twitter_comments(past_days,source,X_api):
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
    
# def get_all_replies_with_sentiment(past_days=50):
#     """Fetch or load replies, apply sentiment analysis, and return the final DataFrame."""
    
#     if os.path.exists('all_replies_with_sentiment.csv'):
#         # Load existing data
#         df_combined = pd.read_csv('all_replies_with_sentiment.csv')
#         return df_combined
#     else:
#         # Fetch new data
#         df1 = get_twitter_comments(past_days,"PineLabs")
#         df2=  get_twitter_comments(past_days,"pinelabsonline")
#         df3 = get_twitter_comments(past_days,"Razorpay")
#         df4 = get_twitter_comments(past_days,"Paytm")
#         #df2 = get_playstore_reviews(past_days,source)
#         #df3 = get_apple_store_reviews(past_days,source)
        
#         # Combine and save
#         df_combined = pd.concat([df1,df2,df3,df4], ignore_index=True)
#         df_combined['source'] = df_combined['source'].replace('pinelabsonline', 'PineLabs')
    
#     # Apply sentiment analysis
#     df_combined[['sentiment', 'score']] = df_combined['review'].apply(lambda x: pd.Series(predict_sentiment(x)))
#     df_combined[['category']] = df_combined['review'].apply(
#     lambda x: pd.Series(categorize_comment(x)['predicted_category'])
# )
#     # Save final dataset
#     #df_combined.to_csv('all_replies_with_sentiment.csv', index=False)
    
#     return df_combined    

def is_cache_stale(file_path, max_age_hours=12):
    if not os.path.exists(file_path):
        return True
    last_modified_time = os.path.getmtime(file_path)
    age_hours = (time.time() - last_modified_time) / 3600  # convert to hours
    return age_hours > max_age_hours

def get_all_replies_with_sentiment(X_api, groq_api, past_days=7, max_cache_age_hours=24):
    """Fetch or load replies, apply sentiment analysis, and return the final DataFrame."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"replies_{past_days}days.csv")
    
    if os.path.exists(cache_file) and not is_cache_stale(cache_file, max_cache_age_hours):
        df_combined = pd.read_csv(cache_file, parse_dates=['at'])
        return df_combined

    # Fetch fresh data from Twitter API
    df1 = get_twitter_comments(past_days, "PineLabs", X_api)
    df2 = get_twitter_comments(past_days, "pinelabsonline", X_api)
    df3 = get_twitter_comments(past_days, "Razorpay", X_api)
    df4 = get_twitter_comments(past_days, "Paytm", X_api)

    # Combine and normalize
    df_combined = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df_combined['source'] = df_combined['source'].replace('pinelabsonline', 'PineLabs')

    # Apply sentiment & categorization
    df_combined[['sentiment', 'score']] = df_combined['review'].apply(lambda x: pd.Series(predict_sentiment(x)))
    df_combined[['category']] = df_combined['review'].apply(
        lambda x: pd.Series(categorize_comment(x,groq_api)['predicted_category'])
    )

    # Save to cache
    df_combined.to_csv(cache_file, index=False)

    return df_combined