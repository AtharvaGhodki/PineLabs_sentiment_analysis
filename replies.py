from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

def get_user_replies(twitter_usr, twitter_pass, email, past_days=7, max_scrolls=5):
    """
    Extracts replies from a user's 'with_replies' section on X (formerly Twitter) within the past specified days.

    Parameters:
    - username: str - The target user's handle (e.g., 'PineLabs').
    - twitter_usr: str - Your X username for login.
    - twitter_pass: str - Your X password for login.
    - email: str - Your email associated with the X account (used during login verification).
    - past_days: int - Number of past days to filter replies.
    - max_scrolls: int - Maximum number of times to scroll down to load more replies.

    Returns:
    - DataFrame containing the filtered replies with columns: 'Date of Tweet', 'Replying to', 'Tweet'.
    """

    # Initialize headless Firefox WebDriver
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)

    def x_login(driver):
        """Logs into X using provided credentials."""
        driver.get('https://x.com/login')
        wait = WebDriverWait(driver, 10)

        # Enter username
        user_input = wait.until(EC.presence_of_element_located((By.NAME, "text")))
        user_input.send_keys(twitter_usr)
        user_input.send_keys(Keys.RETURN)
        sleep(2)

        # Handle optional email verification
        try:
            email_input = wait.until(EC.presence_of_element_located((By.NAME, "text")))
            email_input.send_keys(email)
            email_input.send_keys(Keys.RETURN)
            sleep(2)
        except:
            pass

        # Enter password
        password_input = wait.until(EC.presence_of_element_located((By.NAME, "password")))
        password_input.send_keys(twitter_pass)
        password_input.send_keys(Keys.RETURN)
        print("Login Successful")

    try:
        # Perform login
        x_login(driver)
        sleep(2)

        # Navigate to the user's 'with_replies' section
        user_replies_url = f'https://x.com/PineLabs/with_replies'
        driver.get(user_replies_url)
        sleep(6)

        # Initialize list to store tweet data
        tweets_data = []

        for _ in range(max_scrolls):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(2)

            tweets = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')

            for t in tweets:
                try:
                    date = t.find_element(By.XPATH, './/time').get_attribute('datetime')
                    date = pd.to_datetime(date)
                except:
                    date = None

                try:
                    user = t.find_element(By.XPATH, './/div[@dir="ltr"]/span').text
                except:
                    user = None

                try:
                    replying_to = t.find_element(By.XPATH, './/div[contains(text(),"Replying to")]/a').text
                except:
                    replying_to = None

                try:
                    text = t.find_element(By.XPATH, './/div[@data-testid="tweetText"]').text
                except:
                    text = None

                if date and text:
                    tweets_data.append([date, user, replying_to, text])

            sleep(2)

        driver.quit()

        # Create DataFrame from collected data
        df = pd.DataFrame(tweets_data, columns=["Date of Tweet", "User", "Replying to", "Tweet"])

        # Convert to datetime with UTC
        df["Date of Tweet"] = pd.to_datetime(df["Date of Tweet"], utc=True)

        # Use timezone-aware end_date and start_date
        end_date = datetime.datetime.now(datetime.UTC)
        start_date = end_date - datetime.timedelta(days=past_days)

        # Filter
        df = df[df["Date of Tweet"] >= start_date]
        df = df.reset_index(drop=True)

        df = df[~df['Tweet'].str.startswith('Hi', na=False)]
        # Reshape dataframe as required
        df = df.rename(columns={"Tweet": "review", "Date of Tweet": "at"})
        df["source"] = "twitter"
        df = df[["review", "at", "source"]]

        return df

    except Exception as e:
        driver.quit()
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=["review", "at", "source"])

# Access environment variables
X_api = os.getenv("X_API_KEY")

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