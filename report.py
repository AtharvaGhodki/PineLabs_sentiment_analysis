from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# def categorize_comment_1(comment):
#     prompt = f"""
# You are an expert at categorizing user feedback into specific problem categories.

# Given a category list: [transaction-success, settlement, servicing, onboarding, pricing, device-issue], select the most appropriate category for the provided comment. Your response should only contain the category name and nothing else.

# Sample Format:
# Sample Comment: I was charged twice but didn't receive confirmation.
# Response: transaction-success

# Now, process the following comment accordingly.

# Comment: {comment}
# """

#     response = openai.ChatCompletion.create(
#         model="gpt-4o",  # or "gpt-3.5-turbo"
#         messages=[
#             {"role": "system", "content": "You are an expert in categorizing feedback."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0,
#         max_tokens=50,
#     )

#     category = response.choices[0].message["content"].strip()

#     return {
#         "predicted_category": category
#     }

def categorize_comment(comment,groq_api):
    template = """
You are an expert at categorizing user feedback into specific problem categories.

Given a category list: [transaction-success, settlement, servicing, onboarding, pricing, device-issue], select the most appropriate category for the provided comment. Your response should only contain the category name and nothing else.

Sample Format:
Sample Comment: I was charged twice but didn't receive confirmation.
Response: transaction success

Now, process the following comment accordingly.

Comment: {comment}
"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(
        model="llama3-70b-8192",  # You can change to llama-3-8b if needed
        api_key=groq_api,
        temperature=0,
        max_tokens=50,
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"comment": comment})

    return {
        "predicted_category": result.strip()
    }


def generate_improvement_report(negative_comments,groq_api):
    template = """
    You are an expert AI specializing in customer feedback analysis and product improvement.
    Generate a **concise and professional improvement report** based on the top 5 negative comments provided below.
    
    ### **Negative Feedback Given**
    {negative_comments}
    
    ### **Response Expectations**
    - Summarize the key issues found in the comments.
    - Suggest actionable improvements.
    - Keep the response **brief (100-150 words)**.
    - Maintain a professional but constructive tone.
    
    **Output Example**:
    "Customers frequently complain about slow transaction speeds and poor customer support. To address this, Pine Labs should optimize backend processing for faster payments and enhance support response times by introducing AI-based chat assistance..."
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api,
        temperature=0.7,
        max_tokens=400
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"negative_comments": "\n".join(negative_comments)})
    
    return response



# print(categorize_comment_1("@AnqFinance after 30 days I will complaint on RBI ombudsman for your non service response."))