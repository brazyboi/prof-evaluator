from openai import OpenAI
from data_models import RMPData
import os
import requests
import json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a quantitative data extractor. Analyze the following student reviews and extract the core metrics. Rate your confidence lower if there are very few reviews or conflicting opinions.
"""

def run_rmp_agent(raw_scraped_reviews: str) -> RMPData:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract metrics from these reviews: {raw_scraped_reviews}"},
        ],
        response_format=RMPData
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Failed to parse RMPData from model response")

    return parsed

def post_request(query: str, variables: dict) -> requests.Response:
    api_url = "https://www.ratemyprofessors.com/graphql"
    headers = {
        "Authorization": "Basic dGVzdDp0ZXN0",
    }

    response = requests.post(api_url, json={'query': query, 'variables': variables}, headers=headers)
    return response

def get_professor_data(professor_id: str):
    query = """
    query TeacherRatingsPageQuery($id: ID!) {
      node(id: $id) {
        ... on Teacher {
          firstName
          lastName
          avgRating
          avgDifficulty
          wouldTakeAgainPercent
          ratings(first: 20) {
            edges {
              node {
                comment
                grade
                class
                difficultyRating
                ratingTags
              }
            }
          }
        }
      }
    }
    """
    variables = {"id": professor_id}

    response = post_request(query, variables)

    if response.status_code == 200:
        return response.json()['data']['node']
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")


def get_school_id(school_name: str):
    api_url = "https://www.ratemyprofessors.com/graphql"
    query = """
    query SchoolSearchQuery($query: String!) {
      newSearch {
        schools(query: {text: $query}) {
          edges {
            node {
              id
              name
              city
              state
            }
          }
        }
      }
    }
    """

    variables = {"query": school_name}

    response = post_request(query, variables)

    return response.json()['data']['newSearch']['schools']['edges'][0]['node']['id']


def find_professor_id(prof_name: str, school_id: str):
    query = """
    query TeacherSearchQuery($query: TeacherSearchQuery!) {
      newSearch {
        teachers(query: $query) {
          edges {
            node {
              id
              firstName
              lastName
              department
              avgRating
            }
          }
        }
      }
    }
    """
    variables = {
        "query": {
            "text": prof_name,
            "schoolID": school_id
        }
    }

    response = post_request(query, variables)

    return response.json()['data']['newSearch']['teachers']['edges']





