from openai import OpenAI
from data_models import RMPData, SchoolCandidate, SchoolSelection
import os
import requests
import json
from typing import Any

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a quantitative data extractor. Analyze the following student reviews and extract the core metrics. Rate your confidence lower if there are very few reviews or conflicting opinions.
"""

SCHOOL_DISAMBIGUATION_PROMPT = """
You are selecting the best matching school from RateMyProfessors search candidates.
Pick the school that most likely matches the user's intent.

Rules:
- Nicknames and abbreviations should map to formal names when clearly related.
- Prefer exact semantic match over lexical overlap.
- If multiple schools are plausible, still pick the best one but reduce confidence.
- Confidence must be between 0.0 and 1.0.
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

def _extract_json(response: requests.Response) -> dict[str, Any]:
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    return response.json()

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

    payload = _extract_json(response)
    return payload['data']['node']


def get_school_candidates(school_query: str, limit: int = 5) -> list[SchoolCandidate]:
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

    variables = {"query": school_query}

    response = post_request(query, variables)
    payload = _extract_json(response)
    edges = payload['data']['newSearch']['schools']['edges']
    candidates = []
    for edge in edges[:limit]:
        node = edge['node']
        candidates.append(
            SchoolCandidate(
                id=node['id'],
                name=node['name'],
                city=node.get('city') or "",
                state=node.get('state') or "",
            )
        )
    return candidates

def get_school_id(school_name: str) -> list[SchoolCandidate]:
    """
    Backwards-compatible name kept for callers, now returns top school candidates.
    """
    return get_school_candidates(school_name, limit=5)

def pick_best_school(user_input: str, candidates: list[SchoolCandidate]) -> SchoolSelection:
    if not candidates:
        raise ValueError("No school candidates available for selection")

    candidate_json = json.dumps([candidate.model_dump() for candidate in candidates], indent=2)
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SCHOOL_DISAMBIGUATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f'The user is looking for: "{user_input}"\n'
                    f"Candidates:\n{candidate_json}\n\n"
                    "Return the selected school id, confidence, and concise reasoning."
                ),
            },
        ],
        response_format=SchoolSelection
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Failed to parse SchoolSelection from model response")
    return parsed

def resolve_school_query(school_query: str) -> dict[str, Any]:
    candidates = get_school_candidates(school_query, limit=5)
    if not candidates:
        return {
            "status": "not_found",
            "message": f'No schools found for "{school_query}".',
            "candidates": [],
        }

    selection = pick_best_school(school_query, candidates)

    if selection.confidence_score > 0.8:
        return {
            "status": "high_confidence",
            "selected_id": selection.selected_id,
            "confidence_score": selection.confidence_score,
            "reasoning": selection.reasoning,
            "candidates": [candidate.model_dump() for candidate in candidates],
        }

    if selection.confidence_score >= 0.5:
        matched = next((candidate for candidate in candidates if candidate.id == selection.selected_id), None)
        matched_name = matched.name if matched else selection.selected_id
        return {
            "status": "medium_confidence",
            "selected_id": selection.selected_id,
            "confidence_score": selection.confidence_score,
            "reasoning": selection.reasoning,
            "log": f"Found '{matched_name}'. Proceeding with evaluation.",
            "candidates": [candidate.model_dump() for candidate in candidates],
        }

    readable = [f"{c.name} ({c.city}, {c.state})" for c in candidates[:3]]
    return {
        "status": "low_confidence",
        "confidence_score": selection.confidence_score,
        "reasoning": selection.reasoning,
        "candidates": [candidate.model_dump() for candidate in candidates],
        "message": (
            f'I found {len(readable)} schools that might match "{school_query}": '
            + ", ".join(readable)
            + ". Which school did you mean?"
        ),
    }


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
    payload = _extract_json(response)
    return payload['data']['newSearch']['teachers']['edges']




